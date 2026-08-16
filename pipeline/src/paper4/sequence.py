from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper4.features import static_columns
from paper4.metrics import evaluate_predictions
from paper4.models_tabular import inverse_target, target_values
from paper4.models_torch import TorchModelFactory, require_torch


class Standardizer:
    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)


def _sequence_indices(df: pd.DataFrame, seq_len: int, split: str) -> list[int]:
    idxs = []
    for _, g in df.groupby("ID", sort=False):
        positions = list(g.index)
        for loc, idx in enumerate(positions):
            if not df.loc[idx, "split"] == split:
                continue
            if loc + 1 >= seq_len:
                window = positions[loc - seq_len + 1 : loc + 1]
                if len(window) == seq_len:
                    idxs.append(idx)
    return idxs


class _WindowView:
    """Lazy window stack: realizes (n, seq_len, n_dyn) batches on demand to avoid materializing the full eval tensor."""

    def __init__(self, dyn_values: np.ndarray, end_arr: np.ndarray, seq_len: int):
        self._dyn = dyn_values
        self._ends = end_arr
        self._seq_len = int(seq_len)
        self._offsets = np.arange(seq_len - 1, -1, -1, dtype=np.int64)
        self.shape = (int(end_arr.shape[0]), int(seq_len), int(dyn_values.shape[1]))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        ends = self._ends[key]
        if np.ndim(ends) == 0:
            ends = np.asarray([ends], dtype=np.int64)
        idx = ends[:, None] - self._offsets[None, :]
        return self._dyn[idx]


def build_sequence_arrays(frame: pd.DataFrame, feature_cols: list[str], cfg: dict):
    seq_len = int(cfg["sequence_models"]["sequence_length"])
    seed = int(cfg["sequence_models"].get("random_seed", 42))
    s_cols = static_columns(cfg, frame)
    dynamic_cols = [c for c in feature_cols if c not in s_cols]
    static_cols = [c for c in s_cols if c in frame.columns]

    df = frame.sort_values(["ID", "date"]).reset_index(drop=True)
    train_rows = df[df["split"].eq("train")]
    dyn_scaler = Standardizer().fit(train_rows[dynamic_cols].to_numpy(dtype=float))
    stat_scaler = Standardizer().fit(train_rows[static_cols].to_numpy(dtype=float)) if static_cols else None
    dyn_values = dyn_scaler.transform(df[dynamic_cols].to_numpy(dtype=float)).astype(np.float32, copy=False)
    if static_cols:
        static_values = stat_scaler.transform(df[static_cols].to_numpy(dtype=float)).astype(np.float32, copy=False)
    else:
        static_values = np.zeros((len(df), 1), dtype=np.float32)
    y_values = target_values(df, cfg).astype(np.float32, copy=False)

    # Kratzert et al. 2019 basin-normalized NSE loss needs per-sample basin std on
    # the *training* portion of the target. We compute std in transformed-target
    # space (matches the loss target). Floor with 1e-3 so a degenerate basin
    # cannot produce a divide-by-zero.
    train_target_by_id = (
        df[df["split"].eq("train")]
          .assign(_y=y_values[df["split"].eq("train").to_numpy()])
          .groupby("ID")["_y"].std(ddof=0)
          .fillna(0.0)
          .clip(lower=1e-3)
          .to_dict()
    )
    y_std_per_row = df["ID"].map(train_target_by_id).fillna(1.0).to_numpy(dtype=np.float32)
    meta_cols = ["ID", "date", "YYYY", "MM", "DD", "DOY", "water_year", "split", "q_mm_day", "prec"]

    eval_splits = cfg.get("evaluation", {}).get("prediction_splits", ["val", "test", "spatial_test"])
    splits = ["train"] + [s for s in eval_splits if s != "train"]
    splits = list(dict.fromkeys(splits))

    arrays = {}
    for split in splits:
        end_idxs = _sequence_indices(df, seq_len, split)
        limit_key = "sample_limit_train" if split == "train" else "sample_limit_eval"
        limit = cfg["sequence_models"].get(limit_key)
        if limit and len(end_idxs) > int(limit):
            rng = np.random.default_rng(seed)
            end_idxs = sorted(rng.choice(end_idxs, size=int(limit), replace=False).tolist())
        if end_idxs:
            end_arr = np.asarray(end_idxs, dtype=np.int64)
            arrays[split] = {
                "x_dyn": _WindowView(dyn_values, end_arr, seq_len),
                "x_static": static_values[end_arr],
                "y": y_values[end_arr],
                "y_std": y_std_per_row[end_arr],
                "meta": df.loc[end_arr, meta_cols].reset_index(drop=True),
            }
    return arrays, dynamic_cols, static_cols


def _batch_to_device(arr: dict, idx, device, torch):
    """Slice a batch via either a contiguous slice or an index array.

    `idx` may be a slice (e.g. range(start, stop)) or a numpy array of integer
    indices — used by the per-epoch shuffle path.
    """
    if isinstance(idx, slice):
        sl = idx
    elif isinstance(idx, range):
        sl = slice(idx.start, idx.stop, idx.step or 1)
    else:
        sl = np.asarray(idx, dtype=np.int64)
    x_dyn_batch = arr["x_dyn"][sl]
    x_dyn = torch.from_numpy(np.ascontiguousarray(x_dyn_batch)).to(device)
    x_static = torch.from_numpy(np.ascontiguousarray(arr["x_static"][sl])).to(device)
    y = torch.from_numpy(np.ascontiguousarray(arr["y"][sl])).to(device)
    y_std = arr.get("y_std")
    y_std_t = (
        torch.from_numpy(np.ascontiguousarray(y_std[sl])).to(device)
        if y_std is not None else None
    )
    return x_dyn, x_static, y, y_std_t


def _apply_loss(loss_fn, pred, y, y_std, kind: str):
    if kind == "nse" and y_std is not None:
        # Kratzert et al. 2019 basin-normalized NSE loss
        eps = 0.1
        weight = 1.0 / (y_std + eps) ** 2
        return ((pred - y) ** 2 * weight).mean()
    return loss_fn(pred, y)


def _batched_loss(model, arr: dict, batch_size: int, loss_fn, device, torch,
                  loss_kind: str = "mse", limit: int | None = None,
                  seed: int = 42) -> float:
    total_loss = 0.0
    total_n = 0
    model.eval()
    with torch.no_grad():
        n = arr["x_dyn"].shape[0]
        if limit is not None and 0 < int(limit) < n:
            rng = np.random.default_rng(seed)
            order = np.sort(rng.choice(n, size=int(limit), replace=False))
            batches = (order[start : start + batch_size] for start in range(0, len(order), batch_size))
        else:
            batches = (slice(start, min(start + batch_size, n)) for start in range(0, n, batch_size))
        for batch_idx in batches:
            x_dyn, x_static, y, y_std = _batch_to_device(arr, batch_idx, device, torch)
            pred = model(x_dyn, x_static)
            loss = _apply_loss(loss_fn, pred, y, y_std, loss_kind)
            n_batch = len(batch_idx) if not isinstance(batch_idx, slice) else batch_idx.stop - batch_idx.start
            total_loss += float(loss.detach().cpu()) * n_batch
            total_n += n_batch
    model.train()
    return total_loss / max(total_n, 1)


def train_sequence_model(frame: pd.DataFrame, feature_cols: list[str], cfg: dict, model_name: str, out_dir: Path):
    torch, nn = require_torch()
    seed = int(cfg["sequence_models"].get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg["sequence_models"].get("torch_threads", 1)))
    print(f"[sequence:{model_name}] building sequence arrays", flush=True)
    arrays, dynamic_cols, static_cols = build_sequence_arrays(frame, feature_cols, cfg)
    if "train" not in arrays:
        raise RuntimeError("No sequence training samples available. Reduce sequence_length or adjust date splits.")
    print(f"[sequence:{model_name}] samples " + ", ".join(f"{k}={v['x_dyn'].shape[0]}" for k, v in arrays.items()), flush=True)

    requested_device = cfg["sequence_models"].get("device", "cpu")
    device_by_model = cfg["sequence_models"].get("device_by_model", {})
    requested_device = device_by_model.get(model_name, requested_device)
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = torch.cuda.is_available()
    python_machine = platform.machine()
    use_mps_auto = mps_available and python_machine != "x86_64"
    print(
        f"[sequence:{model_name}] requested_device={requested_device} "
        f"cuda_available={cuda_available} mps_available={mps_available} python_machine={python_machine}",
        flush=True,
    )
    if requested_device == "auto":
        if cuda_available:
            device = torch.device("cuda")
        elif use_mps_auto:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    n_dyn = arrays["train"]["x_dyn"].shape[-1]
    n_static = arrays["train"]["x_static"].shape[-1]
    if model_name == "xlstm":
        model = TorchModelFactory.xlstm(n_dyn, n_static, cfg)
    elif model_name == "transformer":
        model = TorchModelFactory.transformer(n_dyn, n_static, cfg)
    else:
        raise ValueError(model_name)
    model.to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"[sequence:{model_name}] device={device} parameters={n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["sequence_models"]["learning_rate"]), weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loss_kind = str(cfg["sequence_models"].get("loss", "mse")).lower()
    grad_clip = cfg["sequence_models"].get("grad_clip", 1.0)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    batch_size = int(cfg["sequence_models"]["batch_size"])
    epochs_by_model = cfg["sequence_models"].get("epochs_by_model", {})
    default_epochs = int(cfg["sequence_models"].get("epochs", max(epochs_by_model.values(), default=12)))
    epochs = int(epochs_by_model.get(model_name, default_epochs))
    patience_cfg = cfg["sequence_models"].get("early_stopping_patience")
    patience = int(patience_cfg) if patience_cfg is not None else None
    use_cosine = bool(cfg["sequence_models"].get("cosine_schedule", True))
    scheduler = None
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(cfg["sequence_models"]["learning_rate"]) * 0.05)

    train_arr = arrays["train"]
    val_arr = arrays.get("val")
    n = train_arr["x_dyn"].shape[0]
    score_limit_cfg = cfg["sequence_models"].get("sample_limit_score")
    score_limit = int(score_limit_cfg) if score_limit_cfg is not None else None
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sample_counts = {k: int(v["x_dyn"].shape[0]) for k, v in arrays.items()}
    setup = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "xLSTM-style gated recurrent memory" if model_name == "xlstm" else "Transformer self-attention encoder",
        "literature_starting_point": (
            "Regional neural hydrology setup following the Kratzert et al. LSTM line: "
            "shared regional model across basins, static attributes, long lookback, "
            "per-epoch shuffled sequence batches, validation selection, and early stopping."
        ),
        "device": str(device),
        "requested_device": str(requested_device),
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "python_machine": python_machine,
        "parameters": n_params,
        "sequence_length": int(cfg["sequence_models"]["sequence_length"]),
        "dynamic_feature_count": len(dynamic_cols),
        "static_feature_count": len(static_cols),
        "dynamic_features": dynamic_cols,
        "static_features": static_cols,
        "sample_counts": sample_counts,
        "sample_limit_score": score_limit,
        "batch_size": batch_size,
        "epochs_max": epochs,
        "early_stopping_patience": patience,
        "learning_rate_initial": float(cfg["sequence_models"]["learning_rate"]),
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "loss": loss_kind,
        "grad_clip": grad_clip,
        "cosine_schedule": use_cosine,
        "target": cfg.get("target", {}),
        "selection_split": "val" if val_arr is not None else "train",
    }
    (tables_dir / f"sequence_model_setup_{model_name}.json").write_text(
        json.dumps(setup, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[sequence:{model_name}] tensors ready n={n} batch={batch_size} epochs={epochs} "
        f"loss={loss_kind} grad_clip={grad_clip} cosine={use_cosine} seed={seed}",
        flush=True,
    )
    model.train()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    rng_shuffle = np.random.default_rng(seed)
    history_rows: list[dict] = []
    history_path = tables_dir / f"sequence_training_history_{model_name}.csv"
    last_epoch = 0
    stopped_early = False
    for epoch in range(epochs):
        last_epoch = epoch + 1
        # Per-epoch random permutation — Kratzert et al. 2019 standard.
        perm = rng_shuffle.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch_idx = perm[start:stop]
            x_dyn, x_static, y, y_std = _batch_to_device(train_arr, batch_idx, device, torch)
            opt.zero_grad()
            pred = model(x_dyn, x_static)
            loss = _apply_loss(loss_fn, pred, y, y_std, loss_kind)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        train_loss = float(np.mean(losses))
        score_loss = (
            _batched_loss(
                model,
                val_arr,
                batch_size,
                loss_fn,
                device,
                torch,
                loss_kind,
                limit=score_limit,
                seed=seed + epoch + 1,
            )
            if val_arr else train_loss
        )
        if score_loss < best_loss - 1e-6:
            best_loss = score_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        current_lr = opt.param_groups[0]["lr"]
        row = {
            "model": model_name,
            "epoch": epoch + 1,
            "epochs_max": epochs,
            "learning_rate": float(current_lr),
            "train_loss": train_loss,
            "val_loss": score_loss,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "stale_epochs": stale_epochs,
            "batch_size": batch_size,
            "sample_count_train": n,
            "sample_count_val": int(val_arr["x_dyn"].shape[0]) if val_arr is not None else 0,
            "loss": loss_kind,
            "grad_clip": grad_clip,
            "cosine_schedule": use_cosine,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(
            f"[sequence:{model_name}] epoch={epoch+1}/{epochs} lr={current_lr:.2e} "
            f"train_loss={train_loss:.5f} val_loss={score_loss:.5f} best_epoch={best_epoch}",
            flush=True,
        )
        if patience is not None and stale_epochs >= patience:
            print(f"[sequence:{model_name}] early stopping after {stale_epochs} stale epochs", flush=True)
            stopped_early = True
            break

    print(f"[sequence:{model_name}] best validation epoch={best_epoch} val_loss={best_loss:.5f}", flush=True)
    setup.update({
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "stopped_early": stopped_early,
        "stop_reason": "early_stopping" if stopped_early else "max_epochs",
    })
    (tables_dir / f"sequence_model_setup_{model_name}.json").write_text(
        json.dumps(setup, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    combined_path = tables_dir / "sequence_training_history.csv"
    old_history = pd.read_csv(combined_path) if combined_path.exists() else pd.DataFrame()
    if not old_history.empty and "model" in old_history.columns:
        old_history = old_history[old_history["model"] != model_name]
    pd.concat([old_history, pd.DataFrame(history_rows)], ignore_index=True).to_csv(combined_path, index=False)

    setup_summary_path = tables_dir / "model_setup_summary.json"
    setup_summary = json.loads(setup_summary_path.read_text(encoding="utf-8")) if setup_summary_path.exists() else {}
    setup_summary.setdefault("sequence_models", {})[model_name] = setup
    setup_summary_path.write_text(json.dumps(setup_summary, indent=2, sort_keys=True), encoding="utf-8")
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_frames = []
    model.eval()
    print(f"[sequence:{model_name}] predicting", flush=True)
    with torch.no_grad():
        for split, arr in arrays.items():
            preds = []
            n_split = arr["x_dyn"].shape[0]
            for start in range(0, n_split, batch_size):
                stop = min(start + batch_size, n_split)
                xd, xs, _, _ = _batch_to_device(arr, slice(start, stop), device, torch)
                out = model(xd, xs)
                preds.append(out.detach().cpu().numpy())
            yhat = inverse_target(np.concatenate(preds), cfg)
            meta = arr["meta"].copy()
            meta["model"] = model_name
            meta["q_pred_mm_day"] = yhat
            pred_frames.append(meta)

    preds = pd.concat(pred_frames, ignore_index=True)
    if cfg.get("artifacts", {}).get("save_models", False):
        print(f"[sequence:{model_name}] saving checkpoint", flush=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": model_name,
                "dynamic_cols": dynamic_cols,
                "static_cols": static_cols,
                "config": cfg,
            },
            out_dir / "models" / f"{model_name}.pt",
        )
    else:
        print(f"[sequence:{model_name}] checkpoint skipped artifacts.save_models=false", flush=True)
    print(f"[sequence:{model_name}] evaluating", flush=True)
    metrics, sigs = evaluate_predictions(preds)
    return preds, metrics, sigs
