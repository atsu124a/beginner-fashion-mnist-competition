# uv run src/train.py

from __future__ import annotations

import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from load_fashion_mnist import load_train_data
from network import TorchEnsembleModel, TorchFashionCNN, TorchFastCNN, TorchWideResNet

OUTPUT_PATH = Path("sample_weight.pkl")
TRAIN_META_STACKER = False
META_BASE_WEIGHT_PATHS = (
    Path("sample_weight_0.956600_backup.pkl"),
    Path("sample_weight_0.956200_6models_backup.pkl"),
)
META_EPOCHS = 1600
META_LR = 1.0e-3
META_WEIGHT_DECAY = 2.0e-4
ARCH = "wrn"
EPOCHS = 55
SEEDS = (4242,)
BATCH_SIZE = 512
LEARNING_RATE = 1.2e-3
MIN_LR = 1.5e-5
WEIGHT_DECAY = 5.0e-4
WIDTH = 64
HIDDEN_SIZE = 512
WRN_DEPTH = 16
WRN_WIDEN_FACTOR = 4
DROPOUT = 0.10
LABEL_SMOOTHING = 0.03
MIXUP_ALPHA = 0.10
EMA_DECAY = 0.998
TRAIN_ON_FULL_DATA = True
RESUME_EXISTING = False
FINETUNE_EXISTING = False
FINETUNE_EPOCHS = 8
FINETUNE_LR = 2.0e-4
USE_RANDOM_AFFINE = False
USE_SWA = True
SWA_START_EPOCH = 26
OPTIMIZE_ENSEMBLE_WEIGHTS = True
GPU_TEMP_LIMIT_C = 78
GPU_COOLDOWN_SECONDS = 45
BATCH_SLEEP_SECONDS = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
torch.set_num_threads(4)


def get_gpu_temperature() -> int | None:
    if DEVICE.type != "cuda":
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return int(first)
    except ValueError:
        return None


def cool_down_if_needed() -> None:
    temp = get_gpu_temperature()
    if temp is not None and temp >= GPU_TEMP_LIMIT_C:
        print(
            f"GPU temp {temp}C >= {GPU_TEMP_LIMIT_C}C; cooling down for {GPU_COOLDOWN_SECONDS}s",
            flush=True,
        )
        time.sleep(GPU_COOLDOWN_SECONDS)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    flip_mask = torch.rand(out.shape[0], device=out.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=(3,))

    padded = F.pad(out, (2, 2, 2, 2))
    crop_y = torch.randint(0, 5, (out.shape[0],), device=out.device)
    crop_x = torch.randint(0, 5, (out.shape[0],), device=out.device)
    shifted = torch.empty_like(out)
    for yy in range(5):
        for xx in range(5):
            mask = (crop_y == yy) & (crop_x == xx)
            if torch.any(mask):
                shifted[mask] = padded[mask, :, yy : yy + 28, xx : xx + 28]
    out = shifted

    if USE_RANDOM_AFFINE:
        angles = (torch.rand(out.shape[0], device=out.device) * 2.0 - 1.0) * (8.0 * np.pi / 180.0)
        scales = 1.0 + (torch.rand(out.shape[0], device=out.device) * 2.0 - 1.0) * 0.06
        theta = torch.zeros(out.shape[0], 2, 3, device=out.device, dtype=out.dtype)
        cos = torch.cos(angles) / scales
        sin = torch.sin(angles) / scales
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        grid = F.affine_grid(theta, out.shape, align_corners=False)
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    erase_indices = torch.nonzero(torch.rand(out.shape[0], device=out.device) < 0.20, as_tuple=False).flatten()
    for index in erase_indices:
        size = int(torch.randint(4, 9, (), device=out.device).item())
        y0 = int(torch.randint(0, 29 - size, (), device=out.device).item())
        x0 = int(torch.randint(0, 29 - size, (), device=out.device).item())
        out[index, :, y0 : y0 + size, x0 : x0 + size] = 0.0

    if torch.rand((), device=out.device) < 0.35:
        out = (out + torch.randn_like(out) * 0.015).clamp_(0.0, 1.0)
    return out


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if MIXUP_ALPHA <= 0:
        return x, y, y, 1.0
    beta = torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA)
    lam = float(beta.sample().item())
    index = torch.randperm(x.shape[0], device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mean: float, std: float) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, x.shape[0], 1024):
        xb = x[start : start + 1024].to(DEVICE, non_blocking=True)
        xb = xb.contiguous(memory_format=torch.channels_last)
        xb = (xb - mean) / std
        logits = model(xb)
        pred = logits.argmax(dim=1)
        yb = y[start : start + 1024].to(DEVICE, non_blocking=True)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / total


def update_ema(ema_params: dict[str, torch.Tensor], model: nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            ema_params[name].mul_(EMA_DECAY).add_(param.detach(), alpha=1.0 - EMA_DECAY)


def copy_model_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def update_swa(
    swa_params: dict[str, torch.Tensor] | None,
    model: nn.Module,
    count: int,
) -> tuple[dict[str, torch.Tensor], int]:
    current = copy_model_params(model)
    if swa_params is None:
        return current, 1
    next_count = count + 1
    with torch.no_grad():
        for name, param in current.items():
            swa_params[name].mul_(count / next_count).add_(param, alpha=1.0 / next_count)
    return swa_params, next_count


def apply_ema(ema_params: dict[str, torch.Tensor], model: nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(ema_params[name])


@torch.no_grad()
def refresh_batch_norm(model: nn.Module, x: torch.Tensor, mean: float, std: float) -> None:
    momenta: dict[nn.Module, float | None] = {}
    dropout_ps: dict[nn.Module, float] = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None
        elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
            dropout_ps[module] = module.p
            module.p = 0.0

    model.train()
    for start in range(0, x.shape[0], 1024):
        xb = x[start : start + 1024].to(DEVICE, non_blocking=True)
        xb = xb.contiguous(memory_format=torch.channels_last)
        xb = (xb - mean) / std
        model(xb)
    model.eval()

    for module, momentum in momenta.items():
        module.momentum = momentum
    for module, p in dropout_ps.items():
        module.p = p


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }


def build_model(dropout: float) -> nn.Module:
    if ARCH == "fast":
        return TorchFastCNN(width=WIDTH, hidden_size=HIDDEN_SIZE, dropout=dropout)
    if ARCH == "resnet":
        return TorchFashionCNN(width=WIDTH, dropout=dropout)
    if ARCH == "wrn":
        return TorchWideResNet(depth=WRN_DEPTH, widen_factor=WRN_WIDEN_FACTOR, dropout=dropout)
    raise ValueError(f"Unsupported ARCH: {ARCH}")


def model_width_for_state() -> int:
    return WRN_WIDEN_FACTOR if ARCH == "wrn" else WIDTH


def save_states(
    states: list[dict[str, np.ndarray]],
    mean: float,
    std: float,
    state_weights: np.ndarray | None = None,
) -> None:
    final_state = {
        "model_type": "TorchEnsembleCNN",
        "states": states,
        "state_weights": state_weights.astype(np.float32) if state_weights is not None else None,
        "mean": mean,
        "std": std,
        "width": model_width_for_state(),
        "arch": ARCH,
        "depth": WRN_DEPTH if ARCH == "wrn" else None,
        "widen_factor": WRN_WIDEN_FACTOR if ARCH == "wrn" else None,
        "hidden_size": HIDDEN_SIZE,
        "dropout": 0.0,
        "batch_size": 1024,
        "use_tta": True,
    }
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(final_state, f)
    print(f"Saved {len(states)} model(s): {OUTPUT_PATH.resolve()}", flush=True)


def optimize_ensemble_weights(
    states: list[dict[str, np.ndarray]],
    mean: float,
    std: float,
    x_valid: np.ndarray,
    t_valid: np.ndarray,
) -> np.ndarray | None:
    if TRAIN_ON_FULL_DATA or not OPTIMIZE_ENSEMBLE_WEIGHTS or len(states) <= 1:
        return None

    probs: list[np.ndarray] = []
    for index, state in enumerate(states, start=1):
        model = TorchEnsembleModel(
            states=[state],
            mean=mean,
            std=std,
            width=model_width_for_state(),
            dropout=0.0,
            batch_size=1024,
            use_tta=True,
            arch=ARCH,
            hidden_size=HIDDEN_SIZE,
            depth=WRN_DEPTH,
        )
        probs.append(model.predict_proba(x_valid))
        print(f"Collected holdout probabilities for model {index}/{len(states)}", flush=True)

    stacked = np.stack(probs, axis=0)
    weights = np.ones(len(states), dtype=np.float32) / len(states)
    best_acc = float(np.mean((stacked * weights[:, None, None]).sum(axis=0).argmax(axis=1) == t_valid))
    best_weights = weights.copy()
    rng = np.random.default_rng(10_000 + len(states))

    for alpha in (0.5, 1.0, 2.0):
        for _ in range(1200):
            candidate = rng.dirichlet(np.ones(len(states), dtype=np.float32) * alpha).astype(np.float32)
            pred = (stacked * candidate[:, None, None]).sum(axis=0).argmax(axis=1)
            acc = float(np.mean(pred == t_valid))
            if acc > best_acc:
                best_acc = acc
                best_weights = candidate

    print(
        "Holdout ensemble weights="
        + np.array2string(best_weights, precision=4, floatmode="fixed")
        + f" valid_acc={best_acc:.4f}",
        flush=True,
    )
    return best_weights


def load_existing_states() -> tuple[list[dict[str, np.ndarray]], float, float]:
    if not RESUME_EXISTING:
        return [], 0.0, 1.0
    if not OUTPUT_PATH.exists():
        return [], 0.0, 1.0
    try:
        with OUTPUT_PATH.open("rb") as f:
            state = pickle.load(f)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Existing weight ignored: {exc}", flush=True)
        return [], 0.0, 1.0

    if not isinstance(state, dict) or state.get("model_type") != "TorchEnsembleCNN":
        return [], 0.0, 1.0
    if state.get("arch") != ARCH:
        return [], 0.0, 1.0
    if ARCH == "wrn":
        if int(state.get("depth", 0)) != WRN_DEPTH:
            return [], 0.0, 1.0
        if int(state.get("widen_factor", state.get("width", 0))) != WRN_WIDEN_FACTOR:
            return [], 0.0, 1.0
    elif int(state.get("width", 0)) != WIDTH:
        return [], 0.0, 1.0
    if ARCH == "fast" and int(state.get("hidden_size", 0)) != HIDDEN_SIZE:
        return [], 0.0, 1.0

    states_obj = state.get("states")
    if not isinstance(states_obj, list):
        return [], 0.0, 1.0
    states: list[dict[str, np.ndarray]] = []
    for item in states_obj:
        if isinstance(item, dict):
            states.append({str(k): v for k, v in item.items() if isinstance(v, np.ndarray)})
    if len(states) > len(SEEDS):
        print(
            f"Existing ensemble has {len(states)} model(s), "
            f"but this training config expects at most {len(SEEDS)}; starting a new run.",
            flush=True,
        )
        return [], 0.0, 1.0
    if states:
        print(f"Resuming from {len(states)} saved model(s)", flush=True)
    return states, float(state.get("mean", 0.0)), float(state.get("std", 1.0))


def load_meta_base_state() -> dict[str, object]:
    merged_states: list[dict[str, np.ndarray]] = []
    base_meta: dict[str, object] | None = None
    for path in META_BASE_WEIGHT_PATHS:
        with path.open("rb") as f:
            state = pickle.load(f)
        if not isinstance(state, dict) or state.get("model_type") != "TorchEnsembleCNN":
            raise ValueError(f"Invalid base ensemble: {path}")
        if base_meta is None:
            base_meta = {
                "model_type": "TorchEnsembleCNN",
                "mean": state.get("mean"),
                "std": state.get("std"),
                "width": state.get("width"),
                "arch": state.get("arch"),
                "depth": state.get("depth"),
                "widen_factor": state.get("widen_factor"),
                "hidden_size": state.get("hidden_size"),
                "dropout": 0.0,
                "batch_size": 1024,
                "use_tta": True,
            }
        else:
            for key in ("mean", "std", "width", "arch", "hidden_size"):
                if state.get(key) != base_meta.get(key):
                    raise ValueError(f"Base ensemble mismatch for {key}: {path}")

        states_obj = state.get("states")
        if not isinstance(states_obj, list):
            raise ValueError(f"Base ensemble has no states list: {path}")
        for item in states_obj:
            if isinstance(item, dict):
                merged_states.append({str(k): v for k, v in item.items() if isinstance(v, np.ndarray)})

    if base_meta is None or not merged_states:
        raise ValueError("No base states were loaded for meta stacker")
    base_meta["states"] = merged_states
    base_meta["state_weights"] = np.ones(len(merged_states), dtype=np.float32) / len(merged_states)
    return base_meta


def collect_meta_features(model: TorchEnsembleModel, x: np.ndarray) -> np.ndarray:
    probs: list[np.ndarray] = []
    for index, member in enumerate(model.models, start=1):
        probs.append(model._predict_single_model_proba(member, x))  # pylint: disable=protected-access
        print(f"Collected meta features for base model {index}/{len(model.models)}", flush=True)
    return np.stack(probs, axis=1).reshape(x.shape[0], -1).astype(np.float32)


def build_meta_model(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(128, 10),
    )


def train_meta_model(
    x_train_meta: np.ndarray,
    t_train: np.ndarray,
    x_valid_meta: np.ndarray,
    t_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], int, float]:
    torch.manual_seed(123456)
    np.random.seed(123456)
    x_train_tensor = torch.from_numpy(x_train_meta).to(DEVICE)
    y_train_tensor = torch.from_numpy(t_train.astype(np.int64)).to(DEVICE)
    x_valid_tensor = torch.from_numpy(x_valid_meta).to(DEVICE)
    y_valid_tensor = torch.from_numpy(t_valid.astype(np.int64)).to(DEVICE)

    model = build_meta_model(x_train_meta.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=META_LR, weight_decay=META_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    best_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, META_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == META_EPOCHS:
            model.eval()
            with torch.no_grad():
                pred = model(x_valid_tensor).argmax(dim=1)
                valid_acc = float((pred == y_valid_tensor).float().mean().item())
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"Meta epoch {epoch:04d}/{META_EPOCHS} "
                f"loss={float(loss.item()):.4f} valid_acc={valid_acc:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Meta stacker did not produce a candidate")
    print(f"Selected meta epoch {best_epoch} by holdout valid_acc={best_acc:.4f}", flush=True)
    return (
        {key: value.numpy().astype(np.float32) for key, value in best_state.items()},
        best_epoch,
        best_acc,
    )


def retrain_meta_on_all(
    x_all_meta: np.ndarray,
    t_all: np.ndarray,
    epochs: int,
) -> dict[str, np.ndarray]:
    torch.manual_seed(123456)
    x_tensor = torch.from_numpy(x_all_meta).to(DEVICE)
    y_tensor = torch.from_numpy(t_all.astype(np.int64)).to(DEVICE)
    model = build_meta_model(x_all_meta.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=META_LR, weight_decay=META_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Meta all-data epoch {epoch:04d}/{epochs} loss={float(loss.item()):.4f}", flush=True)

    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }


def train_meta_stacker() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    base_state = load_meta_base_state()
    base_model = TorchEnsembleModel.from_state(base_state)

    print("Collecting legal meta features from train split", flush=True)
    x_train_meta = collect_meta_features(base_model, x_train)
    print("Collecting legal meta features from holdout validation split", flush=True)
    x_valid_meta = collect_meta_features(base_model, x_valid)
    _best_meta, best_epoch, best_acc = train_meta_model(
        x_train_meta,
        t_train,
        x_valid_meta,
        t_valid,
    )

    x_all_meta = np.concatenate([x_train_meta, x_valid_meta], axis=0)
    t_all = np.concatenate([t_train, t_valid], axis=0)
    final_meta = retrain_meta_on_all(x_all_meta, t_all, best_epoch)
    base_state["meta_mlp"] = final_meta
    base_state["batch_size"] = 1024

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(base_state, f)
    backup_path = Path(f"sample_weight_meta_train_valid_{best_acc:.4f}_backup.pkl")
    with backup_path.open("wb") as f:
        pickle.dump(base_state, f)
    print(f"Saved legal train-meta stacker to {OUTPUT_PATH} and {backup_path}", flush=True)
    return 0


def train_one(
    seed: int,
    x_fit: np.ndarray,
    t_fit: np.ndarray,
    x_valid: np.ndarray,
    t_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = float(x_fit.mean())
    std = float(x_fit.std() + 1e-6)
    print(
        f"device={DEVICE} amp={USE_AMP} arch={ARCH} "
        f"width={model_width_for_state()} depth={WRN_DEPTH if ARCH == 'wrn' else '-'}",
        flush=True,
    )

    x_tensor = torch.from_numpy(x_fit.reshape(-1, 1, 28, 28).astype(np.float32))
    y_tensor = torch.from_numpy(t_fit.astype(np.int64))
    xv_tensor = torch.from_numpy(x_valid.reshape(-1, 1, 28, 28).astype(np.float32))
    yv_tensor = torch.from_numpy(t_valid.astype(np.int64))

    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=DEVICE.type == "cuda",
    )
    model = build_model(dropout=DROPOUT).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    ema_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    swa_params: dict[str, torch.Tensor] | None = None
    swa_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            xb = xb.contiguous(memory_format=torch.channels_last)
            xb = augment_batch(xb)
            xb = (xb - mean) / std
            xb, y_a, y_b, lam = mixup(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_params, model)

            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            if BATCH_SLEEP_SECONDS > 0.0:
                time.sleep(BATCH_SLEEP_SECONDS)
            if step % 20 == 0:
                cool_down_if_needed()
        scheduler.step()
        if USE_SWA and epoch >= SWA_START_EPOCH:
            swa_params, swa_count = update_swa(swa_params, model, swa_count)

        valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={total_loss / total:.4f} valid_acc={valid_acc:.4f}"
        , flush=True)

    final_params = copy_model_params(model)
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [
        ("raw", final_params),
        ("ema", ema_params),
    ]
    if swa_params is not None:
        candidates.append(("swa", swa_params))

    best_name = ""
    best_acc = -1.0
    best_state: dict[str, np.ndarray] | None = None
    for name, params in candidates:
        apply_ema(params, model)
        refresh_batch_norm(model, x_tensor, mean, std)
        valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
        print(f"{name.upper()} refreshed valid_acc={valid_acc:.4f}", flush=True)
        if valid_acc > best_acc:
            best_name = name
            best_acc = valid_acc
            best_state = state_to_numpy(model)

    if best_state is None:
        raise RuntimeError("No model candidate was produced")
    print(f"Selected {best_name.upper()} by holdout valid_acc={best_acc:.4f}", flush=True)
    return best_state, mean, std


def fine_tune_one(
    seed: int,
    state: dict[str, np.ndarray],
    x_fit: np.ndarray,
    t_fit: np.ndarray,
    x_valid: np.ndarray,
    t_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = float(x_fit.mean())
    std = float(x_fit.std() + 1e-6)
    print(f"fine-tune device={DEVICE} amp={USE_AMP} seed={seed}", flush=True)

    x_tensor = torch.from_numpy(x_fit.reshape(-1, 1, 28, 28).astype(np.float32))
    y_tensor = torch.from_numpy(t_fit.astype(np.int64))
    xv_tensor = torch.from_numpy(x_valid.reshape(-1, 1, 28, 28).astype(np.float32))
    yv_tensor = torch.from_numpy(t_valid.astype(np.int64))

    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=DEVICE.type == "cuda",
    )
    model = build_model(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict({key: torch.from_numpy(value) for key, value in state.items()})
    model = model.to(memory_format=torch.channels_last)
    ema_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY * 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=MIN_LR
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.005)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            xb = xb.contiguous(memory_format=torch.channels_last)
            xb = augment_batch(xb)
            xb = (xb - mean) / std

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_params, model)

            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
            if BATCH_SLEEP_SECONDS > 0.0:
                time.sleep(BATCH_SLEEP_SECONDS)
            if step % 20 == 0:
                cool_down_if_needed()
        scheduler.step()
        valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
        print(
            f"FineTune {epoch:02d}/{FINETUNE_EPOCHS} "
            f"loss={total_loss / total:.4f} valid_acc={valid_acc:.4f}",
            flush=True,
        )

    apply_ema(ema_params, model)
    refresh_batch_norm(model, x_tensor, mean, std)
    valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
    print(f"FineTune EMA refreshed valid_acc={valid_acc:.4f}", flush=True)
    return state_to_numpy(model), mean, std


def main() -> int:
    if TRAIN_META_STACKER:
        return train_meta_stacker()

    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    if TRAIN_ON_FULL_DATA:
        x_fit = np.concatenate([x_train, x_valid], axis=0)
        t_fit = np.concatenate([t_train, t_valid], axis=0)
        print(
            "Training on all labeled train data; validation metrics are monitoring only.",
            flush=True,
        )
    else:
        x_fit = x_train
        t_fit = t_train
        print(
            "Training on train split only; selecting candidates by holdout validation.",
            flush=True,
        )

    states, mean, std = load_existing_states()
    if FINETUNE_EXISTING and len(states) == len(SEEDS):
        tuned_states: list[dict[str, np.ndarray]] = []
        for index, (seed, state) in enumerate(zip(SEEDS, states), start=1):
            print(f"\nFine-tuning saved model {index}/{len(states)} seed={seed}", flush=True)
            tuned_state, mean, std = fine_tune_one(seed + 100000, state, x_fit, t_fit, x_valid, t_valid)
            tuned_states.append(tuned_state)
            current_states = tuned_states + states[index:]
            state_weights = optimize_ensemble_weights(current_states, mean, std, x_valid, t_valid)
            save_states(current_states, mean, std, state_weights=state_weights)
        return 0

    start_index = len(states)
    for index, seed in enumerate(SEEDS[start_index:], start=start_index + 1):
        print(f"\nTorch {ARCH} CNN {index}/{len(SEEDS)} seed={seed}", flush=True)
        state, mean, std = train_one(seed, x_fit, t_fit, x_valid, t_valid)
        states.append(state)
        state_weights = optimize_ensemble_weights(states, mean, std, x_valid, t_valid)
        save_states(states, mean, std, state_weights=state_weights)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
