# uv run src/train.py

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from load_fashion_mnist import load_train_data
from network import TorchFastCNN

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 38
SEEDS = (42, 123, 777, 2026, 31415, 27182)
BATCH_SIZE = 512
LEARNING_RATE = 1.4e-3
MIN_LR = 1.5e-5
WEIGHT_DECAY = 2.5e-4
WIDTH = 64
HIDDEN_SIZE = 512
DROPOUT = 0.20
LABEL_SMOOTHING = 0.02
MIXUP_ALPHA = 0.20
EMA_DECAY = 0.997
FINETUNE_EXISTING = True
FINETUNE_EPOCHS = 8
FINETUNE_LR = 2.0e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


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

    erase_mask = torch.rand(out.shape[0], device=out.device) < 0.15
    if torch.any(erase_mask):
        for size in (4, 5, 6, 7):
            size_mask = erase_mask & (torch.randint(4, 8, (out.shape[0],), device=out.device) == size)
            if not torch.any(size_mask):
                continue
            y0 = int(torch.randint(0, 29 - size, (), device=out.device).item())
            x0 = int(torch.randint(0, 29 - size, (), device=out.device).item())
            out[size_mask, :, y0 : y0 + size, x0 : x0 + size] = 0.0
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


def apply_ema(ema_params: dict[str, torch.Tensor], model: nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(ema_params[name])


@torch.no_grad()
def refresh_batch_norm(model: nn.Module, x: torch.Tensor, mean: float, std: float) -> None:
    momenta: dict[nn.Module, float | None] = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None

    model.train()
    for start in range(0, x.shape[0], 1024):
        xb = x[start : start + 1024].to(DEVICE, non_blocking=True)
        xb = xb.contiguous(memory_format=torch.channels_last)
        xb = (xb - mean) / std
        model(xb)
    model.eval()

    for module, momentum in momenta.items():
        module.momentum = momentum


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }


def save_states(states: list[dict[str, np.ndarray]], mean: float, std: float) -> None:
    final_state = {
        "model_type": "TorchEnsembleCNN",
        "states": states,
        "mean": mean,
        "std": std,
        "width": WIDTH,
        "arch": "fast",
        "hidden_size": HIDDEN_SIZE,
        "dropout": 0.0,
        "batch_size": 1024,
        "use_tta": True,
    }
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(final_state, f)
    print(f"Saved {len(states)} model(s): {OUTPUT_PATH.resolve()}", flush=True)


def load_existing_states() -> tuple[list[dict[str, np.ndarray]], float, float]:
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
    if state.get("arch") != "fast" or int(state.get("width", 0)) != WIDTH:
        return [], 0.0, 1.0
    if int(state.get("hidden_size", 0)) != HIDDEN_SIZE:
        return [], 0.0, 1.0

    states_obj = state.get("states")
    if not isinstance(states_obj, list):
        return [], 0.0, 1.0
    states: list[dict[str, np.ndarray]] = []
    for item in states_obj:
        if isinstance(item, dict):
            states.append({str(k): v for k, v in item.items() if isinstance(v, np.ndarray)})
    if states:
        print(f"Resuming from {len(states)} saved model(s)", flush=True)
    return states, float(state.get("mean", 0.0)), float(state.get("std", 1.0))


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
    print(f"device={DEVICE} amp={USE_AMP} arch=fast width={WIDTH}", flush=True)

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
    model = TorchFastCNN(width=WIDTH, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    ema_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
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
        scheduler.step()

        valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"loss={total_loss / total:.4f} valid_acc={valid_acc:.4f}"
        , flush=True)

    apply_ema(ema_params, model)
    refresh_batch_norm(model, x_tensor, mean, std)
    valid_acc = evaluate(model, xv_tensor, yv_tensor, mean, std)
    print(f"EMA refreshed valid_acc={valid_acc:.4f}", flush=True)
    return state_to_numpy(model), mean, std


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
    model = TorchFastCNN(width=WIDTH, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
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
        for xb, yb in loader:
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
    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    x_fit = np.concatenate([x_train, x_valid], axis=0)
    t_fit = np.concatenate([t_train, t_valid], axis=0)

    states, mean, std = load_existing_states()
    if FINETUNE_EXISTING and len(states) == len(SEEDS):
        tuned_states: list[dict[str, np.ndarray]] = []
        for index, (seed, state) in enumerate(zip(SEEDS, states), start=1):
            print(f"\nFine-tuning saved model {index}/{len(states)} seed={seed}", flush=True)
            tuned_state, mean, std = fine_tune_one(seed + 100000, state, x_fit, t_fit, x_valid, t_valid)
            tuned_states.append(tuned_state)
            save_states(tuned_states + states[index:], mean, std)
        return 0

    start_index = len(states)
    for index, seed in enumerate(SEEDS[start_index:], start=start_index + 1):
        print(f"\nTorch Fast CNN {index}/{len(SEEDS)} seed={seed}", flush=True)
        state, mean, std = train_one(seed, x_fit, t_fit, x_valid, t_valid)
        states.append(state)
        save_states(states, mean, std)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
