from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - only needed when torch is unavailable.
    torch = None
    nn = None
    F = None


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int, smoothing: float) -> np.ndarray:
    out = np.full((labels.shape[0], num_classes), smoothing / num_classes, dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] += 1.0 - smoothing
    return out


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_size: int = 256
    hidden_sizes: tuple[int, ...] | None = None
    output_size: int = 10
    learning_rate: float = 8e-4
    batch_size: int = 96
    seed: int = 42
    weight_decay: float = 2e-4
    dropout_rate: float = 0.25
    label_smoothing: float = 0.05
    augment_shift: bool = True
    conv_channels: tuple[int, int] = (32, 64)
    epochs: int = 28
    use_tta: bool = True


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _augment_images(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    images = x.reshape(-1, 28, 28)
    out = np.empty_like(images)
    for i, image in enumerate(images):
        dy = int(rng.integers(-2, 3))
        dx = int(rng.integers(-2, 3))
        shifted = np.zeros_like(image)
        src_y0 = max(0, -dy)
        src_y1 = min(28, 28 - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(28, 28 + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(28, 28 - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(28, 28 + dx)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        if rng.random() < 0.5:
            shifted = shifted[:, ::-1]
        if rng.random() < 0.35:
            size = int(rng.integers(4, 8))
            y0 = int(rng.integers(0, 29 - size))
            x0 = int(rng.integers(0, 29 - size))
            shifted[y0 : y0 + size, x0 : x0 + size] = 0.0
        out[i] = shifted
    return out.reshape(x.shape)


def _transform_flat_images(x: np.ndarray, dy: int = 0, dx: int = 0, flip: bool = False) -> np.ndarray:
    images = x.reshape(-1, 28, 28)
    if flip:
        images = images[:, :, ::-1]
    if dy == 0 and dx == 0:
        return images.reshape(x.shape)

    shifted = np.zeros_like(images)
    src_y0 = max(0, -dy)
    src_y1 = min(28, 28 - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(28, 28 + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(28, 28 - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(28, 28 + dx)
    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, src_y0:src_y1, src_x0:src_x1]
    return shifted.reshape(x.shape)


def _conv_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int
) -> tuple[np.ndarray, tuple[Any, ...]]:
    n, c, h, width = x.shape
    filters, _, kh, kw = w.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kh, kw), axis=(2, 3))
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * width, c * kh * kw)
    out = cols @ w.reshape(filters, -1).T + b
    out = out.reshape(n, h, width, filters).transpose(0, 3, 1, 2)
    return out.astype(np.float32), (x.shape, cols, w, pad)


def _conv_backward(dout: np.ndarray, cache: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_shape, cols, w, pad = cache
    n, c, h, width = x_shape
    filters, _, kh, kw = w.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(n * h * width, filters)

    dw = (dout_flat.T @ cols).reshape(w.shape)
    db = np.sum(dout_flat, axis=0)
    dcols = dout_flat @ w.reshape(filters, -1)
    dcols = dcols.reshape(n, h, width, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    dx_pad = np.zeros((n, c, h + 2 * pad, width + 2 * pad), dtype=np.float32)
    for yy in range(kh):
        for xx in range(kw):
            dx_pad[:, :, yy : yy + h, xx : xx + width] += dcols[:, :, yy, xx]
    dx = dx_pad[:, :, pad : pad + h, pad : pad + width] if pad else dx_pad
    return dx.astype(np.float32), dw.astype(np.float32), db.astype(np.float32)


def _maxpool_forward(x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n, c, h, w = x.shape
    x_view = x.reshape(n, c, h // 2, 2, w // 2, 2)
    out = x_view.max(axis=(3, 5))
    return out, (x_view, out)


def _maxpool_backward(dout: np.ndarray, cache: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    x_view, out = cache
    mask = x_view == out[:, :, :, None, :, None]
    counts = mask.sum(axis=(3, 5), keepdims=True)
    dx_view = mask * (dout[:, :, :, None, :, None] / counts)
    n, c, h2, _, w2, _ = x_view.shape
    return dx_view.reshape(n, c, h2 * 2, w2 * 2).astype(np.float32)


def _batch_norm_2d_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool,
    momentum: float = 0.9,
) -> tuple[np.ndarray, tuple[Any, ...] | None]:
    axes = (0, 2, 3)
    shape = (1, x.shape[1], 1, 1)
    if training:
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        running_mean *= momentum
        running_mean += (1.0 - momentum) * mean.reshape(-1)
        running_var *= momentum
        running_var += (1.0 - momentum) * var.reshape(-1)
    else:
        mean = running_mean.reshape(shape)
        var = running_var.reshape(shape)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    x_hat = (x - mean) * inv_std
    out = gamma.reshape(shape) * x_hat + beta.reshape(shape)
    cache = (x_hat, inv_std, gamma, axes) if training else None
    return out.astype(np.float32), cache


def _batch_norm_1d_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool,
    momentum: float = 0.9,
) -> tuple[np.ndarray, tuple[Any, ...] | None]:
    if training:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        running_mean *= momentum
        running_mean += (1.0 - momentum) * mean.reshape(-1)
        running_var *= momentum
        running_var += (1.0 - momentum) * var.reshape(-1)
    else:
        mean = running_mean.reshape(1, -1)
        var = running_var.reshape(1, -1)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    x_hat = (x - mean) * inv_std
    out = gamma.reshape(1, -1) * x_hat + beta.reshape(1, -1)
    cache = (x_hat, inv_std, gamma, (0,)) if training else None
    return out.astype(np.float32), cache


def _batch_norm_backward(
    dout: np.ndarray, cache: tuple[Any, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat, inv_std, gamma, axes = cache
    reduce_shape = tuple(1 if axis in axes else dout.shape[axis] for axis in range(dout.ndim))
    m = np.prod([dout.shape[axis] for axis in axes])
    gamma_view = gamma.reshape(reduce_shape)
    dx_hat = dout * gamma_view
    sum_dx_hat = dx_hat.sum(axis=axes, keepdims=True)
    sum_dx_hat_xhat = (dx_hat * x_hat).sum(axis=axes, keepdims=True)
    dx = (dx_hat * m - sum_dx_hat - x_hat * sum_dx_hat_xhat) * inv_std / m
    dgamma = (dout * x_hat).sum(axis=axes)
    dbeta = dout.sum(axis=axes)
    return dx.astype(np.float32), dgamma.astype(np.float32), dbeta.astype(np.float32)


class SimpleCNN:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        c1, c2 = config.conv_channels
        rng = np.random.default_rng(config.seed)
        self.params: dict[str, np.ndarray] = {
            "Wc1": (rng.standard_normal((c1, 1, 3, 3)) * np.sqrt(2 / 9)).astype(np.float32),
            "bc1": np.zeros(c1, dtype=np.float32),
            "g1": np.ones(c1, dtype=np.float32),
            "be1": np.zeros(c1, dtype=np.float32),
            "Wc2": (rng.standard_normal((c1, c1, 3, 3)) * np.sqrt(2 / (c1 * 9))).astype(
                np.float32
            ),
            "bc2": np.zeros(c1, dtype=np.float32),
            "g2": np.ones(c1, dtype=np.float32),
            "be2": np.zeros(c1, dtype=np.float32),
            "Wc3": (rng.standard_normal((c2, c1, 3, 3)) * np.sqrt(2 / (c1 * 9))).astype(
                np.float32
            ),
            "bc3": np.zeros(c2, dtype=np.float32),
            "g3": np.ones(c2, dtype=np.float32),
            "be3": np.zeros(c2, dtype=np.float32),
            "Wc4": (rng.standard_normal((c2, c2, 3, 3)) * np.sqrt(2 / (c2 * 9))).astype(
                np.float32
            ),
            "bc4": np.zeros(c2, dtype=np.float32),
            "g4": np.ones(c2, dtype=np.float32),
            "be4": np.zeros(c2, dtype=np.float32),
            "Wf1": (rng.standard_normal((c2 * 7 * 7, config.hidden_size)) * np.sqrt(2 / (c2 * 7 * 7))).astype(
                np.float32
            ),
            "bf1": np.zeros(config.hidden_size, dtype=np.float32),
            "gf1": np.ones(config.hidden_size, dtype=np.float32),
            "bef1": np.zeros(config.hidden_size, dtype=np.float32),
            "Wf2": (
                rng.standard_normal((config.hidden_size, config.output_size))
                * np.sqrt(1 / config.hidden_size)
            ).astype(np.float32),
            "bf2": np.zeros(config.output_size, dtype=np.float32),
        }
        self.running: dict[str, np.ndarray] = {}
        for name, size in {"1": c1, "2": c1, "3": c2, "4": c2, "f1": config.hidden_size}.items():
            self.running[f"mean{name}"] = np.zeros(size, dtype=np.float32)
            self.running[f"var{name}"] = np.ones(size, dtype=np.float32)
        self.adam_m: dict[str, np.ndarray] = {}
        self.adam_v: dict[str, np.ndarray] = {}
        self.adam_step = 0
        self.input_mean = np.array([0.0], dtype=np.float32)
        self.input_std = np.array([1.0], dtype=np.float32)

    def set_standardization(self, x: np.ndarray) -> None:
        self.input_mean = np.array([float(x.mean())], dtype=np.float32)
        self.input_std = np.array([float(x.std() + 1e-6)], dtype=np.float32)

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        out = x.reshape(-1, 1, 28, 28).astype(np.float32)
        return (out - self.input_mean.reshape(1, 1, 1, 1)) / self.input_std.reshape(1, 1, 1, 1)

    def _forward(
        self,
        x: np.ndarray,
        training: bool,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[tuple[str, Any]]]:
        caches: list[tuple[str, Any]] = []
        out = self._prepare_input(x)

        for idx in (1, 2):
            out, conv_cache = _conv_forward(out, self.params[f"Wc{idx}"], self.params[f"bc{idx}"], pad=1)
            out, bn_cache = _batch_norm_2d_forward(
                out,
                self.params[f"g{idx}"],
                self.params[f"be{idx}"],
                self.running[f"mean{idx}"],
                self.running[f"var{idx}"],
                training,
            )
            relu_input = out
            out = _relu(out)
            caches.append((f"conv{idx}", (conv_cache, bn_cache, relu_input)))
        out, pool1_cache = _maxpool_forward(out)
        caches.append(("pool", pool1_cache))

        for idx in (3, 4):
            out, conv_cache = _conv_forward(out, self.params[f"Wc{idx}"], self.params[f"bc{idx}"], pad=1)
            out, bn_cache = _batch_norm_2d_forward(
                out,
                self.params[f"g{idx}"],
                self.params[f"be{idx}"],
                self.running[f"mean{idx}"],
                self.running[f"var{idx}"],
                training,
            )
            relu_input = out
            out = _relu(out)
            caches.append((f"conv{idx}", (conv_cache, bn_cache, relu_input)))
        out, pool2_cache = _maxpool_forward(out)
        caches.append(("pool", pool2_cache))

        flat = out.reshape(out.shape[0], -1)
        fc1_linear = flat @ self.params["Wf1"] + self.params["bf1"]
        fc1_bn, fc1_bn_cache = _batch_norm_1d_forward(
            fc1_linear,
            self.params["gf1"],
            self.params["bef1"],
            self.running["meanf1"],
            self.running["varf1"],
            training,
        )
        fc1 = _relu(fc1_bn)
        dropout_mask = None
        if training and self.config.dropout_rate > 0.0 and rng is not None:
            keep_prob = 1.0 - self.config.dropout_rate
            dropout_mask = (rng.random(fc1.shape) < keep_prob).astype(np.float32) / keep_prob
            fc1 = fc1 * dropout_mask
        logits = fc1 @ self.params["Wf2"] + self.params["bf2"]
        caches.append(("head", (out.shape, flat, fc1_linear, fc1_bn_cache, fc1_bn, dropout_mask, fc1)))
        return logits.astype(np.float32), caches

    def _predict_proba_once(self, x: np.ndarray) -> np.ndarray:
        probs: list[np.ndarray] = []
        for x_batch in np.array_split(x, max(1, int(np.ceil(x.shape[0] / self.config.batch_size)))):
            logits, _ = self._forward(x_batch, training=False)
            probs.append(_softmax(logits))
        return np.concatenate(probs, axis=0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.config.use_tta:
            return self._predict_proba_once(x)

        transforms = (
            (0, 0, False),
            (0, 0, True),
            (-1, 0, False),
            (1, 0, False),
            (0, -1, False),
            (0, 1, False),
        )
        probs = [
            self._predict_proba_once(_transform_flat_images(x, dy=dy, dx=dx, flip=flip))
            for dy, dx, flip in transforms
        ]
        return np.mean(probs, axis=0).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean(pred == y))

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch * 1009)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0

        for start in range(0, x.shape[0], self.config.batch_size):
            batch_idx = indices[start : start + self.config.batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]
            if self.config.augment_shift:
                x_batch = _augment_images(x_batch, rng)

            logits, caches = self._forward(x_batch, training=True, rng=rng)
            probs = _softmax(logits)
            targets = _one_hot(y_batch, self.config.output_size, self.config.label_smoothing)
            loss = -np.mean(np.sum(targets * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            grads: dict[str, np.ndarray] = {}
            dout = (probs - targets) / x_batch.shape[0]

            _, head_cache = caches.pop()
            conv_out_shape, flat, _fc1_linear, fc1_bn_cache, fc1_bn, dropout_mask, fc1 = head_cache
            grads["Wf2"] = flat * 0.0
            grads["Wf2"] = (fc1.T @ dout).astype(np.float32)
            grads["bf2"] = dout.sum(axis=0).astype(np.float32)
            d_fc1 = dout @ self.params["Wf2"].T
            if dropout_mask is not None:
                d_fc1 *= dropout_mask
            d_fc1 *= fc1_bn > 0.0
            d_fc1_linear, grads["gf1"], grads["bef1"] = _batch_norm_backward(d_fc1, fc1_bn_cache)
            grads["Wf1"] = (flat.T @ d_fc1_linear).astype(np.float32)
            grads["bf1"] = d_fc1_linear.sum(axis=0).astype(np.float32)
            dout = (d_fc1_linear @ self.params["Wf1"].T).reshape(conv_out_shape)

            for name, cache in reversed(caches):
                if name == "pool":
                    dout = _maxpool_backward(dout, cache)
                    continue
                layer = name[-1]
                conv_cache, bn_cache, relu_input = cache
                dout *= relu_input > 0.0
                dout, grads[f"g{layer}"], grads[f"be{layer}"] = _batch_norm_backward(dout, bn_cache)
                dout, grads[f"Wc{layer}"], grads[f"bc{layer}"] = _conv_backward(dout, conv_cache)

            self._adamw_update(grads, epoch)

        return total_loss / max(steps, 1)

    def _adamw_update(self, grads: dict[str, np.ndarray], epoch: int) -> None:
        self.adam_step += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        progress = min(epoch, self.config.epochs) / max(self.config.epochs, 1)
        lr = self.config.learning_rate * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * progress)))

        for key, grad in grads.items():
            grad = np.clip(grad, -3.0, 3.0).astype(np.float32)
            if key not in self.adam_m:
                self.adam_m[key] = np.zeros_like(grad)
                self.adam_v[key] = np.zeros_like(grad)
            self.adam_m[key] = beta1 * self.adam_m[key] + (1.0 - beta1) * grad
            self.adam_v[key] = beta2 * self.adam_v[key] + (1.0 - beta2) * (grad * grad)
            m_hat = self.adam_m[key] / (1.0 - beta1**self.adam_step)
            v_hat = self.adam_v[key] / (1.0 - beta2**self.adam_step)
            if key.startswith("W"):
                self.params[key] *= 1.0 - lr * self.config.weight_decay
            self.params[key] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleCNN",
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "hidden_sizes": self.config.hidden_sizes,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
                "weight_decay": self.config.weight_decay,
                "dropout_rate": self.config.dropout_rate,
                "label_smoothing": self.config.label_smoothing,
                "augment_shift": self.config.augment_shift,
                "conv_channels": self.config.conv_channels,
                "epochs": self.config.epochs,
                "use_tta": self.config.use_tta,
            },
            "params": self.params,
            "running": self.running,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleCNN":
        config_dict = state.get("config")
        if not isinstance(config_dict, dict):
            raise ValueError("Invalid CNN state: 'config' must be a dict")
        conv_channels_obj = config_dict.get("conv_channels", (32, 64))
        config = NetworkConfig(
            input_size=int(config_dict.get("input_size", 784)),
            hidden_size=int(config_dict.get("hidden_size", 256)),
            hidden_sizes=tuple(config_dict["hidden_sizes"])
            if isinstance(config_dict.get("hidden_sizes"), (list, tuple))
            else None,
            output_size=int(config_dict.get("output_size", 10)),
            learning_rate=float(config_dict.get("learning_rate", 8e-4)),
            batch_size=int(config_dict.get("batch_size", 96)),
            seed=int(config_dict.get("seed", 42)),
            weight_decay=float(config_dict.get("weight_decay", 2e-4)),
            dropout_rate=float(config_dict.get("dropout_rate", 0.0)),
            label_smoothing=float(config_dict.get("label_smoothing", 0.0)),
            augment_shift=bool(config_dict.get("augment_shift", False)),
            conv_channels=tuple(int(v) for v in conv_channels_obj),
            epochs=int(config_dict.get("epochs", 28)),
            use_tta=bool(config_dict.get("use_tta", True)),
        )
        model = cls(config)
        params = state.get("params")
        running = state.get("running")
        if not isinstance(params, dict) or not isinstance(running, dict):
            raise ValueError("Invalid CNN state: missing params/running")
        model.params = {str(k): v for k, v in params.items() if isinstance(v, np.ndarray)}
        model.running = {str(k): v for k, v in running.items() if isinstance(v, np.ndarray)}
        input_mean = state.get("input_mean")
        input_std = state.get("input_std")
        if isinstance(input_mean, np.ndarray) and isinstance(input_std, np.ndarray):
            model.input_mean = input_mean
            model.input_std = input_std
        return model


class EnsembleModel:
    def __init__(self, models: list[Any]) -> None:
        if not models:
            raise ValueError("EnsembleModel requires at least one model")
        self.models = models
        self.config = models[0].config

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = [model.predict_proba(x) for model in self.models]
        return np.mean(probs, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def to_state(self) -> dict[str, object]:
        return {"model_type": "EnsembleModel", "models": [model.to_state() for model in self.models]}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "EnsembleModel":
        model_states = state.get("models")
        if not isinstance(model_states, list):
            raise ValueError("Invalid ensemble state: 'models' must be a list")
        return cls([SimpleMLP.from_state(model_state) for model_state in model_states])


if nn is not None:

    class ConvBnAct(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.bn(self.conv(x)))


    class SqueezeExcite(nn.Module):
        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(channels // reduction, 8)
            self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
            self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            scale = F.adaptive_avg_pool2d(x, 1)
            scale = F.silu(self.fc1(scale))
            scale = torch.sigmoid(self.fc2(scale))
            return x * scale


    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = ConvBnAct(in_channels, out_channels, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.se = SqueezeExcite(out_channels)
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv1(x)
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            return F.silu(out + self.skip(x))


    class TorchFashionCNN(nn.Module):
        def __init__(self, width: int = 64, dropout: float = 0.15) -> None:
            super().__init__()
            c1 = width
            c2 = width * 2
            c3 = width * 3
            self.stem = ConvBnAct(1, c1)
            self.stage1 = nn.Sequential(
                ResidualBlock(c1, c1),
                ResidualBlock(c1, c1),
            )
            self.stage2 = nn.Sequential(
                ResidualBlock(c1, c2, stride=2),
                ResidualBlock(c2, c2),
                ResidualBlock(c2, c2),
            )
            self.stage3 = nn.Sequential(
                ResidualBlock(c2, c3, stride=2),
                ResidualBlock(c3, c3),
                ResidualBlock(c3, c3),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(c3, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            return self.head(x)


    class TorchFastCNN(nn.Module):
        def __init__(self, width: int = 64, hidden_size: int = 512, dropout: float = 0.20) -> None:
            super().__init__()
            c1 = width
            c2 = width * 2
            c3 = width * 3
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.05),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.10),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c3 * 7 * 7, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))


class TorchEnsembleModel:
    def __init__(
        self,
        states: list[dict[str, np.ndarray]],
        mean: float,
        std: float,
        width: int = 64,
        dropout: float = 0.0,
        batch_size: int = 512,
        use_tta: bool = True,
        arch: str = "resnet",
        hidden_size: int = 512,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("TorchEnsembleModel requires torch to be installed")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = float(mean)
        self.std = float(std)
        self.batch_size = int(batch_size)
        self.use_tta = bool(use_tta)
        self.arch = arch
        self.hidden_size = int(hidden_size)
        self.models: list[Any] = []
        self.config = NetworkConfig(batch_size=batch_size, use_tta=use_tta)

        for state in states:
            if arch == "fast":
                model = TorchFastCNN(width=width, hidden_size=hidden_size, dropout=dropout).to(self.device)
            else:
                model = TorchFashionCNN(width=width, dropout=dropout).to(self.device)
            torch_state = {
                key: torch.from_numpy(value).to(self.device)
                for key, value in state.items()
            }
            model.load_state_dict(torch_state)
            model.eval()
            self.models.append(model)

    def _predict_one_transform(self, x: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                batch = x[start : start + self.batch_size].reshape(-1, 1, 28, 28).astype(np.float32)
                xb = torch.from_numpy(batch).to(self.device, non_blocking=True)
                xb = (xb - self.mean) / self.std
                probs = []
                for model in self.models:
                    probs.append(torch.softmax(model(xb), dim=1))
                avg = torch.stack(probs, dim=0).mean(dim=0)
                outputs.append(avg.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        transforms = ((0, 0, False),) if not self.use_tta else (
            (0, 0, False),
            (0, 0, True),
            (-1, 0, False),
            (1, 0, False),
            (0, -1, False),
            (0, 1, False),
        )
        probs = [
            self._predict_one_transform(_transform_flat_images(x, dy=dy, dx=dx, flip=flip))
            for dy, dx, flip in transforms
        ]
        return np.mean(probs, axis=0).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def to_state(self) -> dict[str, object]:
        width = (
            self.models[0].features[0].out_channels
            if self.arch == "fast"
            else self.models[0].stem.conv.out_channels
        )
        return {
            "model_type": "TorchEnsembleCNN",
            "states": [
                {
                    key: value.detach().cpu().numpy().astype(np.float32)
                    for key, value in model.state_dict().items()
                }
                for model in self.models
            ],
            "mean": self.mean,
            "std": self.std,
            "width": width,
            "arch": self.arch,
            "hidden_size": self.hidden_size,
            "dropout": 0.0,
            "batch_size": self.batch_size,
            "use_tta": self.use_tta,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "TorchEnsembleModel":
        states_obj = state.get("states")
        if not isinstance(states_obj, list):
            raise ValueError("Invalid TorchEnsembleCNN state: 'states' must be a list")
        states: list[dict[str, np.ndarray]] = []
        for item in states_obj:
            if not isinstance(item, dict):
                raise ValueError("Invalid TorchEnsembleCNN state item")
            states.append({str(key): value for key, value in item.items() if isinstance(value, np.ndarray)})
        return cls(
            states=states,
            mean=float(state.get("mean", 0.0)),
            std=float(state.get("std", 1.0)),
            width=int(state.get("width", 64)),
            dropout=float(state.get("dropout", 0.0)),
            batch_size=int(state.get("batch_size", 512)),
            use_tta=bool(state.get("use_tta", True)),
            arch=str(state.get("arch", "resnet")),
            hidden_size=int(state.get("hidden_size", 512)),
        )


class SimpleMLP:
    @classmethod
    def from_state(cls, state: dict[str, object]) -> Any:
        model_type = state.get("model_type")
        if model_type == "TorchEnsembleCNN":
            return TorchEnsembleModel.from_state(state)
        if model_type == "SimpleCNN":
            return SimpleCNN.from_state(state)
        if model_type in {"EnsembleModel", "EnsembleMLP"}:
            return EnsembleModel.from_state(state)
        raise ValueError(f"Unsupported model_type: {model_type}")
