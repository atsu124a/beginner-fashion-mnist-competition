from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    learning_rate: float = 0.1
    batch_size: int = 128
    seed: int = 42


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.params: dict[str, np.ndarray] = {
            "W1": (rng.standard_normal((config.input_size, config.hidden_size)) * 0.01).astype(
                np.float32
            ),
            "b1": np.zeros(config.hidden_size, dtype=np.float32),
            "W2": (rng.standard_normal((config.hidden_size, config.output_size)) * 0.01).astype(
                np.float32
            ),
            "b2": np.zeros(config.output_size, dtype=np.float32),
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z1 = np.tanh(np.dot(x, self.params["W1"]) + self.params["b1"])
        logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        total = x.shape[0]
        batch_size = self.config.batch_size
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            pred = self.predict(x_batch)
            correct += int(np.sum(pred == y_batch))
        return float(correct) / float(total)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
            z1 = np.tanh(z1_linear)
            logits = np.dot(z1, self.params["W2"]) + self.params["b2"]
            probs = _softmax(logits)

            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            dW2 = np.dot(z1.T, d_logits)
            db2 = np.sum(d_logits, axis=0)

            d_z1 = np.dot(d_logits, self.params["W2"].T)
            d_z1_linear = d_z1 * (1.0 - np.square(z1))
            dW1 = np.dot(x_batch.T, d_z1_linear)
            db1 = np.sum(d_z1_linear, axis=0)

            lr = self.config.learning_rate
            self.params["W1"] -= lr * dW1.astype(np.float32)
            self.params["b1"] -= lr * db1.astype(np.float32)
            self.params["W2"] -= lr * dW2.astype(np.float32)
            self.params["b2"] -= lr * db2.astype(np.float32)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config_dict: dict[str, Any] = config_obj

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_size=int(config_dict["hidden_size"]),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.1)),
            batch_size=int(config_dict.get("batch_size", 128)),
            seed=int(config_dict.get("seed", 42)),
        )

        params_obj = state.get("params")
        if not isinstance(params_obj, dict):
            raise ValueError("Invalid state: 'params' must be a dict")
        params: dict[str, np.ndarray] = {}
        for key, value in params_obj.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray):
                raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
            params[key] = value

        model = cls(config)
        model.params = params
        return model
