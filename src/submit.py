# uv run src/submit.py

import pickle
from pathlib import Path

import numpy as np

from load_fashion_mnist import load_test_data
from network import SimpleMLP

WEIGHTS_PATH = Path("sample_weight.pkl")


def main() -> int:
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] weights file not found: {WEIGHTS_PATH}")
        return 1

    with WEIGHTS_PATH.open("rb") as f:
        state = pickle.load(f)

    model = SimpleMLP.from_state(state)

    x_test, t_test = load_test_data()

    pred = model.predict(x_test)
    acc = float(np.mean(pred == t_test))
    print(f"Test Accuracy: {acc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
