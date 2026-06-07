import gzip
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
VALID_SIZE = 10000
SEED = 42
SHUFFLE = True
NORMALIZE = True
FLATTEN = True

LABEL_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number: {magic} ({path})")
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8)
    return images.reshape(num_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic number: {magic} ({path})")
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8)
    if labels.shape[0] != num_labels:
        raise ValueError(
            f"Label count mismatch: header={num_labels}, actual={labels.shape[0]} ({path})"
        )
    return labels


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    out = images
    if FLATTEN:
        out = out.reshape(out.shape[0], -1)
    if NORMALIZE:
        out = out.astype(np.float32) / 255.0
    return out


def load_train_data() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    train_images = _read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    train_images = _preprocess_images(train_images)

    if VALID_SIZE <= 0 or VALID_SIZE >= train_images.shape[0]:
        raise ValueError(
            f"VALID_SIZE must be between 1 and {train_images.shape[0] - 1}, got {VALID_SIZE}"
        )

    if SHUFFLE:
        rng = np.random.default_rng(SEED)
        indices = rng.permutation(train_images.shape[0])
    else:
        indices = np.arange(train_images.shape[0])

    valid_indices = indices[:VALID_SIZE]
    train_indices = indices[VALID_SIZE:]

    x_valid = train_images[valid_indices]
    t_valid = train_labels[valid_indices]
    x_train = train_images[train_indices]
    t_train = train_labels[train_indices]

    return (x_train, t_train), (x_valid, t_valid)


def load_eval_data() -> tuple[np.ndarray, np.ndarray]:
    _, valid = load_train_data()
    return valid


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    x_test = _read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    t_test = _read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return _preprocess_images(x_test), t_test


def main() -> None:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    x_test, t_test = load_test_data()

    print("Loaded Fashion-MNIST")
    print(f"x_train shape: {x_train.shape}")
    print(f"t_train shape: {t_train.shape}")
    print(f"x_valid shape: {x_valid.shape}")
    print(f"t_valid shape: {t_valid.shape}")
    print(f"x_test shape : {x_test.shape}")
    print(f"t_test shape : {t_test.shape}")

    unique, counts = np.unique(t_train, return_counts=True)
    print("\\nTrain label counts:")
    for label_id, count in zip(unique, counts):
        print(f"  {label_id}: {LABEL_NAMES[int(label_id)]} -> {int(count)}")


if __name__ == "__main__":
    main()
