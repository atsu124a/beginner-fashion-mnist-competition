# uv run src/download_fashion_mnist.py

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_file(filename: str, expected_md5: str, output_dir: Path, force: bool) -> None:
    path = output_dir / filename
    if path.exists() and not force:
        current = md5sum(path)
        if current == expected_md5:
            print(f"[OK] {filename} already exists and checksum is valid")
            return
        print(f"[WARN] {filename} exists but checksum mismatch ({current})")
        print("       Re-downloading...")

    url = f"{BASE_URL}/{filename}"
    print(f"[DL] {url}")
    download_file(url, path)

    current = md5sum(path)
    if current != expected_md5:
        raise RuntimeError(
            f"Checksum mismatch for {filename}: expected {expected_md5}, got {current}"
        )

    print(f"[OK] {filename} downloaded and verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Fashion-MNIST dataset files and verify MD5 checksums"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to store .gz files (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if already present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, checksum in FILES.items():
            ensure_file(name, checksum, output_dir, force=args.force)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"\nCompleted. Files are in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
