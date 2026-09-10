from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


ARCHIVE_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip?download=true"
ARCHIVE_SIZE = 2_184_723_441
CHUNK_SIZE = 32 * 1024 * 1024
TARGET_CLASSES = {
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
}


def download_archive(cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = cache_path.stat().st_size if cache_path.exists() else 0
    if existing_size > ARCHIVE_SIZE:
        cache_path.unlink()
        existing_size = 0

    print(f"Downloading PlantVillage archive to {cache_path} (resuming at {existing_size:,} bytes)")
    with cache_path.open("ab") as output:
        while existing_size < ARCHIVE_SIZE:
            end = min(existing_size + CHUNK_SIZE, ARCHIVE_SIZE) - 1
            request = Request(ARCHIVE_URL, headers={"Range": f"bytes={existing_size}-{end}"})
            with urlopen(request) as response:
                chunk = response.read()
            expected = end - existing_size + 1
            if len(chunk) != expected:
                raise RuntimeError(f"Expected {expected:,} bytes but received {len(chunk):,} bytes.")
            output.write(chunk)
            existing_size = end + 1
            print(f"Downloaded {existing_size:,} / {ARCHIVE_SIZE:,} bytes")

    if cache_path.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError(
            f"Archive download is incomplete ({cache_path.stat().st_size:,} bytes). "
            "Run the command again to resume."
        )


def extract_target_classes(archive_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member for member in archive.infolist()
            if member.filename.startswith("raw/color/")
            and len(Path(member.filename).parts) >= 4
            and Path(member.filename).parts[2] in TARGET_CLASSES
            and not member.is_dir()
        ]
        if not members:
            raise RuntimeError("No target classes were found in the PlantVillage archive.")

        for member in members:
            class_name = Path(member.filename).parts[2]
            destination = output_root / class_name / Path(member.filename).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))

    print(f"Extracted {len(members)} images into {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected PlantVillage crop classes.")
    parser.add_argument("--archive", type=Path, default=Path("downloads/plantvillage-data.zip"))
    parser.add_argument("--output", type=Path, default=Path("datsets/PlantVillage"))
    args = parser.parse_args()

    download_archive(args.archive)
    extract_target_classes(args.archive, args.output)


if __name__ == "__main__":
    main()