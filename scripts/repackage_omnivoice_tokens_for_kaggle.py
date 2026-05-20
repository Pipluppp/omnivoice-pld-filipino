# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Repackage OmniVoice token shards for Kaggle Dataset upload
# Kaggle can expand `.tar` files inside uploaded Datasets, which breaks
# OmniVoice manifests that expect WebDataset tar shard files. This script creates
# a Kaggle-safe copy of the original tokenized output where tar-format
# shard bytes are stored with a `.wds` extension and manifests point to `.wds`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Settings
# Run this locally from the project root after tokenization has completed.

# %% [code] {"jupyter":{"outputs_hidden":false}}
SOURCE_PACKAGE_DIR = (
    Path("data_artifacts") / "omnivoice_filipino_pld_tokens_full_original_tar"
)
OUTPUT_ROOT = Path("data") / "omnivoice_pld_audio_tokens_kaggle_safe"
OUTPUT_PACKAGE_DIR = OUTPUT_ROOT / "omnivoice_filipino_pld_tokens_full"
OUTPUT_ZIP = Path("data") / "omnivoice_pld_audio_tokens_kaggle_safe.zip"

TOKEN_DIR_NAME = "tokens_full"
SHARD_EXTENSION = ".wds"
SPLITS = ["train", "dev"]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Helpers

# %% [code] {"jupyter":{"outputs_hidden":false}}
def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_source_audio_shard(split_dir: Path, audio_rel: Path) -> Path:
    tar_path = split_dir / audio_rel
    if tar_path.is_file():
        return tar_path

    expanded_dir = tar_path.with_suffix("")
    if expanded_dir.is_dir():
        return expanded_dir

    fallback_dir = split_dir / audio_rel.parent / audio_rel.stem
    if fallback_dir.is_dir():
        return fallback_dir

    raise FileNotFoundError(
        "Could not find source shard as tar file or expanded directory:\n"
        f"tar: {tar_path}\n"
        f"dir: {expanded_dir}"
    )


def write_expanded_dir_as_tar(src_dir: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst_path, "w") as archive:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file():
                archive.add(
                    file_path,
                    arcname=file_path.relative_to(src_dir).as_posix(),
                )


def copy_or_pack_audio_shard(src_shard: Path, dst_shard: Path) -> None:
    dst_shard.parent.mkdir(parents=True, exist_ok=True)
    if src_shard.is_file():
        shutil.copy2(src_shard, dst_shard)
    elif src_shard.is_dir():
        write_expanded_dir_as_tar(src_shard, dst_shard)
    else:
        raise FileNotFoundError(f"Missing shard source: {src_shard}")


def read_manifest_line(line: str) -> tuple[Path, Path, str, str]:
    audio_path_raw, text_path_raw, count, duration = line.strip().split()
    return Path(audio_path_raw), Path(text_path_raw), count, duration


def repackage_split(split: str) -> dict[str, int | float]:
    src_split_dir = SOURCE_PACKAGE_DIR / TOKEN_DIR_NAME / split
    dst_split_dir = OUTPUT_PACKAGE_DIR / TOKEN_DIR_NAME / split
    src_manifest = src_split_dir / "data_portable.lst"
    if not src_manifest.exists():
        src_manifest = src_split_dir / "data.lst"
    if not src_manifest.exists():
        raise FileNotFoundError(f"Missing manifest for {split}: {src_split_dir}")

    dst_manifest = dst_split_dir / "data_portable.lst"
    dst_data_manifest = dst_split_dir / "data.lst"
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)

    total_items = 0
    total_duration = 0.0
    total_shards = 0

    with src_manifest.open("r", encoding="utf-8") as src, dst_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as portable_dst, dst_data_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as data_dst:
        for line in src:
            if not line.strip():
                continue

            audio_rel, text_rel, count, duration = read_manifest_line(line)
            audio_rel = Path(audio_rel)
            text_rel = Path(text_rel)

            if audio_rel.is_absolute() or text_rel.is_absolute():
                raise ValueError(
                    "Expected portable relative paths in token manifest. "
                    f"Got: {audio_rel} {text_rel}"
                )

            src_audio_shard = find_source_audio_shard(src_split_dir, audio_rel)
            src_text_shard = src_split_dir / text_rel
            if not src_text_shard.exists():
                raise FileNotFoundError(f"Missing text shard: {src_text_shard}")

            dst_audio_rel = audio_rel.with_suffix(SHARD_EXTENSION)
            dst_text_rel = text_rel
            dst_audio_shard = dst_split_dir / dst_audio_rel
            dst_text_shard = dst_split_dir / dst_text_rel

            copy_or_pack_audio_shard(src_audio_shard, dst_audio_shard)
            copy_file(src_text_shard, dst_text_shard)

            manifest_line = (
                f"{dst_audio_rel.as_posix()} {dst_text_rel.as_posix()} "
                f"{count} {duration}\n"
            )
            portable_dst.write(manifest_line)
            data_dst.write(manifest_line)

            total_items += int(count)
            total_duration += float(duration)
            total_shards += 1

    return {
        "shards": total_shards,
        "items": total_items,
        "duration_seconds": round(total_duration, 3),
    }


def copy_package_metadata() -> None:
    for child in SOURCE_PACKAGE_DIR.iterdir():
        if child.name == TOKEN_DIR_NAME:
            continue
        src = child
        dst = OUTPUT_PACKAGE_DIR / child.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            copy_file(src, dst)


def write_metadata(split_stats: dict[str, dict[str, int | float]]) -> None:
    metadata_path = OUTPUT_PACKAGE_DIR / "token_dataset_metadata.json"
    metadata = {}
    if (SOURCE_PACKAGE_DIR / "token_dataset_metadata.json").exists():
        metadata = json.loads(
            (SOURCE_PACKAGE_DIR / "token_dataset_metadata.json").read_text(
                encoding="utf-8"
            )
        )

    metadata.update(
        {
            "kaggle_safe_repackaged": True,
            "audio_shard_extension": SHARD_EXTENSION,
            "reason": "Avoid Kaggle auto-expanding WebDataset .tar shards.",
            "portable_manifests": {
                split: f"{TOKEN_DIR_NAME}/{split}/data_portable.lst"
                for split in SPLITS
            },
        }
    )
    for split, stats in split_stats.items():
        metadata[f"{split}_token_stats"] = stats

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme_path = OUTPUT_PACKAGE_DIR / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# OmniVoice Filipino PLD tokenized shards",
                "",
                "This is the Kaggle-safe package for OmniVoice training.",
                "",
                "The audio shard files use `.wds` extension but still contain tar-format WebDataset bytes.",
                "This avoids Kaggle expanding `.tar` shards into directories during Dataset upload.",
                "",
                "Attach this Kaggle Dataset to `omnivoice_kaggle_initial.py` and point",
                "`PRETOKENIZED_DATASET_DIR` to the mounted `omnivoice_filipino_pld_tokens_full` folder.",
            ]
        ),
        encoding="utf-8",
    )


def zip_output_folder() -> Path:
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_ROOT))
    return OUTPUT_ZIP


def verify_output(split_stats: dict[str, dict[str, int | float]]) -> None:
    for split, stats in split_stats.items():
        split_dir = OUTPUT_PACKAGE_DIR / TOKEN_DIR_NAME / split
        manifest = split_dir / "data_portable.lst"
        lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != stats["shards"]:
            raise RuntimeError(f"Manifest shard count mismatch for {split}")

        for line in lines:
            audio_rel, text_rel, _count, _duration = read_manifest_line(line)
            if audio_rel.suffix != SHARD_EXTENSION:
                raise RuntimeError(f"Expected .wds shard path, got: {audio_rel}")
            if not (split_dir / audio_rel).exists():
                raise FileNotFoundError(split_dir / audio_rel)
            if not (split_dir / text_rel).exists():
                raise FileNotFoundError(split_dir / text_rel)


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Build Kaggle-safe package

# %% [code] {"jupyter":{"outputs_hidden":false}}
if not SOURCE_PACKAGE_DIR.exists():
    raise FileNotFoundError(f"Missing source package: {SOURCE_PACKAGE_DIR.resolve()}")

reset_output_dir(OUTPUT_ROOT)
copy_package_metadata()

split_stats = {}
for split_name in SPLITS:
    print("Repackaging split:", split_name)
    split_stats[split_name] = repackage_split(split_name)
    print(split_name, split_stats[split_name])

write_metadata(split_stats)
verify_output(split_stats)
zip_path = zip_output_folder()

print("Kaggle-safe folder:", OUTPUT_PACKAGE_DIR.resolve())
print("Kaggle-safe zip:", zip_path.resolve())
print("Upload the folder inside the zip as a new Kaggle Dataset version.")
