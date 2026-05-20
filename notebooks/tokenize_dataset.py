# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # OmniVoice Filipino PLD tokenized dataset builder
# Run this in a separate Kaggle notebook to create reusable OmniVoice audio-token
# shards for the Filipino PLD dataset. After it finishes, save or download
# `/kaggle/working/omnivoice_filipino_pld_tokens_full` and upload it as a Kaggle
# Dataset. Attach that dataset to the training notebook to skip tokenization.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Settings
# This notebook only does manifest rewriting and audio-token extraction.

# %% [code] {"jupyter":{"outputs_hidden":false}}
RUN_PROFILE = "full"
PROFILE_CONFIGS = {
    "full": {
        "train_samples": None,
        "dev_samples": None,
        "samples_per_shard": 1000,
    },
    "t4_trial": {
        "train_samples": 2000,
        "dev_samples": 200,
        "samples_per_shard": 256,
    },
    "smoke": {
        "train_samples": 96,
        "dev_samples": 24,
        "samples_per_shard": 64,
    },
}

RUN_SETTINGS = PROFILE_CONFIGS[RUN_PROFILE]
FORCE_RETOKENIZE = False
USE_WANDB = True
WANDB_ENTITY = "duncanb013-polytechnic-university-of-the-philippines"
WANDB_PROJECT = "OmniVoice-PLD-Filipino"
WANDB_MODE = "online"
WANDB_ARTIFACT_NAME = f"omnivoice-filipino-pld-tokens-{RUN_PROFILE}"

GPU_IDS = "0"
OMNIVOICE_REPO_URL = "https://github.com/k2-fsa/OmniVoice.git"
OMNIVOICE_DIR = Path("/kaggle/working/OmniVoice")
DATA_CLEAN_DIR = Path("/kaggle/input/datasets/pipluppp/pld-filipino-cleaned/data_clean")
AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"

PACKAGE_DIR = Path(f"/kaggle/working/omnivoice_filipino_pld_tokens_{RUN_PROFILE}")
ZIP_PATH = Path(f"/kaggle/working/omnivoice_filipino_pld_tokens_{RUN_PROFILE}.zip")
TOKEN_DIR = PACKAGE_DIR / f"tokens_{RUN_PROFILE}"
MANIFEST_DIR = PACKAGE_DIR / "manifests"
TEMP_DIR = Path(f"/kaggle/temp/omnivoice_filipino_pld_tokenize_{RUN_PROFILE}")
HF_CACHE_DIR = TEMP_DIR / "hf_cache"
UV_CACHE_DIR = TEMP_DIR / "uv_cache"
WANDB_LOCAL_DIR = TEMP_DIR / "wandb"

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
os.environ["WANDB_DIR"] = str(WANDB_LOCAL_DIR)
os.environ["WANDB_CACHE_DIR"] = str(WANDB_LOCAL_DIR / "cache")
os.environ["WANDB_DATA_DIR"] = str(WANDB_LOCAL_DIR / "data")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for path in [
    PACKAGE_DIR,
    TOKEN_DIR,
    MANIFEST_DIR,
    TEMP_DIR,
    HF_CACHE_DIR,
    UV_CACHE_DIR,
    WANDB_LOCAL_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Helpers

# %% [code] {"jupyter":{"outputs_hidden":false}}
def run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def kaggle_wandb_login_if_available() -> None:
    if not USE_WANDB or WANDB_MODE == "disabled":
        return

    if os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY already set.")
        return

    try:
        from kaggle_secrets import UserSecretsClient

        secret_value = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        print("No Kaggle WANDB_API_KEY secret found; WandB may ask you to log in.")
        print("Secret lookup detail:", repr(exc))
        return

    if secret_value:
        os.environ["WANDB_API_KEY"] = secret_value
        print("Loaded WANDB_API_KEY from Kaggle Secrets.")


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(
            chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")
        )


def resolve_audio_path(audio_path: str, dataset_root: Path) -> str:
    raw = Path(audio_path)
    if raw.is_absolute() and raw.exists():
        return str(raw)

    candidate = dataset_root / audio_path
    if candidate.exists():
        return str(candidate)

    parts = list(raw.parts)
    if "wavs" in parts:
        idx = parts.index("wavs")
        candidate = dataset_root.joinpath(*parts[idx:])
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Audio not found for manifest path: {audio_path}")


def safe_webdataset_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_")


def rewrite_manifest(
    src_path: Path,
    dst_path: Path,
    dataset_root: Path,
    limit: int | None = None,
) -> int:
    written = 0
    with src_path.open("r", encoding="utf-8") as src, dst_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            original_id = row["id"]
            row["original_id"] = original_id
            row["id"] = safe_webdataset_id(original_id)
            row["audio_path"] = resolve_audio_path(row["audio_path"], dataset_root)
            row.setdefault("language_id", "fil")
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def write_portable_manifest(split_dir: Path) -> Path:
    manifest = split_dir / "data.lst"
    portable_manifest = split_dir / "data_portable.lst"
    with manifest.open("r", encoding="utf-8") as src, portable_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            tar_path_raw, jsonl_path_raw, count, duration = line.strip().split()
            tar_rel = Path(tar_path_raw).resolve().relative_to(split_dir.resolve())
            jsonl_rel = Path(jsonl_path_raw).resolve().relative_to(split_dir.resolve())
            dst.write(
                f"{tar_rel.as_posix()} {jsonl_rel.as_posix()} {count} {duration}\n"
            )
    return portable_manifest


def verify_portable_manifest(split_dir: Path) -> dict[str, float | int]:
    portable_manifest = split_dir / "data_portable.lst"
    if not portable_manifest.exists():
        raise FileNotFoundError(f"Missing portable manifest: {portable_manifest}")

    total_items = 0
    total_duration = 0.0
    total_shards = 0
    with portable_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            tar_rel, jsonl_rel, count, duration = line.strip().split()
            tar_path = split_dir / tar_rel
            jsonl_path = split_dir / jsonl_rel
            if not tar_path.exists():
                raise FileNotFoundError(f"Missing tar shard: {tar_path}")
            if not jsonl_path.exists():
                raise FileNotFoundError(f"Missing text shard: {jsonl_path}")
            total_shards += 1
            total_items += int(count)
            total_duration += float(duration)
    return {
        "shards": total_shards,
        "items": total_items,
        "duration_seconds": round(total_duration, 3),
    }


def zip_package(package_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))
    return zip_path


def import_or_install_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "wandb"])
        import wandb

        return wandb


def log_wandb_token_artifact(
    zip_path: Path,
    package_dir: Path,
    artifact_metadata: dict,
) -> None:
    if not USE_WANDB or WANDB_MODE == "disabled":
        print("Skipping WandB artifact logging.")
        return

    kaggle_wandb_login_if_available()
    wandb = import_or_install_wandb()

    run_kwargs = {
        "project": WANDB_PROJECT,
        "name": f"tokenize-{RUN_PROFILE}",
        "job_type": "tokenize",
        "mode": WANDB_MODE,
        "config": artifact_metadata,
    }
    if WANDB_ENTITY:
        run_kwargs["entity"] = WANDB_ENTITY

    wandb_run = wandb.init(**run_kwargs)
    artifact = wandb.Artifact(
        WANDB_ARTIFACT_NAME,
        type="dataset",
        metadata=artifact_metadata,
        description="Reusable OmniVoice Filipino PLD tokenized WebDataset shards.",
    )
    artifact.add_file(str(zip_path), name=zip_path.name)
    artifact.add_file(
        str(package_dir / "token_dataset_metadata.json"),
        name="token_dataset_metadata.json",
    )
    artifact.add_file(str(package_dir / "README.md"), name="README.md")
    wandb_run.log_artifact(artifact)
    wandb_run.finish()

    print("Logged WandB artifact:", WANDB_ARTIFACT_NAME)
    print("Artifact type: dataset")


def disk_usage_line(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"{path}: total={usage.total / 1024**3:.2f}GiB "
        f"used={usage.used / 1024**3:.2f}GiB "
        f"free={usage.free / 1024**3:.2f}GiB"
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Environment and OmniVoice setup

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("Python:", sys.version)
print("Working disk:", disk_usage_line(Path("/kaggle/working")))
print("Temp disk:", disk_usage_line(Path("/kaggle/temp")))
print("HF_HOME:", os.environ["HF_HOME"])

if not OMNIVOICE_DIR.exists():
    run(["git", "clone", OMNIVOICE_REPO_URL, str(OMNIVOICE_DIR)])
else:
    print("Using existing OmniVoice repo:", OMNIVOICE_DIR)

try:
    run(["uv", "--version"])
except (FileNotFoundError, subprocess.CalledProcessError):
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])

run(["uv", "sync"], cwd=OMNIVOICE_DIR)

OMNIVOICE_VENV_PYTHON = OMNIVOICE_DIR / ".venv" / "bin" / "python"
run(
    ["uv", "pip", "install", "--python", str(OMNIVOICE_VENV_PYTHON), "wrapt"],
    cwd=OMNIVOICE_DIR,
)

os.environ["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{os.environ.get('PYTHONPATH', '')}"

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Rewrite train/dev manifests

# %% [code] {"jupyter":{"outputs_hidden":false}}
required_dataset_paths = [
    DATA_CLEAN_DIR,
    DATA_CLEAN_DIR / "train.jsonl",
    DATA_CLEAN_DIR / "dev.jsonl",
    DATA_CLEAN_DIR / "wavs",
]
missing_dataset_paths = [
    str(path) for path in required_dataset_paths if not path.exists()
]
if missing_dataset_paths:
    raise FileNotFoundError(
        "Missing expected Kaggle dataset paths:\n" + "\n".join(missing_dataset_paths)
    )

TRAIN_JSONL = MANIFEST_DIR / f"train_{RUN_PROFILE}.jsonl"
DEV_JSONL = MANIFEST_DIR / f"dev_{RUN_PROFILE}.jsonl"

train_count = rewrite_manifest(
    DATA_CLEAN_DIR / "train.jsonl",
    TRAIN_JSONL,
    DATA_CLEAN_DIR,
    RUN_SETTINGS["train_samples"],
)
dev_count = rewrite_manifest(
    DATA_CLEAN_DIR / "dev.jsonl",
    DEV_JSONL,
    DATA_CLEAN_DIR,
    RUN_SETTINGS["dev_samples"],
)

print("Train manifest:", TRAIN_JSONL, train_count)
print("Dev manifest:", DEV_JSONL, dev_count)
print("Original full train rows:", count_jsonl(DATA_CLEAN_DIR / "train.jsonl"))
print("Original full dev rows:", count_jsonl(DATA_CLEAN_DIR / "dev.jsonl"))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Extract OmniVoice audio tokens

# %% [code] {"jupyter":{"outputs_hidden":false}}
def tokenize_split(split: str, input_jsonl: Path) -> Path:
    split_token_dir = TOKEN_DIR / split
    manifest = split_token_dir / "data.lst"

    if FORCE_RETOKENIZE and split_token_dir.exists():
        print(f"Removing existing tokenized {split} shards:", split_token_dir)
        shutil.rmtree(split_token_dir)

    if manifest.exists():
        print(f"Token manifest already exists for {split}: {manifest}")
        return manifest

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    env["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{env.get('PYTHONPATH', '')}"

    run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "omnivoice.scripts.extract_audio_tokens",
            "--input_jsonl",
            str(input_jsonl),
            "--tar_output_pattern",
            str(split_token_dir / "audios" / "shard-%06d.tar"),
            "--jsonl_output_pattern",
            str(split_token_dir / "txts" / "shard-%06d.jsonl"),
            "--tokenizer_path",
            AUDIO_TOKENIZER,
            "--nj_per_gpu",
            "1",
            "--loader_workers",
            "2",
            "--samples_per_shard",
            str(RUN_SETTINGS["samples_per_shard"]),
            "--min_num_shards",
            "1",
            "--shuffle",
            "True",
            "--skip_errors",
        ],
        cwd=OMNIVOICE_DIR,
        env=env,
    )

    if not manifest.exists():
        raise FileNotFoundError(
            f"Tokenization finished but no manifest was found: {manifest}"
        )
    return manifest


TRAIN_TOKEN_MANIFEST = tokenize_split("train", TRAIN_JSONL)
DEV_TOKEN_MANIFEST = tokenize_split("dev", DEV_JSONL)

TRAIN_PORTABLE_MANIFEST = write_portable_manifest(TOKEN_DIR / "train")
DEV_PORTABLE_MANIFEST = write_portable_manifest(TOKEN_DIR / "dev")
train_token_stats = verify_portable_manifest(TOKEN_DIR / "train")
dev_token_stats = verify_portable_manifest(TOKEN_DIR / "dev")

print("Train token manifest:", TRAIN_TOKEN_MANIFEST)
print("Train portable manifest:", TRAIN_PORTABLE_MANIFEST)
print("Train token stats:", train_token_stats)
print("Dev token manifest:", DEV_TOKEN_MANIFEST)
print("Dev portable manifest:", DEV_PORTABLE_MANIFEST)
print("Dev token stats:", dev_token_stats)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Write package metadata

# %% [code] {"jupyter":{"outputs_hidden":false}}
metadata = {
    "name": f"omnivoice_filipino_pld_tokens_{RUN_PROFILE}",
    "run_profile": RUN_PROFILE,
    "audio_tokenizer": AUDIO_TOKENIZER,
    "train_count": train_count,
    "dev_count": dev_count,
    "train_token_stats": train_token_stats,
    "dev_token_stats": dev_token_stats,
    "token_dir": str(TOKEN_DIR),
    "portable_manifests": {
        "train": str(TRAIN_PORTABLE_MANIFEST.relative_to(PACKAGE_DIR)),
        "dev": str(DEV_PORTABLE_MANIFEST.relative_to(PACKAGE_DIR)),
    },
}
(PACKAGE_DIR / "token_dataset_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
(PACKAGE_DIR / "README.md").write_text(
    "\n".join(
        [
            "# OmniVoice Filipino PLD tokenized shards",
            "",
            f"Profile: `{RUN_PROFILE}`",
            f"Train rows: `{train_count}`",
            f"Dev rows: `{dev_count}`",
            "",
            "Attach this Kaggle Dataset to `omnivoice_kaggle_initial.py` and set:",
            "",
            "```python",
            "USE_PRETOKENIZED_DATASET = True",
            'PRETOKENIZED_DATASET_DIR = Path("/kaggle/input/<your-dataset-slug>")',
            "```",
            "",
            "The training notebook localizes `data_portable.lst` to the mounted Kaggle input path.",
            "",
            f"If W&B logging was enabled, the zip was also logged as dataset artifact `{WANDB_ARTIFACT_NAME}`.",
        ]
    ),
    encoding="utf-8",
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Zip package and log dataset artifact

# %% [code] {"jupyter":{"outputs_hidden":false}}
ZIP_PATH = zip_package(PACKAGE_DIR, ZIP_PATH)
metadata_for_artifact = {
    **metadata,
    "zip_file": ZIP_PATH.name,
    "zip_size_gib": round(ZIP_PATH.stat().st_size / 1024**3, 3),
    "wandb_artifact_name": WANDB_ARTIFACT_NAME,
    "wandb_artifact_type": "dataset",
}

log_wandb_token_artifact(ZIP_PATH, PACKAGE_DIR, metadata_for_artifact)

print("Package ready:", PACKAGE_DIR)
print("Zip ready:", ZIP_PATH)
print("The zip is available in Kaggle output and, if enabled, W&B Artifacts.")
print("Working disk:", disk_usage_line(Path("/kaggle/working")))
