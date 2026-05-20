# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # OmniVoice Filipino PLD fine-tuned checkpoint evaluation on Kaggle
# This notebook-style script evaluates one or more fine-tuned OmniVoice
# checkpoints on the full PLD Filipino test split. It is intended for comparing
# sweep outputs or later training runs after the base-model metrics have already
# been measured once with the same evaluation setup.
# The base-model row is included only as a reference summary. Do not recompute
# the base model here unless the test split, manifest construction, inference
# parameters, OmniVoice commit, or evaluator model versions change.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Settings
# `FINETUNE_EVAL_TARGETS` contains the two new W&B model artifacts from the
# later full-training runs. Each artifact should point to a complete Hugging
# Face-style checkpoint folder containing `config.json`, tokenizer files, and
# model weights. The base reference metrics below came from the previous full
# evaluation run and are included only for comparison, not recomputed here.

# %% [code] {"jupyter":{"outputs_hidden":false}}
RUN_NAME = "full-filipino-pld-test-finetune-eval"

PLD_DATA_DIR = Path("/kaggle/input/datasets/pipluppp/pld-filipino-cleaned/data_clean")

OMNIVOICE_REPO_URL = "https://github.com/k2-fsa/OmniVoice.git"
OMNIVOICE_DIR = Path("/kaggle/working/OmniVoice")

WORK_DIR = Path("/kaggle/working/omnivoice_filipino_pld_finetune_eval")
TEMP_DIR = Path("/kaggle/temp/omnivoice_filipino_pld_finetune_eval")
MANIFEST_DIR = WORK_DIR / "manifests"
RESULTS_DIR = WORK_DIR / "results" / "full_test_finetunes"
METRICS_DIR = WORK_DIR / "metrics" / "full_test_finetunes"
CHECKPOINT_DOWNLOAD_DIR = WORK_DIR / "wandb_checkpoint_artifacts"
EVAL_MODELS_DIR = WORK_DIR / "tts_eval_models"

HF_CACHE_DIR = TEMP_DIR / "hf_cache"
XDG_CACHE_DIR = TEMP_DIR / "xdg_cache"
UV_CACHE_DIR = TEMP_DIR / "uv_cache"
WANDB_LOCAL_DIR = TEMP_DIR / "wandb"

LANGUAGE_ID = "fil"
LANGUAGE_NAME = "Filipino"
WHISPER_LANGUAGE_NAME = "tagalog"

GPU_IDS = "0,1"
NUM_GPUS = 2

INFERENCE_NUM_STEPS = 32
INFERENCE_BATCH_SIZE = 2
INFERENCE_NJ_PER_GPU = 1

WER_BATCH_SIZE = 1
WER_CHUNK_SIZE = 4
METRIC_NJ_PER_GPU = 1

# Use None for the full PLD test set. Set a small integer only for smoke tests.
EVAL_MAX_SAMPLES: int | None = None

USE_WANDB = True
WANDB_ENTITY = "duncanb013-polytechnic-university-of-the-philippines"
WANDB_PROJECT = "OmniVoice-PLD-Filipino"
WANDB_MODE = "online"
WANDB_TAGS = [
    "omnivoice",
    "filipino",
    "pld",
    "kaggle",
    "evaluation",
    "full-test",
    "finetune-comparison",
]
WANDB_EVAL_RUN_ID = os.environ.get("WANDB_FINETUNE_EVAL_RUN_ID") or (
    f"omnivoice-fil-finetune-eval-{uuid.uuid4().hex[:8]}"
)

# Reference row from the already completed base-model evaluation. Reuse this
# only when the evaluation setup is unchanged.
INCLUDE_BASE_REFERENCE_ROW = True
BASE_REFERENCE_METRICS = {
    "model_label": "base_reference",
    "wer_percent": 22.55,
    "wer_insertions": 2668.0,
    "wer_deletions": 640.0,
    "wer_substitutions": 2858.0,
    "wer_words": 27342.0,
    "sim_o": 0.602,
    "utmos": 3.64,
    "source": "previous_full_eval",
    "notes": "Base model metrics reused from omnivoice_evaluation_metrics.py full run.",
}

# Add one item per fine-tuned checkpoint. These two rows are the only models
# evaluated by this notebook; the base model is not synthesized again.
FINETUNE_EVAL_TARGETS: list[dict[str, Any]] = [
    {
        "label": "finetuned_2000",
        "wandb_artifact": (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/"
            "omnivoice-filipino-full-checkpoint-2000:latest"
        ),
        "local_path": None,
        "notes": (
            "Run full-filipino-pld, 2000 training steps, checkpoint-2000, "
            "learning_rate=0.000005."
        ),
    },
    {
        "label": "finetuned_best",
        "wandb_artifact": (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/"
            "omnivoice-filipino-full-checkpoint-best:latest"
        ),
        "local_path": None,
        "notes": (
            "Run full-filipino-pld-best-eval, checkpoint-best selected by "
            "best_eval_loss=4.1618266105651855 at step 4900."
        ),
    },
]

DOWNLOAD_EVAL_MODELS = True
FORCE_RERUN_INFERENCE = False
FORCE_RERUN_METRICS = False
LOG_AUDIO_EXAMPLES_TO_WANDB = True
WANDB_AUDIO_EXAMPLE_LIMIT = 12
LOG_GENERATED_WAV_DIRS_TO_WANDB = False

SELECTED_AUDIO_WAV_NAMES = [
    "0105.111124.050515.0394.wav",
    "0085.110923.071602.0437.wav",
    "0093.111003.081411.0343.wav",
    "0153.120215.053407.0255.wav",
    "0166.120314.005502.0171.wav",
]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Environment setup
# Kaggle working storage is limited, so caches are directed to `/kaggle/temp`
# where possible. W&B is initialized through the Kaggle secret named
# `WANDB_API_KEY`, allowing this notebook to pull checkpoint artifacts and log
# the comparison run without manual downloads.

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_DIR)
os.environ["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
os.environ["WANDB_DIR"] = str(WANDB_LOCAL_DIR)
os.environ["WANDB_CACHE_DIR"] = str(WANDB_LOCAL_DIR / "cache")
os.environ["WANDB_DATA_DIR"] = str(WANDB_LOCAL_DIR / "data")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

for path in [
    WORK_DIR,
    TEMP_DIR,
    MANIFEST_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    CHECKPOINT_DOWNLOAD_DIR,
    EVAL_MODELS_DIR,
    HF_CACHE_DIR,
    XDG_CACHE_DIR,
    UV_CACHE_DIR,
    WANDB_LOCAL_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("\n$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def disk_usage_line(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"{path}: total={usage.total / 1024**3:.2f}GiB "
        f"used={usage.used / 1024**3:.2f}GiB "
        f"free={usage.free / 1024**3:.2f}GiB"
    )


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
        print("No Kaggle WANDB_API_KEY secret found; W&B may ask you to log in.")
        print("Secret lookup detail:", repr(exc))
        return

    if secret_value:
        os.environ["WANDB_API_KEY"] = secret_value
        print("Loaded WANDB_API_KEY from Kaggle Secrets.")


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    env["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = WANDB_MODE if USE_WANDB else "disabled"
    return env


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
    if not sanitized:
        raise ValueError("Model label cannot be empty after sanitization.")
    return sanitized


kaggle_wandb_login_if_available()

print("Python:", sys.version)
print("Working disk:", disk_usage_line(Path("/kaggle/working")))
print("Temp disk:", disk_usage_line(Path("/kaggle/temp")))
print("HF_HOME:", os.environ["HF_HOME"])
print("WANDB_DIR:", os.environ["WANDB_DIR"])
print("W&B eval run id:", WANDB_EVAL_RUN_ID)

try:
    import torch

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch import failed:", repr(exc))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Install OmniVoice and evaluation dependencies
# The metrics use the upstream OmniVoice evaluation modules directly. This keeps
# fine-tune comparisons aligned with the earlier base-vs-fine-tuned evaluation.

# %% [code] {"jupyter":{"outputs_hidden":false}}
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
    [
        "uv",
        "pip",
        "install",
        "--python",
        str(OMNIVOICE_VENV_PYTHON),
        "jiwer==3.1.0",
        "huggingface_hub",
        "s3prl",
        "wrapt",
        "zhconv",
        "zhon",
        "unidecode",
        "wandb",
    ],
    cwd=OMNIVOICE_DIR,
)

if USE_WANDB:
    run([sys.executable, "-m", "pip", "install", "-q", "wandb"])

os.environ["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{os.environ.get('PYTHONPATH', '')}"

run(
    [
        "uv",
        "run",
        "python",
        "-c",
        (
            "import torch, jiwer, s3prl, wrapt, zhconv, wandb; "
            "print('uv torch:', torch.__version__); "
            "print('uv cuda available:', torch.cuda.is_available()); "
            "print('uv cuda devices:', torch.cuda.device_count())"
        ),
    ],
    cwd=OMNIVOICE_DIR,
    env=command_env(),
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Start W&B run and resolve fine-tuned checkpoints
# W&B artifacts are downloaded with `run.use_artifact(...)` so the evaluation
# run records each checkpoint as an input.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def import_wandb():
    import wandb

    return wandb


wandb = import_wandb() if USE_WANDB else None
wandb_run = None

active_targets = []
for target in FINETUNE_EVAL_TARGETS:
    label = sanitize_label(str(target["label"]))
    normalized = dict(target)
    normalized["label"] = label

    if not normalized.get("wandb_artifact") and not normalized.get("local_path"):
        raise ValueError(f"{label} must define wandb_artifact or local_path.")

    active_targets.append(normalized)

if not active_targets:
    raise RuntimeError("No active fine-tuned checkpoints to evaluate.")

if USE_WANDB and WANDB_MODE != "disabled":
    wandb_kwargs = {
        "project": WANDB_PROJECT,
        "id": WANDB_EVAL_RUN_ID,
        "name": RUN_NAME,
        "resume": "allow",
        "job_type": "evaluation",
        "tags": WANDB_TAGS,
        "mode": WANDB_MODE,
        "config": {
            "run_name": RUN_NAME,
            "fine_tune_targets": active_targets,
            "base_reference_metrics": BASE_REFERENCE_METRICS
            if INCLUDE_BASE_REFERENCE_ROW
            else None,
            "language_id": LANGUAGE_ID,
            "language_name": LANGUAGE_NAME,
            "whisper_language_name": WHISPER_LANGUAGE_NAME,
            "eval_max_samples": EVAL_MAX_SAMPLES,
            "inference_num_steps": INFERENCE_NUM_STEPS,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "wer_batch_size": WER_BATCH_SIZE,
            "wer_chunk_size": WER_CHUNK_SIZE,
        },
    }
    if WANDB_ENTITY:
        wandb_kwargs["entity"] = WANDB_ENTITY
    wandb_run = wandb.init(**wandb_kwargs)


def find_checkpoint_dir(root: Path) -> Path:
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for path in candidates:
        has_config = (path / "config.json").exists()
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        has_tokenizer = (path / "tokenizer.json").exists() or (
            path / "tokenizer_config.json"
        ).exists()
        if has_config and has_weights and has_tokenizer:
            return path
    raise FileNotFoundError(
        "Could not find a complete OmniVoice checkpoint directory under "
        f"{root}. Expected config.json, model weights, and tokenizer files."
    )


def resolve_checkpoint(target: dict[str, Any]) -> Path:
    label = target["label"]
    local_path = target.get("local_path")
    artifact_name = target.get("wandb_artifact")

    if local_path:
        checkpoint = find_checkpoint_dir(Path(local_path))
        print(label, "local checkpoint:", checkpoint)
        return checkpoint

    if wandb_run is None:
        raise RuntimeError(
            f"W&B is disabled, but {label} does not define a local_path."
        )

    artifact_root = CHECKPOINT_DOWNLOAD_DIR / label
    print("Downloading W&B artifact:", artifact_name)
    artifact = wandb_run.use_artifact(str(artifact_name), type="model")
    downloaded_root = Path(artifact.download(root=str(artifact_root)))
    checkpoint = find_checkpoint_dir(downloaded_root)
    print(label, "artifact checkpoint:", checkpoint)
    return checkpoint


resolved_targets = []
for target in active_targets:
    checkpoint_path = resolve_checkpoint(target)
    resolved = dict(target)
    resolved["checkpoint_path"] = checkpoint_path
    resolved_targets.append(resolved)
    print("Checkpoint files:", sorted(path.name for path in checkpoint_path.iterdir()))

print("Resolved fine-tune targets:", [target["label"] for target in resolved_targets])
print("Working disk after checkpoints:", disk_usage_line(Path("/kaggle/working")))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Download OmniVoice evaluation models
# SIM-o, UTMOS, and WER use separate pretrained evaluator models. Reusing the
# same evaluator model paths as the prior base evaluation is what makes the base
# reference row comparable to these fine-tuned rows.

# %% [code] {"jupyter":{"outputs_hidden":false}}
required_eval_model_paths = [
    EVAL_MODELS_DIR / "speaker_similarity" / "wavlm_large_finetune.pth",
    EVAL_MODELS_DIR / "speaker_similarity" / "wavlm_large",
    EVAL_MODELS_DIR / "mos" / "utmos22_strong_step7459_v1.pt",
    EVAL_MODELS_DIR / "wer" / "whisper-large-v3",
]

if DOWNLOAD_EVAL_MODELS and not all(path.exists() for path in required_eval_model_paths):
    run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                "snapshot_download("
                "repo_id='k2-fsa/TTS_eval_models', "
                f"local_dir={str(EVAL_MODELS_DIR)!r}, "
                "allow_patterns=["
                "'speaker_similarity/*', "
                "'mos/*', "
                "'wer/whisper-large-v3/*'"
                "]"
                ")"
            ),
        ],
        cwd=OMNIVOICE_DIR,
        env=command_env(),
    )

missing_eval_model_paths = [
    str(path) for path in required_eval_model_paths if not path.exists()
]
if missing_eval_model_paths:
    raise FileNotFoundError(
        "Missing expected evaluation model paths:\n"
        + "\n".join(missing_eval_model_paths)
    )

print("Evaluation model dir:", EVAL_MODELS_DIR)
print("Working disk after eval models:", disk_usage_line(Path("/kaggle/working")))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Build the PLD full-test evaluation manifest
# This must match the earlier base evaluation manifest logic. The generated
# manifest starts from `data_clean/test.jsonl`, resolves raw test WAV paths, and
# pairs each target with a different reference utterance from the same speaker
# when available.

# %% [code] {"jupyter":{"outputs_hidden":false}}
required_pld_paths = [
    PLD_DATA_DIR,
    PLD_DATA_DIR / "test.jsonl",
    PLD_DATA_DIR / "wavs",
]
missing_pld_paths = [str(path) for path in required_pld_paths if not path.exists()]
if missing_pld_paths:
    raise FileNotFoundError(
        "Missing expected PLD dataset paths:\n" + "\n".join(missing_pld_paths)
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_audio_path(audio_path: str, dataset_root: Path) -> str:
    raw = Path(audio_path)
    if raw.is_absolute() and raw.exists():
        return str(raw)

    candidate = dataset_root / audio_path
    if candidate.exists():
        return str(candidate)

    parts = list(raw.parts)
    if "data_clean" in parts:
        idx = parts.index("data_clean")
        candidate = dataset_root.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return str(candidate)

    if raw.parent.name and raw.name:
        candidate = dataset_root / "wavs" / raw.parent.name / raw.name
        if candidate.exists():
            return str(candidate)

    candidate = dataset_root / "wavs" / raw.name
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(f"Could not resolve audio path: {audio_path}")


def build_eval_manifest(src_path: Path, dst_path: Path) -> int:
    rows = read_jsonl(src_path)
    if EVAL_MAX_SAMPLES is not None:
        rows = rows[:EVAL_MAX_SAMPLES]

    for row in rows:
        row["audio_path"] = resolve_audio_path(row["audio_path"], PLD_DATA_DIR)
        row.setdefault("language_id", LANGUAGE_ID)
        row.setdefault("language_name", LANGUAGE_NAME)
        row.setdefault("speaker_id", Path(row["audio_path"]).parent.name)

    rows_by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[str(row["speaker_id"])].append(row)

    with dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            speaker_rows = rows_by_speaker[str(row["speaker_id"])]
            if len(speaker_rows) > 1:
                idx = speaker_rows.index(row)
                ref_row = speaker_rows[(idx + 1) % len(speaker_rows)]
            else:
                ref_row = row

            sample = {
                "id": row["id"],
                "text": row["text"],
                "ref_audio": ref_row["audio_path"],
                "ref_text": ref_row["text"],
                "language_id": LANGUAGE_ID,
                "language_name": WHISPER_LANGUAGE_NAME,
                "display_language_name": LANGUAGE_NAME,
                "speaker_id": row.get("speaker_id"),
                "source": row.get("source"),
                "target_audio": row["audio_path"],
                "target_duration": row.get("duration", row.get("audio_duration")),
                "ref_duration": ref_row.get("duration", ref_row.get("audio_duration")),
                "ref_id": ref_row["id"],
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(rows)


EVAL_JSONL = MANIFEST_DIR / "pld_filipino_full_test_eval.jsonl"
eval_count = build_eval_manifest(PLD_DATA_DIR / "test.jsonl", EVAL_JSONL)
selected_audio_ids = [Path(name).stem for name in SELECTED_AUDIO_WAV_NAMES]
eval_rows_by_id = {row["id"]: row for row in read_jsonl(EVAL_JSONL)}
missing_selected_audio_ids = [
    sample_id for sample_id in selected_audio_ids if sample_id not in eval_rows_by_id
]
if missing_selected_audio_ids:
    raise RuntimeError(
        "Selected audio samples are not present in the evaluation manifest: "
        f"{missing_selected_audio_ids}"
    )

print("Evaluation rows:", eval_count)
print("Evaluation manifest:", EVAL_JSONL)
print("Selected audio sample ids:", selected_audio_ids)
print("First rows:")
print("\n".join(EVAL_JSONL.read_text(encoding="utf-8").splitlines()[:3]))

if wandb_run is not None:
    wandb_run.config.update({"eval_samples": eval_count}, allow_val_change=True)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Run fine-tuned inference
# Only fine-tuned checkpoints are synthesized in this notebook. The base model is
# excluded because its generated WAVs and metrics were already produced under
# the same setup.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def expected_wavs_exist(output_dir: Path, manifest_path: Path) -> bool:
    if not output_dir.exists():
        return False
    rows = read_jsonl(manifest_path)
    return all((output_dir / f"{row['id']}.wav").exists() for row in rows)


def run_inference_set(model_path: str | Path, label: str, output_dir: Path) -> Path:
    if FORCE_RERUN_INFERENCE and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if expected_wavs_exist(output_dir, EVAL_JSONL):
        print(f"Skipping {label} inference; all expected WAVs already exist.")
        return output_dir

    start = time.time()
    run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "omnivoice.cli.infer_batch",
            "--model",
            str(model_path),
            "--test_list",
            str(EVAL_JSONL),
            "--res_dir",
            str(output_dir),
            "--num_step",
            str(INFERENCE_NUM_STEPS),
            "--nj_per_gpu",
            str(INFERENCE_NJ_PER_GPU),
            "--batch_size",
            str(INFERENCE_BATCH_SIZE),
            "--lang_id",
            LANGUAGE_ID,
            "--preprocess_prompt",
            "False",
            "--postprocess_output",
            "True",
            "--audio_chunk_threshold",
            "1000",
        ],
        cwd=OMNIVOICE_DIR,
        env=command_env(),
    )
    elapsed = time.time() - start
    (output_dir / "_inference_seconds.txt").write_text(str(elapsed), encoding="utf-8")
    print(f"{label} inference seconds: {elapsed:.2f}")
    return output_dir


generated_dirs = {}
for target in resolved_targets:
    label = target["label"]
    output_dir = RESULTS_DIR / label
    generated_dirs[label] = run_inference_set(
        target["checkpoint_path"],
        label,
        output_dir,
    )

for label, directory in generated_dirs.items():
    wav_count = sum(1 for _ in directory.glob("*.wav"))
    print(label, "generated wavs:", wav_count, "dir:", directory)

print("Working disk after inference:", disk_usage_line(Path("/kaggle/working")))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. Run WER, SIM-o, and UTMOS
# WER is the main intelligibility metric; lower is better. SIM-o checks speaker
# similarity against the prompt; higher is better. UTMOS estimates naturalness;
# higher is better. Small UTMOS differences should be treated cautiously and
# followed with listening checks.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def metric_log_has_summary(metric_name: str, path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if metric_name == "wer":
        return "Overall WER:" in text
    if metric_name == "sim":
        return "Average SIM-o:" in text
    if metric_name == "utmos":
        return "Average UTMOS:" in text
    return False


def run_metric_if_needed(
    metric_name: str,
    label: str,
    wav_dir: Path,
    decode_path: Path,
) -> Path:
    if (
        decode_path.exists()
        and metric_log_has_summary(metric_name, decode_path)
        and not FORCE_RERUN_METRICS
    ):
        print(f"Skipping {label} {metric_name}; log already exists:", decode_path)
        return decode_path

    if metric_name == "wer":
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "omnivoice.eval.wer.minimax",
            "--wav-path",
            str(wav_dir),
            "--test-list",
            str(EVAL_JSONL),
            "--decode-path",
            str(decode_path),
            "--model-dir",
            str(EVAL_MODELS_DIR),
            "--lang",
            LANGUAGE_ID,
            "--batch-size",
            str(WER_BATCH_SIZE),
            "--nj-per-gpu",
            str(METRIC_NJ_PER_GPU),
            "--chunk-size",
            str(WER_CHUNK_SIZE),
        ]
    elif metric_name == "sim":
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "omnivoice.eval.speaker_similarity.sim",
            "--wav-path",
            str(wav_dir),
            "--test-list",
            str(EVAL_JSONL),
            "--decode-path",
            str(decode_path),
            "--model-dir",
            str(EVAL_MODELS_DIR),
            "--nj-per-gpu",
            str(METRIC_NJ_PER_GPU),
        ]
    elif metric_name == "utmos":
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "omnivoice.eval.mos.utmos",
            "--wav-path",
            str(wav_dir),
            "--test-list",
            str(EVAL_JSONL),
            "--decode-path",
            str(decode_path),
            "--model-dir",
            str(EVAL_MODELS_DIR),
            "--nj-per-gpu",
            str(METRIC_NJ_PER_GPU),
        ]
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    run(cmd, cwd=OMNIVOICE_DIR, env=command_env())
    return decode_path


metric_logs: dict[str, dict[str, Path]] = {}
for label, wav_dir in generated_dirs.items():
    metric_logs[label] = {
        "wer": run_metric_if_needed(
            "wer", label, wav_dir, METRICS_DIR / f"{label}.wer.log"
        ),
        "sim": run_metric_if_needed(
            "sim", label, wav_dir, METRICS_DIR / f"{label}.sim.log"
        ),
        "utmos": run_metric_if_needed(
            "utmos", label, wav_dir, METRICS_DIR / f"{label}.utmos.log"
        ),
    }

print("Metric logs:", metric_logs)
print("Working disk after metrics:", disk_usage_line(Path("/kaggle/working")))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. Parse and log metric summaries
# This section extracts headline values into a compact JSON summary and W&B
# table. The base reference row is included with `source=previous_full_eval` so
# it is clearly separated from metrics generated by this notebook run.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def parse_metric_log(metric_name: str, path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed: dict[str, float] = {}

    if metric_name == "wer":
        match = re.search(r"Overall WER:\s*([0-9.]+)%", text)
        if match:
            parsed["wer_percent"] = float(match.group(1))
        err = re.search(
            r"Overall Errors:\s*([0-9.]+)\s+ins,\s*([0-9.]+)\s+del,\s*"
            r"([0-9.]+)\s+sub\s*/\s*([0-9.]+)\s+words",
            text,
        )
        if err:
            parsed["wer_insertions"] = float(err.group(1))
            parsed["wer_deletions"] = float(err.group(2))
            parsed["wer_substitutions"] = float(err.group(3))
            parsed["wer_words"] = float(err.group(4))
    elif metric_name == "sim":
        match = re.search(r"Average SIM-o:\s*([0-9.]+)", text)
        if match:
            parsed["sim_o"] = float(match.group(1))
    elif metric_name == "utmos":
        match = re.search(r"Average UTMOS:\s*([0-9.]+)", text)
        if match:
            parsed["utmos"] = float(match.group(1))

    return parsed


target_lookup = {target["label"]: target for target in resolved_targets}
summary_rows = []
wandb_summary = {}

if INCLUDE_BASE_REFERENCE_ROW:
    summary_rows.append(dict(BASE_REFERENCE_METRICS))

for label, logs in metric_logs.items():
    target = target_lookup[label]
    row: dict[str, Any] = {
        "model_label": label,
        "source": "current_eval",
        "wandb_artifact": target.get("wandb_artifact"),
        "local_path": target.get("local_path"),
        "checkpoint_path": str(target.get("checkpoint_path")),
        "notes": target.get("notes"),
    }
    for metric_name, path in logs.items():
        row.update(parse_metric_log(metric_name, path))
    summary_rows.append(row)

    for key, value in row.items():
        if key not in {"model_label", "source", "notes", "wandb_artifact", "local_path"}:
            if isinstance(value, (int, float)):
                wandb_summary[f"{label}/{key}"] = value

SUMMARY_JSON = METRICS_DIR / "metric_summary.json"
SUMMARY_JSON.write_text(
    json.dumps(summary_rows, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Metric summary:")
print(SUMMARY_JSON.read_text(encoding="utf-8"))

if wandb_run is not None:
    wandb_run.summary.update(wandb_summary)

    metrics_table = wandb.Table(
        columns=[
            "model_label",
            "source",
            "wer_percent",
            "wer_insertions",
            "wer_deletions",
            "wer_substitutions",
            "wer_words",
            "sim_o",
            "utmos",
            "wandb_artifact",
            "local_path",
            "notes",
        ]
    )
    for row in summary_rows:
        metrics_table.add_data(
            row.get("model_label"),
            row.get("source"),
            row.get("wer_percent"),
            row.get("wer_insertions"),
            row.get("wer_deletions"),
            row.get("wer_substitutions"),
            row.get("wer_words"),
            row.get("sim_o"),
            row.get("utmos"),
            row.get("wandb_artifact"),
            row.get("local_path"),
            row.get("notes"),
        )
    wandb_run.log({"finetune_evaluation_metric_summary": metrics_table})

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. Log evaluation artifacts to W&B
# The final W&B artifact stores the evaluation manifest, raw metric logs, parsed
# summary, and optional generated audio examples for listening checks.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def log_wandb_outputs() -> None:
    if wandb_run is None:
        print("Skipping W&B output logging.")
        return

    results_artifact = wandb.Artifact(
        "omnivoice-filipino-full-test-finetune-eval-results",
        type="evaluation",
        metadata={
            "run_name": RUN_NAME,
            "eval_samples": eval_count,
            "base_reference_metrics": BASE_REFERENCE_METRICS
            if INCLUDE_BASE_REFERENCE_ROW
            else None,
            "fine_tune_targets": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in target.items()
                }
                for target in resolved_targets
            ],
            "language_id": LANGUAGE_ID,
            "language_name": LANGUAGE_NAME,
        },
    )
    for path in [EVAL_JSONL, SUMMARY_JSON]:
        results_artifact.add_file(str(path), name=path.name)
    for label, logs in metric_logs.items():
        for metric_name, path in logs.items():
            results_artifact.add_file(str(path), name=f"{label}.{metric_name}.log")

    if LOG_GENERATED_WAV_DIRS_TO_WANDB:
        for label, wav_dir in generated_dirs.items():
            results_artifact.add_dir(str(wav_dir), name=f"generated_wavs/{label}")

    wandb_run.log_artifact(results_artifact)

    if LOG_AUDIO_EXAMPLES_TO_WANDB:
        all_samples = read_jsonl(EVAL_JSONL)
        selected_ids = [Path(name).stem for name in SELECTED_AUDIO_WAV_NAMES]
        selected_samples = [
            sample for sample in all_samples if sample["id"] in set(selected_ids)
        ]
        filler_samples = [
            sample for sample in all_samples if sample["id"] not in set(selected_ids)
        ][: max(0, WANDB_AUDIO_EXAMPLE_LIMIT - len(selected_samples))]
        samples = selected_samples + filler_samples
        audio_table = wandb.Table(
            columns=[
                "model_label",
                "id",
                "selected_sample",
                "text",
                "ref_text",
                "ref_audio",
                "generated_audio",
            ]
        )
        for label, wav_dir in generated_dirs.items():
            for sample in samples:
                wav_path = wav_dir / f"{sample['id']}.wav"
                if not wav_path.exists():
                    continue
                audio_table.add_data(
                    label,
                    sample["id"],
                    sample["id"] in set(selected_ids),
                    sample["text"],
                    sample["ref_text"],
                    wandb.Audio(
                        sample["ref_audio"],
                        caption=f"reference | {sample['ref_id']}",
                    ),
                    wandb.Audio(
                        str(wav_path),
                        caption=f"{label} | {sample['id']} | {sample['text']}",
                    ),
                )
        wandb_run.log({"finetune_evaluation_audio_examples": audio_table})

    wandb_run.finish()
    print("Logged W&B eval run:", WANDB_EVAL_RUN_ID)


log_wandb_outputs()

print("Final working disk:", disk_usage_line(Path("/kaggle/working")))
print("Final temp disk:", disk_usage_line(Path("/kaggle/temp")))
