# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # OmniVoice Filipino PLD full fine-tuning with best-eval checkpointing on Kaggle, LR 5e-6
# This notebook-style script runs the full training workflow with metric-based checkpointing:
# - attach the cleaned PLD JSONL/audio Kaggle Dataset;
# - attach the Kaggle-safe pretokenized OmniVoice WebDataset shards;
# - fine-tune OmniVoice on the full Filipino PLD train split;
# - evaluate every configured eval interval and keep only the best eval-loss checkpoint;
# - run fixed base and fine-tuned model inference samples;
# - log training curves, configs, the best checkpoint artifact, and playable WAV samples to W&B.

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
from pathlib import Path

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Settings

# %% [code] {"jupyter":{"outputs_hidden":false}}
RUN_NAME = "full-filipino-pld-best-eval-lr-5e-6"

PLD_DATA_DIR = Path("/kaggle/input/datasets/pipluppp/pld-filipino-cleaned/data_clean")
TOKEN_DATASET_DIR = Path(
    "/kaggle/input/datasets/pipluppp/omnivoice-pld-audio-tokens/"
    "omnivoice_filipino_pld_tokens_full"
)

OMNIVOICE_REPO_URL = "https://github.com/k2-fsa/OmniVoice.git"
OMNIVOICE_DIR = Path("/kaggle/working/OmniVoice")
PRETRAINED_MODEL = "k2-fsa/OmniVoice"
AUDIO_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"

WORK_DIR = Path("/kaggle/working/omnivoice_filipino_pld")
TEMP_DIR = Path("/kaggle/temp/omnivoice_filipino_pld")
MANIFEST_DIR = WORK_DIR / "manifests"
CONFIG_DIR = WORK_DIR / "config"
OUTPUT_DIR = WORK_DIR / "exp_full_best_eval_lr_5e_6"
RESULTS_DIR = WORK_DIR / "results" / "full_best_eval_lr_5e_6"
BASELINE_RESULTS_DIR = RESULTS_DIR / "base"
FINETUNED_RESULTS_DIR = RESULTS_DIR / "finetuned"
METRICS_DIR = WORK_DIR / "metrics" / "full_best_eval_lr_5e_6"

HF_CACHE_DIR = TEMP_DIR / "hf_cache"
XDG_CACHE_DIR = TEMP_DIR / "xdg_cache"
UV_CACHE_DIR = TEMP_DIR / "uv_cache"
WANDB_LOCAL_DIR = TEMP_DIR / "wandb"

GPU_IDS = "0"
NUM_GPUS = 1

TRAINING_STEPS = 5000
LEARNING_RATE = 5e-6
BATCH_TOKENS = 1024
GRADIENT_ACCUMULATION_STEPS = 8
MAX_BATCH_SIZE = 4
MAX_SAMPLE_TOKENS = 1200
NUM_WORKERS = 2
LOGGING_STEPS = 25
EVAL_STEPS = 250
SAVE_FINAL_CHECKPOINT = False
BEST_CHECKPOINT_NAME = "checkpoint-best"

INFERENCE_NUM_STEPS = 32
INFERENCE_REFERENCE_WAV_NAMES = [
    "0105.111124.050515.0394.wav",
    "0085.110923.071602.0437.wav",
    "0093.111003.081411.0343.wav",
    "0153.120215.053407.0255.wav",
    "0166.120314.005502.0171.wav",
]

USE_WANDB = True
WANDB_ENTITY = "duncanb013-polytechnic-university-of-the-philippines"
WANDB_PROJECT = "OmniVoice-PLD-Filipino"
WANDB_MODE = "online"
WANDB_TAGS = [
    "omnivoice",
    "filipino",
    "pld",
    "kaggle",
    "full",
    "best-eval",
    "lr-5e-6",
]
LOG_CHECKPOINT_ARTIFACT_TO_WANDB = True
LOG_AUDIO_TO_WANDB = True

CLEAN_OLD_WORKING_CACHES = True
CLEAN_INCOMPLETE_CHECKPOINTS = True
FORCE_RERUN_INFERENCE = True

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Environment setup

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
    CONFIG_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    BASELINE_RESULTS_DIR,
    FINETUNED_RESULTS_DIR,
    METRICS_DIR,
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
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def disk_usage_line(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"{path}: total={usage.total / 1024**3:.2f}GiB "
        f"used={usage.used / 1024**3:.2f}GiB "
        f"free={usage.free / 1024**3:.2f}GiB"
    )


def cleanup_old_working_caches() -> None:
    if not CLEAN_OLD_WORKING_CACHES:
        return

    for path in [
        Path("/kaggle/working/hf_cache"),
        Path("/kaggle/working/OmniVoice/wandb"),
        WORK_DIR / "tts_eval_models",
    ]:
        if path.exists():
            print("Removing old working-cache path:", path)
            shutil.rmtree(path)


WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID") or f"omnivoice-fil-{uuid.uuid4().hex[:8]}"


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


def wandb_env() -> dict[str, str]:
    env = os.environ.copy()
    env["WANDB_DIR"] = str(WANDB_LOCAL_DIR)
    env["WANDB_CACHE_DIR"] = str(WANDB_LOCAL_DIR / "cache")
    env["WANDB_DATA_DIR"] = str(WANDB_LOCAL_DIR / "data")
    env["OMNIVOICE_SAVE_FINAL_CHECKPOINT"] = (
        "true" if SAVE_FINAL_CHECKPOINT else "false"
    )
    env["OMNIVOICE_BEST_CHECKPOINT_NAME"] = BEST_CHECKPOINT_NAME

    if not USE_WANDB or WANDB_MODE == "disabled":
        env["WANDB_MODE"] = "disabled"
        env["OMNIVOICE_ACCELERATE_LOG_WITH"] = "tensorboard"
        return env

    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["WANDB_RUN_ID"] = WANDB_RUN_ID
    env["WANDB_NAME"] = RUN_NAME
    env["WANDB_MODE"] = WANDB_MODE
    env["WANDB_TAGS"] = ",".join(WANDB_TAGS)
    env["OMNIVOICE_ACCELERATE_LOG_WITH"] = "wandb"
    env["OMNIVOICE_TRACKER_PROJECT"] = WANDB_PROJECT
    if WANDB_ENTITY:
        env["WANDB_ENTITY"] = WANDB_ENTITY
    return env


cleanup_old_working_caches()
kaggle_wandb_login_if_available()

print("Python:", sys.version)
print("Working disk:", disk_usage_line(Path("/kaggle/working")))
print("Temp disk:", disk_usage_line(Path("/kaggle/temp")))
print("HF_HOME:", os.environ["HF_HOME"])
print("WANDB_DIR:", os.environ["WANDB_DIR"])
print("W&B run id:", WANDB_RUN_ID)

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
# ## 3. Install OmniVoice and best-eval trainer

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
    ["uv", "pip", "install", "--python", str(OMNIVOICE_VENV_PYTHON), "wrapt"],
    cwd=OMNIVOICE_DIR,
)
if USE_WANDB:
    run(
        ["uv", "pip", "install", "--python", str(OMNIVOICE_VENV_PYTHON), "wandb"],
        cwd=OMNIVOICE_DIR,
    )


def write_best_eval_training_entrypoints(repo_dir: Path) -> None:
    trainer_path = repo_dir / "omnivoice" / "training" / "best_eval_trainer.py"
    cli_path = repo_dir / "omnivoice" / "cli" / "train_best_eval.py"

    trainer_path.write_text(
        r'''#!/usr/bin/env python3
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from typing import Any, Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class BestEvalOmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.accelerator = self._init_accelerator()
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()

        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.best_step = None

    def _init_accelerator(self) -> Accelerator:
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        tracker_names = [
            item.strip()
            for item in os.environ.get("OMNIVOICE_ACCELERATE_LOG_WITH", "tensorboard").split(",")
            if item.strip()
        ]
        log_with = tracker_names[0] if len(tracker_names) == 1 else tracker_names
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=log_with,
            project_dir=self.config.output_dir,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
            ],
            deepspeed_plugin=deepspeed_plugin,
            split_batches=False,
        )

        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.config.save_to_json(os.path.join(self.config.output_dir, "initial_config.json"))
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(self.config.output_dir, "train.log")),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info("Loaded Config: %s", self.config)
        set_seed(self.config.seed)
        accelerator.init_trackers(os.environ.get("OMNIVOICE_TRACKER_PROJECT", "omnivoice"))
        return accelerator

    def create_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        warmup_steps = (
            math.ceil(self.config.steps * self.config.warmup_ratio)
            if self.config.warmup_type == "ratio"
            else self.config.warmup_steps
        )
        if self.config.lr_scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, scheduler

    def load_checkpoint(self, checkpoint_path: str) -> int:
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info("Resumed from step %s", self.global_step)
        return step

    def evaluate(self) -> dict:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        logger.info("Running evaluation at step %s...", self.global_step)
        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**_to_device(batch, self.accelerator.device))
                local_loss_sum += outputs.loss.detach()
                eval_count += 1

        local_mean = local_loss_sum / eval_count if eval_count else local_loss_sum
        eval_loss = self.accelerator.gather(local_mean).mean().item()
        metrics = {"eval/loss": eval_loss}
        self.accelerator.log(metrics, step=self.global_step)
        logger.info("Eval Loss: %.4f", eval_loss)
        self.accelerator.wait_for_everyone()
        self.model.train()
        return metrics

    def save_model_checkpoint(self, checkpoint_name: str, metadata: Optional[dict] = None) -> None:
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        if self.accelerator.is_main_process and os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        self.accelerator.wait_for_everyone()

        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(
            checkpoint_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(checkpoint_dir)
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))
            if metadata is not None:
                with open(
                    os.path.join(checkpoint_dir, "best_checkpoint.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(metadata, handle, indent=2)
            logger.info("Saved checkpoint to %s", checkpoint_dir)
        self.accelerator.wait_for_everyone()

    def save_best_checkpoint(self) -> None:
        checkpoint_name = os.environ.get("OMNIVOICE_BEST_CHECKPOINT_NAME", "checkpoint-best")
        self.save_model_checkpoint(
            checkpoint_name,
            {
                "checkpoint": checkpoint_name,
                "best_step": self.best_step,
                "best_eval_loss": self.best_eval_loss,
            },
        )

    def train(self) -> None:
        logger.info("Starting Training Loop...")
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0, device=self.accelerator.device)
        logging_loss_scalar = 0.0

        while self.global_step < self.config.steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                logger.info("Epoch %s starting. Resetting dataloader...", self.epoch)
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**_to_device(batch, self.accelerator.device))
                loss = outputs.loss
                tr_loss += loss.detach()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_norm = grad_norm.item() if grad_norm is not None else 0.0

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    train_logger.update(self.global_step, loss.item(), current_lr)

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )
                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        avg_loss = (
                            tr_loss_scalar - logging_loss_scalar
                        ) / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar
                        train_logger.log_metrics(
                            self.global_step,
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/grad_norm": grad_norm,
                                "train/epoch": self.epoch,
                                "train/steps_per_sec": steps_per_sec,
                            },
                        )
                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_loss = self.evaluate().get("eval/loss")
                        if eval_loss is not None and eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.best_step = self.global_step
                            logger.info(
                                "New best eval/loss %.4f at step %s.",
                                eval_loss,
                                self.global_step,
                            )
                            self.save_best_checkpoint()

        if self.best_step is None:
            logger.warning("No eval checkpoint was saved; saving final model as best.")
            self.best_step = self.global_step
            self.best_eval_loss = float("nan")
            self.save_best_checkpoint()

        if os.environ.get("OMNIVOICE_SAVE_FINAL_CHECKPOINT", "false").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            self.save_model_checkpoint("checkpoint-final")

        train_logger.close()
        self.accelerator.end_training()
''',
        encoding="utf-8",
    )

    cli_path.write_text(
        r'''#!/usr/bin/env python3
import argparse

from omnivoice.training.best_eval_trainer import BestEvalOmniTrainer
from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="OmniVoice best-eval training")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    trainer = BestEvalOmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    print("Wrote best-eval trainer:", trainer_path)
    print("Wrote best-eval training CLI:", cli_path)


write_best_eval_training_entrypoints(OMNIVOICE_DIR)

run(
    [
        "uv",
        "run",
        "python",
        "-c",
        (
            "import wrapt, torch; "
            "print('wrapt:', wrapt.__version__); "
            "print('uv torch:', torch.__version__); "
            "print('uv cuda available:', torch.cuda.is_available()); "
            "print('uv cuda devices:', torch.cuda.device_count()); "
            "print('uv gpu 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        ),
    ],
    cwd=OMNIVOICE_DIR,
)

os.environ["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{os.environ.get('PYTHONPATH', '')}"

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Validate attached Kaggle datasets

# %% [code] {"jupyter":{"outputs_hidden":false}}
required_pld_paths = [
    PLD_DATA_DIR,
    PLD_DATA_DIR / "train.jsonl",
    PLD_DATA_DIR / "dev.jsonl",
    PLD_DATA_DIR / "test.jsonl",
    PLD_DATA_DIR / "wavs",
]
missing_pld_paths = [str(path) for path in required_pld_paths if not path.exists()]
if missing_pld_paths:
    raise FileNotFoundError(
        "Missing expected PLD dataset paths:\n" + "\n".join(missing_pld_paths)
    )

required_token_paths = [
    TOKEN_DATASET_DIR,
    TOKEN_DATASET_DIR / "tokens_full" / "train" / "data_portable.lst",
    TOKEN_DATASET_DIR / "tokens_full" / "dev" / "data_portable.lst",
]
missing_token_paths = [str(path) for path in required_token_paths if not path.exists()]
if missing_token_paths:
    raise FileNotFoundError(
        "Missing expected pretokenized dataset paths:\n"
        + "\n".join(missing_token_paths)
    )

print("PLD data dir:", PLD_DATA_DIR)
print("Token dataset dir:", TOKEN_DATASET_DIR)
print("PLD files:", sorted(path.name for path in PLD_DATA_DIR.iterdir())[:20])
print("Train token manifest:", TOKEN_DATASET_DIR / "tokens_full" / "train" / "data_portable.lst")
print("Dev token manifest:", TOKEN_DATASET_DIR / "tokens_full" / "dev" / "data_portable.lst")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Rewrite PLD manifests and select listening samples

# %% [code] {"jupyter":{"outputs_hidden":false}}
def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(
            chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")
        )


def safe_webdataset_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_")


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


def rewrite_manifest(src_path: Path, dst_path: Path, dataset_root: Path) -> int:
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
    return written


TRAIN_JSONL = MANIFEST_DIR / "train_full.jsonl"
DEV_JSONL = MANIFEST_DIR / "dev_full.jsonl"
TEST_JSONL = MANIFEST_DIR / "test_full_for_reference.jsonl"

train_count = rewrite_manifest(PLD_DATA_DIR / "train.jsonl", TRAIN_JSONL, PLD_DATA_DIR)
dev_count = rewrite_manifest(PLD_DATA_DIR / "dev.jsonl", DEV_JSONL, PLD_DATA_DIR)
test_count = rewrite_manifest(PLD_DATA_DIR / "test.jsonl", TEST_JSONL, PLD_DATA_DIR)

print("Train rows:", train_count, "->", TRAIN_JSONL)
print("Dev rows:", dev_count, "->", DEV_JSONL)
print("Test rows:", test_count, "->", TEST_JSONL)
print("Original train rows:", count_jsonl(PLD_DATA_DIR / "train.jsonl"))
print("Original dev rows:", count_jsonl(PLD_DATA_DIR / "dev.jsonl"))

LISTENING_PROMPTS = [
    "Magandang araw. Ito ay pagsubok ng OmniVoice sa wikang Filipino.",
    "Sinusuri namin kung malinaw at natural ang boses sa bagong modelo.",
    "Ang layunin ng proyektong ito ay pagbutihin ang voice cloning para sa Filipino.",
    "Pakikinggan natin ang halimbawa upang matukoy ang kalidad ng pagsasalita.",
]

INFERENCE_JSONL = MANIFEST_DIR / "inference_listening_samples.jsonl"


def build_inference_manifest(src_path: Path, dst_path: Path) -> int:
    rows = []
    with src_path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))

    rows_by_wav_name = {Path(row["audio_path"]).name: row for row in rows}
    missing_wav_names = [
        wav_name
        for wav_name in INFERENCE_REFERENCE_WAV_NAMES
        if wav_name not in rows_by_wav_name
    ]
    if missing_wav_names:
        raise RuntimeError(
            "The configured inference reference WAV files were not found in "
            f"{src_path}: {missing_wav_names}"
        )

    selected_rows = [
        rows_by_wav_name[wav_name] for wav_name in INFERENCE_REFERENCE_WAV_NAMES
    ]

    with dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for idx, row in enumerate(selected_rows):
            sample = {
                "id": f"fil_{idx:03d}_{row['id']}",
                "text": LISTENING_PROMPTS[idx % len(LISTENING_PROMPTS)],
                "ref_audio": row["audio_path"],
                "ref_text": row["text"],
                "language_id": "fil",
                "language_name": "Filipino",
                "ref_duration": row.get("duration", row.get("audio_duration")),
                "source_id": row.get("original_id", row["id"]),
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Selected inference references:", INFERENCE_REFERENCE_WAV_NAMES)
    return len(selected_rows)


inference_count = build_inference_manifest(
    TEST_JSONL,
    INFERENCE_JSONL,
)
print("Inference rows:", inference_count, "->", INFERENCE_JSONL)
print(INFERENCE_JSONL.read_text(encoding="utf-8").splitlines()[:2])

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Localize token manifests

# %% [code] {"jupyter":{"outputs_hidden":false}}
def resolve_token_file(raw_path: str, split_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        return split_dir / path
    if path.exists():
        return path

    parts = list(path.parts)
    split_name = split_dir.name
    if split_name in parts:
        idx = len(parts) - 1 - parts[::-1].index(split_name)
        candidate = split_dir.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate

    candidate = split_dir / parts[-2] / parts[-1]
    if candidate.exists():
        return candidate
    return path


def localize_token_manifest(split: str) -> Path:
    split_dir = TOKEN_DATASET_DIR / "tokens_full" / split
    source_manifest = (
        split_dir / "data_portable.lst"
        if (split_dir / "data_portable.lst").exists()
        else split_dir / "data.lst"
    )
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing token manifest: {source_manifest}")

    localized_manifest = MANIFEST_DIR / f"tokens_full_{split}_localized.lst"
    with source_manifest.open("r", encoding="utf-8") as src, localized_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            shard_path_raw, jsonl_path_raw, count, duration = line.strip().split()
            shard_path = resolve_token_file(shard_path_raw, split_dir)
            jsonl_path = resolve_token_file(jsonl_path_raw, split_dir)

            if not shard_path.exists() or not jsonl_path.exists():
                raise FileNotFoundError(
                    "Could not resolve pretokenized shard paths:\n"
                    f"shard: {shard_path}\njsonl: {jsonl_path}\n"
                    f"source manifest: {source_manifest}"
                )

            dst.write(f"{shard_path} {jsonl_path} {count} {duration}\n")

    print(f"Using attached pretokenized {split} shards:", split_dir)
    print(f"Localized {split} token manifest:", localized_manifest)
    return localized_manifest


TRAIN_TOKEN_MANIFEST = localize_token_manifest("train")
DEV_TOKEN_MANIFEST = localize_token_manifest("dev")

print("Train token manifest first lines:")
print("\n".join(TRAIN_TOKEN_MANIFEST.read_text(encoding="utf-8").splitlines()[:3]))
print("Dev token manifest first lines:")
print("\n".join(DEV_TOKEN_MANIFEST.read_text(encoding="utf-8").splitlines()[:3]))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Write OmniVoice configs

# %% [code] {"jupyter":{"outputs_hidden":false}}
DATA_CONFIG = CONFIG_DIR / "data_config_full.json"
TRAIN_CONFIG = CONFIG_DIR / "train_config_full_sdpa.json"

data_config = {
    "train": [
        {
            "language_id": "fil",
            "manifest_path": [str(TRAIN_TOKEN_MANIFEST)],
            "repeat": 1,
        }
    ],
    "dev": [
        {
            "language_id": "fil",
            "manifest_path": [str(DEV_TOKEN_MANIFEST)],
            "repeat": 1,
        }
    ],
}
DATA_CONFIG.write_text(json.dumps(data_config, indent=2), encoding="utf-8")

train_config = {
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,
    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,
    "resume_from_checkpoint": None,
    "init_from_checkpoint": PRETRAINED_MODEL,
    "learning_rate": LEARNING_RATE,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": TRAINING_STEPS,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.01,
    "warmup_steps": 0,
    "batch_tokens": BATCH_TOKENS,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_workers": NUM_WORKERS,
    "mixed_precision": "fp16",
    "allow_tf32": True,
    "attn_implementation": "sdpa",
    "max_sample_tokens": MAX_SAMPLE_TOKENS,
    "min_sample_tokens": 50,
    "max_batch_size": MAX_BATCH_SIZE,
    "logging_steps": LOGGING_STEPS,
    "eval_steps": EVAL_STEPS,
}
TRAIN_CONFIG.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

print("Data config:", DATA_CONFIG)
print(DATA_CONFIG.read_text(encoding="utf-8"))
print("Train config:", TRAIN_CONFIG)
print(TRAIN_CONFIG.read_text(encoding="utf-8"))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. Run fine-tuning

# %% [code] {"jupyter":{"outputs_hidden":false}}
def checkpoint_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_model = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    has_tokenizer = (path / "tokenizer.json").exists() or (
        path / "tokenizer_config.json"
    ).exists()
    return has_config and has_model and has_tokenizer


def clean_incomplete_checkpoints(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir() and not checkpoint_is_complete(checkpoint_dir):
            print("Removing incomplete checkpoint:", checkpoint_dir)
            shutil.rmtree(checkpoint_dir)


if CLEAN_INCOMPLETE_CHECKPOINTS:
    clean_incomplete_checkpoints(OUTPUT_DIR)

print("Working disk before training:", disk_usage_line(Path("/kaggle/working")))
print("Temp disk before training:", disk_usage_line(Path("/kaggle/temp")))

train_env = wandb_env()
train_env["CUDA_VISIBLE_DEVICES"] = GPU_IDS
train_env["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{train_env.get('PYTHONPATH', '')}"

run(
    [
        "uv",
        "run",
        "accelerate",
        "launch",
        "--gpu_ids",
        GPU_IDS,
        "--num_processes",
        str(NUM_GPUS),
        "--num_machines",
        "1",
        "--mixed_precision",
        train_config["mixed_precision"],
        "--dynamo_backend",
        "no",
        "-m",
        "omnivoice.cli.train_best_eval",
        "--train_config",
        str(TRAIN_CONFIG),
        "--data_config",
        str(DATA_CONFIG),
        "--output_dir",
        str(OUTPUT_DIR),
    ],
    cwd=OMNIVOICE_DIR,
    env=train_env,
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. Locate best checkpoint

# %% [code] {"jupyter":{"outputs_hidden":false}}
BEST_CHECKPOINT = OUTPUT_DIR / BEST_CHECKPOINT_NAME
BEST_CHECKPOINT_METADATA = BEST_CHECKPOINT / "best_checkpoint.json"

numeric_checkpoints = sorted(
    [
        path
        for path in OUTPUT_DIR.glob("checkpoint-*")
        if path.is_dir() and path.name.split("-")[-1].isdigit()
    ],
    key=lambda path: int(path.name.split("-")[-1]),
)

best_checkpoint_metadata = {}
if BEST_CHECKPOINT_METADATA.exists():
    best_checkpoint_metadata = json.loads(
        BEST_CHECKPOINT_METADATA.read_text(encoding="utf-8")
    )

LATEST_CHECKPOINT = BEST_CHECKPOINT if BEST_CHECKPOINT.exists() else None
if LATEST_CHECKPOINT is None and numeric_checkpoints:
    LATEST_CHECKPOINT = numeric_checkpoints[-1]

print("Output dir:", OUTPUT_DIR)
print("Best checkpoint:", BEST_CHECKPOINT)
print("Best checkpoint metadata:", best_checkpoint_metadata)
print("Numeric checkpoints:", [str(path) for path in numeric_checkpoints])
print("Selected checkpoint for inference:", LATEST_CHECKPOINT)

if LATEST_CHECKPOINT is None:
    raise RuntimeError("Training completed but no checkpoint was found.")

if (OUTPUT_DIR / "train.log").exists():
    print("\nLast train.log lines:")
    print(
        "\n".join(
            (OUTPUT_DIR / "train.log")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()[-40:]
        )
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. Run base and fine-tuned inference

# %% [code] {"jupyter":{"outputs_hidden":false}}
def run_inference_set(model_path: str | Path, label: str, output_dir: Path) -> Path:
    if FORCE_RERUN_INFERENCE and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    env["PYTHONPATH"] = f"{OMNIVOICE_DIR}:{env.get('PYTHONPATH', '')}"

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
            str(INFERENCE_JSONL),
            "--res_dir",
            str(output_dir),
            "--num_step",
            str(INFERENCE_NUM_STEPS),
            "--nj_per_gpu",
            "1",
            "--batch_size",
            "1",
            "--lang_id",
            "fil",
            "--preprocess_prompt",
            "False",
            "--postprocess_output",
            "True",
            "--audio_chunk_threshold",
            "1000",
        ],
        cwd=OMNIVOICE_DIR,
        env=env,
    )
    elapsed = time.time() - start
    (output_dir / "_inference_seconds.txt").write_text(str(elapsed), encoding="utf-8")
    print(f"{label} inference seconds: {elapsed:.2f}")
    return output_dir


generated_dirs = {
    "base": run_inference_set(PRETRAINED_MODEL, "base", BASELINE_RESULTS_DIR),
    "finetuned": run_inference_set(
        LATEST_CHECKPOINT,
        "finetuned",
        FINETUNED_RESULTS_DIR,
    ),
}

for label, directory in generated_dirs.items():
    print(label, sorted(str(path) for path in directory.glob("*.wav")))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 11. Lightweight audio checks

# %% [code] {"jupyter":{"outputs_hidden":false}}
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_soundfile_available() -> None:
    try:
        import soundfile  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "soundfile"])


def audio_stats(wav_path: Path) -> dict:
    ensure_soundfile_available()
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(wav_path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")
    abs_audio = np.abs(audio)
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak = float(abs_audio.max()) if len(audio) else 0.0
    clipping_pct = float(np.mean(abs_audio >= 0.999) * 100.0) if len(audio) else 0.0
    near_silence_pct = float(np.mean(abs_audio < 1e-4) * 100.0) if len(audio) else 100.0
    return {
        "sample_rate": sample_rate,
        "duration_sec": duration,
        "rms": rms,
        "peak": peak,
        "clipping_pct": clipping_pct,
        "near_silence_pct": near_silence_pct,
    }


def collect_audio_checks(generated: dict[str, Path]) -> list[dict]:
    samples = read_jsonl(INFERENCE_JSONL)
    checks = []
    for label, directory in generated.items():
        for sample in samples:
            wav_path = directory / f"{sample['id']}.wav"
            if not wav_path.exists():
                checks.append(
                    {
                        "model_label": label,
                        "id": sample["id"],
                        "status": "missing",
                        "wav_path": str(wav_path),
                    }
                )
                continue

            checks.append(
                {
                    "model_label": label,
                    "id": sample["id"],
                    "status": "ok",
                    "text": sample["text"],
                    "ref_text": sample["ref_text"],
                    "ref_audio": sample["ref_audio"],
                    "ref_duration": sample.get("ref_duration"),
                    "wav_path": str(wav_path),
                    **audio_stats(wav_path),
                }
            )
    return checks


audio_checks = collect_audio_checks(generated_dirs)
AUDIO_CHECKS_JSONL = METRICS_DIR / "audio_checks.jsonl"
with AUDIO_CHECKS_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
    for row in audio_checks:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Audio checks:", AUDIO_CHECKS_JSONL)
for row in audio_checks:
    print(row)
print(
    "Playable generated audio files:",
    sum(1 for row in audio_checks if row.get("status") == "ok"),
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 12. Log configs, checkpoint, and audio to W&B

# %% [code] {"jupyter":{"outputs_hidden":false}}
def import_or_install_wandb():
    try:
        import wandb

        return wandb
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "wandb"])
        import wandb

        return wandb


def log_wandb_artifacts() -> None:
    if not USE_WANDB or WANDB_MODE == "disabled":
        print("Skipping W&B artifact logging.")
        return

    wandb = import_or_install_wandb()
    run_kwargs = {
        "project": WANDB_PROJECT,
        "id": WANDB_RUN_ID,
        "name": RUN_NAME,
        "resume": "allow",
        "tags": WANDB_TAGS,
        "config": {
            "run_name": RUN_NAME,
            "train_samples": train_count,
            "dev_samples": dev_count,
            "test_reference_samples": test_count,
            "inference_samples": inference_count,
            "pretrained_model": PRETRAINED_MODEL,
            "audio_tokenizer": AUDIO_TOKENIZER,
            "token_dataset_dir": str(TOKEN_DATASET_DIR),
            "save_final_checkpoint": SAVE_FINAL_CHECKPOINT,
            "best_checkpoint_name": BEST_CHECKPOINT_NAME,
            "best_checkpoint_metadata": best_checkpoint_metadata,
            **train_config,
        },
    }
    if WANDB_ENTITY:
        run_kwargs["entity"] = WANDB_ENTITY

    wandb_run = wandb.init(**run_kwargs)

    config_artifact = wandb.Artifact("omnivoice-filipino-run-config", type="config")
    config_paths = [
        DATA_CONFIG,
        TRAIN_CONFIG,
        TRAIN_JSONL,
        DEV_JSONL,
        TEST_JSONL,
        INFERENCE_JSONL,
        AUDIO_CHECKS_JSONL,
    ]
    if BEST_CHECKPOINT_METADATA.exists():
        config_paths.append(BEST_CHECKPOINT_METADATA)
    for path in config_paths:
        config_artifact.add_file(str(path), name=path.name)
    wandb_run.log_artifact(config_artifact)

    if LOG_CHECKPOINT_ARTIFACT_TO_WANDB and LATEST_CHECKPOINT is not None:
        checkpoint_artifact = wandb.Artifact(
            f"omnivoice-{RUN_NAME}-{LATEST_CHECKPOINT.name}",
            type="model",
            metadata={
                "run_name": RUN_NAME,
                "checkpoint": str(LATEST_CHECKPOINT),
                "checkpoint_policy": "best_eval_loss",
                "best_step": best_checkpoint_metadata.get("best_step"),
                "best_eval_loss": best_checkpoint_metadata.get("best_eval_loss"),
                "steps": train_config["steps"],
                "base_model": PRETRAINED_MODEL,
            },
        )
        checkpoint_artifact.add_dir(str(LATEST_CHECKPOINT))
        wandb_run.log_artifact(checkpoint_artifact)
        print("Logged checkpoint artifact:", LATEST_CHECKPOINT)

    if audio_checks:
        columns = list(audio_checks[0].keys())
        checks_table = wandb.Table(columns=columns)
        for row in audio_checks:
            checks_table.add_data(*[row.get(col) for col in columns])
        wandb_run.log({"audio_checks": checks_table})

    if LOG_AUDIO_TO_WANDB and audio_checks:
        audio_table = wandb.Table(
            columns=[
                "model_label",
                "id",
                "text",
                "ref_text",
                "ref_duration",
                "duration_sec",
                "rms",
                "peak",
                "audio",
                "wav_path",
            ]
        )
        for row in audio_checks:
            if row.get("status") != "ok":
                continue
            audio_table.add_data(
                row["model_label"],
                row["id"],
                row.get("text"),
                row.get("ref_text"),
                row.get("ref_duration"),
                row.get("duration_sec"),
                row.get("rms"),
                row.get("peak"),
                wandb.Audio(
                    row["wav_path"],
                    caption=f"{row['model_label']} | {row['id']} | {row.get('text', '')}",
                ),
                row["wav_path"],
            )
            wandb_run.log(
                {
                    f"audio/{row['model_label']}/{row['id']}": wandb.Audio(
                        row["wav_path"],
                        caption=f"{row['model_label']}: {row.get('text', '')}",
                    )
                }
            )
        wandb_run.log({"generated_audio_examples": audio_table})

    wandb_run.finish()
    print("Logged W&B run:", WANDB_RUN_ID)


log_wandb_artifacts()

print("Working disk after run:", disk_usage_line(Path("/kaggle/working")))
print("Temp disk after run:", disk_usage_line(Path("/kaggle/temp")))
