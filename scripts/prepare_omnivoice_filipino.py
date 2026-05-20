#!/usr/bin/env python3
"""Prepare the PLD Filipino subset as OmniVoice fine-tuning JSONL manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRANSCRIPT_RE = re.compile(
    r'^(?P<wav>\S+\.wav)\s+"(?P<source>[^"]*)"\s+"(?P<text>.*)"\s*$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleaned OmniVoice JSONL manifests from PLD Filipino WAV/LOG files."
    )
    parser.add_argument(
        "--data-root",
        default="data_artifacts/original_pld",
        help="Original extracted PLD Filipino data root.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_artifacts/filipino_omnivoice_manifests",
        help="Directory for generated manifests and reports.",
    )
    parser.add_argument("--language-id", default="fil", help="OmniVoice language ID.")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--dev-ratio", "--val-ratio", dest="dev_ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--package-dir",
        default=None,
        help=(
            "Optional cleaned dataset directory to materialize with selected WAVs "
            "and portable JSONL manifests."
        ),
    )
    parser.add_argument(
        "--package-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize WAVs in --package-dir.",
    )
    return parser.parse_args()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / rate


def is_spontaneous_source(source: str) -> bool:
    lowered = source.lower()
    return "spontaneous" in lowered or "_spo_" in lowered


def has_parentheses_or_digits(text: str) -> bool:
    return "(" in text or ")" in text or any(char.isdigit() for char in text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(row["duration"] for row in rows)
    speakers = {row["speaker_id"] for row in rows}
    sources = Counter(row["source"] for row in rows)
    return {
        "samples": len(rows),
        "hours": round(duration / 3600, 3),
        "speakers": len(speakers),
        "top_sources": sources.most_common(20),
    }


def materialize_package(
    package_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    all_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    wav_dir = package_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    linked_or_copied = 0
    reused_existing = 0

    for row in all_rows:
        source_path = Path(row["audio_path"])
        target_path = wav_dir / row["speaker_id"] / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            if target_path.stat().st_size != source_path.stat().st_size:
                raise RuntimeError(f"Existing packaged file has wrong size: {target_path}")
            reused_existing += 1
        else:
            if mode == "hardlink":
                os.link(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            linked_or_copied += 1

        seen_ids.add(row["id"])

    def packaged_manifest_row(row: dict[str, Any]) -> dict[str, str]:
        source_path = Path(row["audio_path"])
        target_path = wav_dir / row["speaker_id"] / source_path.name
        return {
            "id": row["id"],
            "audio_path": target_path.relative_to(package_dir).as_posix(),
            "text": row["text"],
            "language_id": row["language_id"],
        }

    packaged_splits = {
        split: [packaged_manifest_row(row) for row in rows]
        for split, rows in splits.items()
    }

    for split_name, rows in packaged_splits.items():
        write_jsonl(package_dir / f"{split_name}.jsonl", rows)

    package_info = {
        "package_dir": str(package_dir.resolve()),
        "audio_path_style": "relative_to_package_dir",
        "wav_root": "wavs",
        "materialization_mode": mode,
        "files_materialized": linked_or_copied,
        "files_reused_existing": reused_existing,
        "unique_audio_files": len(seen_ids),
        "manifests": ["train.jsonl", "dev.jsonl", "test.jsonl"],
        "manifest_fields": ["id", "audio_path", "text", "language_id"],
    }

    return package_info


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = {path.name: path for path in data_root.rglob("*.wav")}
    log_paths = sorted(data_root.rglob("*.log"))

    parsed_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_existing_wavs: set[str] = set()

    for log_path in log_paths:
        speaker_id = log_path.parent.name
        for line_number, raw_line in enumerate(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            line = raw_line.strip()
            match = TRANSCRIPT_RE.match(line)
            if not match:
                continue

            wav_name = match.group("wav")
            source = match.group("source").strip()
            text = " ".join(match.group("text").strip().split())
            wav_path = wav_paths.get(wav_name)

            base_info = {
                "wav": wav_name,
                "speaker_id": speaker_id,
                "source": source,
                "text": text,
                "log_path": str(log_path),
                "log_line": line_number,
            }

            def reject(reason: str) -> None:
                rejection_counts[reason] += 1
                if len(rejection_examples[reason]) < 10:
                    rejection_examples[reason].append(base_info)

            if wav_path is None:
                reject("log_row_without_wav")
                continue

            matched_existing_wavs.add(wav_name)

            if wav_path.stat().st_size == 0:
                reject("zero_byte_wav")
                continue

            if not text:
                reject("empty_text")
                continue

            if is_spontaneous_source(source):
                reject("spontaneous_source")
                continue

            if has_parentheses_or_digits(text):
                reject("parentheses_or_digits")
                continue

            try:
                duration = wav_duration_seconds(wav_path)
            except Exception as exc:
                base_info["error"] = str(exc)
                reject("unreadable_wav")
                continue

            if duration < args.min_duration:
                base_info["duration"] = round(duration, 3)
                reject("too_short")
                continue

            if duration > args.max_duration:
                base_info["duration"] = round(duration, 3)
                reject("too_long")
                continue

            parsed_rows.append(
                {
                    "id": wav_path.stem,
                    "audio_path": str(wav_path.resolve()),
                    "text": text,
                    "language_id": args.language_id,
                    "speaker_id": speaker_id,
                    "source": source,
                    "duration": round(duration, 6),
                }
            )

    wavs_without_log = sorted(set(wav_paths) - matched_existing_wavs)
    for wav_name in wavs_without_log[:10]:
        rejection_examples["wav_without_log"].append({"wav": wav_name})
    rejection_counts["wav_without_log"] = len(wavs_without_log)

    rows_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        rows_by_speaker[row["speaker_id"]].append(row)

    speaker_ids = sorted(rows_by_speaker)
    rng = random.Random(args.seed)
    rng.shuffle(speaker_ids)

    dev_count = max(1, round(len(speaker_ids) * args.dev_ratio))
    test_count = max(1, round(len(speaker_ids) * args.test_ratio))
    dev_speakers = set(speaker_ids[:dev_count])
    test_speakers = set(speaker_ids[dev_count : dev_count + test_count])
    train_speakers = set(speaker_ids[dev_count + test_count :])

    splits = {
        "train": sorted(
            (row for row in parsed_rows if row["speaker_id"] in train_speakers),
            key=lambda row: row["id"],
        ),
        "dev": sorted(
            (row for row in parsed_rows if row["speaker_id"] in dev_speakers),
            key=lambda row: row["id"],
        ),
        "test": sorted(
            (row for row in parsed_rows if row["speaker_id"] in test_speakers),
            key=lambda row: row["id"],
        ),
    }

    all_rows = sorted(parsed_rows, key=lambda row: row["id"])
    write_jsonl(out_dir / "all_clean.jsonl", all_rows)
    for split_name, split_rows in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", split_rows)

    summary = {
        "input": {
            "data_root": str(data_root.resolve()),
            "wav_files": len(wav_paths),
            "log_files": len(log_paths),
        },
        "filters": {
            "excluded_sources": [
                "sources containing 'spontaneous'",
                "sources containing '_spo_'",
            ],
            "excluded_text": "rows with parentheses or ASCII/Unicode digits",
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "language_id": args.language_id,
        },
        "cleaned": summarize_rows(all_rows),
        "splits": {name: summarize_rows(rows) for name, rows in splits.items()},
        "split_speakers": {
            "train": sorted(train_speakers),
            "dev": sorted(dev_speakers),
            "test": sorted(test_speakers),
        },
        "rejections": dict(rejection_counts),
        "rejection_examples": rejection_examples,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# Filipino OmniVoice Manifests

Generated from `data_artifacts/original_pld/` by `scripts/prepare_omnivoice_filipino.py`.

These JSONL files are ready for OmniVoice tokenization. Each row contains
`id`, `audio_path`, `text`, and `language_id`, plus speaker/source/duration
metadata for auditing.

## Files

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `all_clean.jsonl`
- `summary.json`

## Filters

- Excluded spontaneous prompt sources: any source containing `spontaneous` or `_spo_`.
- Excluded text containing parentheses or digits.
- Excluded unreadable/zero-byte WAVs.
- Kept durations from `{args.min_duration}` to `{args.max_duration}` seconds.
- Used OmniVoice language ID `{args.language_id}`.

## Tokenization

From the `.OmniVoice/` directory, point `examples/run_finetune.sh` at:

```bash
TRAIN_JSONL="../../data_artifacts/filipino_omnivoice_manifests/train.jsonl"
DEV_JSONL="../../data_artifacts/filipino_omnivoice_manifests/dev.jsonl"
```

The script will create token shards under its configured `TOKEN_DIR`, then train
from `examples/config/data_config_finetune.json`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    if args.package_dir:
        package_info = materialize_package(
            package_dir=Path(args.package_dir),
            splits=splits,
            all_rows=all_rows,
            summary=summary,
            mode=args.package_mode,
        )
        summary["package"] = package_info
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
