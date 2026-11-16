#!/usr/bin/env python3
"""Pipeline 3 with the current production-system prompt, unchanged.
Python 3.10+; standard library only; authenticated Codex CLI required.
Three separate specialists followed by one final synthesis session.
"""

from pathlib import Path
import csv

QUESTION = Path(__file__).with_name("question.txt").read_text(encoding="utf-8").strip()
BASE_PROMPTS = {"SYSTEM_CONTEXT": QUESTION}

ROLES = {
    1: (
        "evidence",
        "evidence analyst: identify the product role and measurement-relevant evidence for the shared question",
    ),
    2: (
        "skeptic",
        "skeptical reviewer: examine ambiguity and unsupported assumptions in a monitoring recommendation",
    ),
    3: (
        "application",
        "maintenance-application analyst: assess monitoring of this component within the production system under the shared question",
    ),
}


def setup(b, question):
    (b / "experiment_3_prompts.txt").write_text(QUESTION + "\n", encoding="utf-8")


def prompt(b, question, name, source):
    output = b / "results" / f"{name}.csv"
    if "specialist_" in name:
        number = int(name.split("_")[3])
        role = ROLES[number][1]
        instruction = (
            "You are an independent "
            + role
            + ". Answer the shared question for every product. "
            "Do not read other specialist or prior pipeline outputs. Provide one concise public "
            "evidence summary, at most 18 whitespace-separated words, plus binary and ordinal recommendations."
        )
        schema = "product_id,ManufacturerProductDesignation,evidence_summary,binary_recommendation,ordinal_recommendation"
    else:
        instruction = (
            "You are the final hierarchical judge. Read the three specialist reports in "
            + str(b / "results" / "experiment_3_specialist_panel.csv")
            + ". Synthesize their reports to answer the shared question for each product. "
            "Resolve disagreements; do not merely count votes. Provide binary/ordinal labels "
            "and a concise public rationale of at most 24 whitespace-separated words. "
            "Do not read other pipelines or previous experiments."
        )
        schema = "product_id,ManufacturerProductDesignation,final_binary,final_ordinal,final_rationale"
    return f"""Shared question (unchanged):
{QUESTION}

{instruction}
Read all {len(source)} rows from {b / "data" / "input_product_designations.csv"}. Product-specific inputs are only product_id and ManufacturerProductDesignation. Do not obtain other product-specific data or browse. For the final hierarchical judge, the supplied specialist panel is also available.
Binary labels are TRUE or FALSE. Ordinal labels are NONE, LOW, MODERATE, or HIGH.
Create exactly this UTF-8 CSV: {output}
Header: {schema}
Preserve both original input fields and row order exactly. Write and validate exactly {len(source)} data rows. Modify only the requested output CSV. State row count and validation in the final message."""


def panel(b, source):
    reports = []
    for number, (role, _) in ROLES.items():
        with (b / "results" / f"experiment_3_specialist_{number}_{role}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reports.append(list(csv.DictReader(handle)))
    combined = []
    for i, original in enumerate(source):
        row = dict(original)
        for j, (role, _) in enumerate(ROLES.values()):
            item = reports[j][i]
            row.update(
                {
                    role + "_summary": item["evidence_summary"],
                    role + "_binary": item["binary_recommendation"],
                    role + "_ordinal": item["ordinal_recommendation"],
                }
            )
        combined.append(row)
    with (b / "results" / "experiment_3_specialist_panel.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)


import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

VERSION = "pipeline3-system-context-v1"
BINARY = {"TRUE", "FALSE"}
ORDINAL = {"NONE", "LOW", "MODERATE", "HIGH"}


class PipelineError(RuntimeError):
    pass


class QuotaError(PipelineError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def write_csv(path, rows, header=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_rows(path, limit):
    header, rows = read_csv(path)
    if header != ["product_id", "ManufacturerProductDesignation"]:
        raise PipelineError(
            f"{path}: expected exactly product_id,ManufacturerProductDesignation"
        )
    if not rows or any(
        set(row) != set(header)
        or any(not isinstance(row.get(k), str) or not row[k].strip() for k in header)
        for row in rows
    ):
        raise PipelineError(f"{path}: empty or malformed input rows")
    if len({row["product_id"] for row in rows}) != len(rows):
        raise PipelineError(f"{path}: duplicate product IDs")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise PipelineError(f"--limit must be between 1 and {len(rows)} for {path}")
        rows = rows[:limit]
    return rows


def validate(path, source, final=False, question="SYSTEM_CONTEXT"):
    header, rows = read_csv(path)
    expected = ["product_id", "ManufacturerProductDesignation"] + (
        ["final_binary", "final_ordinal", "final_rationale"]
        if final
        else ["evidence_summary", "binary_recommendation", "ordinal_recommendation"]
    )
    if header != expected or len(rows) != len(source):
        raise PipelineError(f"{path}: incorrect header or row count")
    for original, row in zip(source, rows):
        if set(row) != set(expected) or any(
            row[key] != original[key] for key in original
        ):
            raise PipelineError(f"{path}: source fields/order not preserved")
        binary = row["final_binary" if final else "binary_recommendation"]
        ordinal = row["final_ordinal" if final else "ordinal_recommendation"]
        text = row["final_rationale" if final else "evidence_summary"]
        if binary not in BINARY or ordinal not in ORDINAL:
            raise PipelineError(f"{path}: invalid labels for {row['product_id']}")
        if not isinstance(text, str) or not 0 < len(text.split()) <= (
            24 if final else 18
        ):
            raise PipelineError(
                f"{path}: justification length invalid for {row['product_id']}"
            )
    return rows


def events_from(path):
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except ValueError:
            pass
    return events


def run_stage(base, question, name, source, fingerprint, args, codex):
    published = base / "results" / f"{name}.csv"
    checkpoint = base / "logs" / f"{name}_checkpoint.json"
    final = name == "experiment_3_final_judgment"
    if checkpoint.exists() and args.resume:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise PipelineError(
                f"{base}: resume configuration changed; use a new output folder"
            )
        if saved.get("status") == "complete" and published.exists():
            validate(published, source, final, question)
            if digest(published.read_bytes()) != saved["output_sha256"]:
                raise PipelineError(f"{published}: changed after completion")
            print(f"RETAIN {base.parent.name}/{question}/{name}", flush=True)
            return saved["metrics"]
    workspace = base / "sessions" / name
    for folder in ["data", "results"]:
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    write_csv(workspace / "data" / "input_product_designations.csv", source)
    (workspace / "prompts.txt").write_text(BASE_PROMPTS[question], encoding="utf-8")
    setup(workspace, question)
    if final:
        shutil.copy2(
            base / "results" / "experiment_3_specialist_panel.csv",
            workspace / "results" / "experiment_3_specialist_panel.csv",
        )
    output = workspace / "results" / f"{name}.csv"
    task = prompt(workspace, question, name, source)
    task_path = base / "logs" / f"{name}_task_prompt.txt"
    task_path.write_text(task, encoding="utf-8")
    existing = list((base / "logs").glob(f"{name}_attempt_*_metrics.json"))
    start_number = (
        max(
            [int(p.name.split("_attempt_")[1].split("_")[0]) for p in existing],
            default=0,
        )
        + 1
    )
    for offset in range(args.retries + 1):
        number = start_number + offset
        stem = base / "logs" / f"{name}_attempt_{number}"
        log_path = stem.with_suffix(".jsonl")
        error_path = stem.with_name(stem.name + "_stderr.txt")
        if output.exists():
            output.rename(stem.with_name(stem.name + "_previous_output.csv"))
        command = [
            codex,
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--model",
            args.model,
            "-c",
            f'model_reasoning_effort="{args.effort}"',
            "-C",
            str(workspace),
            "-o",
            str(stem.with_name(stem.name + "_final.txt")),
            "-",
        ]
        started, timer = now(), time.perf_counter()
        print(
            f"START {base.parent.name}/{question}/{name}, attempt {number}", flush=True
        )
        failure, returncode = "", None
        try:
            with (
                log_path.open("w", encoding="utf-8") as log,
                error_path.open("w", encoding="utf-8") as error,
            ):
                result = subprocess.run(
                    command,
                    input=task,
                    text=True,
                    encoding="utf-8",
                    cwd=workspace,
                    stdout=log,
                    stderr=error,
                    timeout=args.timeout,
                )
                returncode = result.returncode
        except subprocess.TimeoutExpired:
            failure = f"Timed out after {args.timeout} seconds"
        events = events_from(log_path)
        messages = "\n".join(
            str(e.get("message", "")) + " " + str(e.get("error", ""))
            for e in events
            if e.get("type") in {"error", "turn.failed"}
        )
        error_text = error_path.read_text(encoding="utf-8", errors="replace")
        quota = any(
            term in (messages + error_text).lower()
            for term in [
                "usage limit",
                "too many requests",
                "429",
                "insufficient_quota",
                "rate limit",
            ]
        )
        complete = any(e.get("type") == "turn.completed" for e in events)
        usage = next(
            (e["usage"] for e in reversed(events) if isinstance(e.get("usage"), dict)),
            {},
        )
        valid = False
        if returncode == 0 and complete:
            try:
                validate(output, source, final, question)
                valid = True
            except (PipelineError, OSError) as exc:
                failure = str(exc)
        elif not failure:
            failure = messages or f"CLI exited {returncode}; see {error_path}"
        metrics = {
            "run": name,
            "item_count": len(source),
            "started_at_utc": started,
            "ended_at_utc": now(),
            "elapsed_seconds": round(time.perf_counter() - timer, 3),
            "exit_code": returncode,
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": "",
            "cost_note": "CLI usage is not an invoice; elapsed time includes tools/service waits.",
        }
        write_json(
            stem.with_name(stem.name + "_metrics.json"),
            {
                **metrics,
                "validation_passed": valid,
                "failure": failure,
                "quota_error": quota,
            },
        )
        if valid:
            shutil.copy2(output, published)
            write_json(
                checkpoint,
                {
                    "fingerprint": fingerprint,
                    "status": "complete",
                    "output_sha256": digest(published.read_bytes()),
                    "metrics": metrics,
                },
            )
            print(f"DONE {base.parent.name}/{question}/{name}", flush=True)
            return metrics
        write_json(
            checkpoint,
            {"fingerprint": fingerprint, "status": "incomplete", "last_error": failure},
        )
        if quota:
            raise QuotaError(
                f"Account/rate limit stopped {name}. Completed work is saved. Restore access and rerun with --resume. Details: {messages or error_text}"
            )
        if offset < args.retries:
            time.sleep(args.retry_delay)
    raise PipelineError(f"{name} failed validation/execution: {failure}")


def run_combo(dataset, question, source, args, codex):
    base = args.output_dir / dataset / question
    for directory in ["data", "results", "logs", "sessions"]:
        (base / directory).mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "dataset": dataset,
        "question": question,
        "model": args.model,
        "effort": args.effort,
        "rows": source,
        "base_prompt": BASE_PROMPTS[question],
        "script_sha256": digest(Path(__file__).read_bytes()),
    }
    fingerprint = digest(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    manifest = base / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not args.resume:
            raise PipelineError(f"{base} exists. Use --resume or a new --output-dir.")
        if previous["fingerprint"] != fingerprint:
            raise PipelineError(
                f"{base}: input/model/prompt selection changed; use a new output directory"
            )
    write_json(
        manifest,
        {
            "fingerprint": fingerprint,
            "version": VERSION,
            "dataset": dataset,
            "question": question,
            "model_requested": args.model,
            "effort_requested": args.effort,
            "item_count": len(source),
            "prompt_sha256": digest(BASE_PROMPTS[question].encode()),
            "status": "running",
        },
    )
    write_csv(base / "data" / f"{dataset}_product_designations.csv", source)
    (base / "prompts.txt").write_text(BASE_PROMPTS[question], encoding="utf-8")
    setup(base, question)
    metrics = []
    if args.workers == 1:
        for n, (role, _) in ROLES.items():
            metrics.append(
                run_stage(
                    base,
                    question,
                    f"experiment_3_specialist_{n}_{role}",
                    source,
                    fingerprint,
                    args,
                    codex,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_stage,
                    base,
                    question,
                    f"experiment_3_specialist_{n}_{role}",
                    source,
                    fingerprint,
                    args,
                    codex,
                )
                for n, (role, _) in ROLES.items()
            ]
            try:
                for future in as_completed(futures):
                    metrics.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    panel(base, source)
    metrics.append(
        run_stage(
            base,
            question,
            "experiment_3_final_judgment",
            source,
            fingerprint,
            args,
            codex,
        )
    )
    write_csv(
        base / "results" / "experiment_3_metrics.csv",
        sorted(metrics, key=lambda x: x["run"]),
    )
    rows = validate(
        base / "results" / "experiment_3_final_judgment.csv", source, True, question
    )
    summary = {
        "dataset": dataset,
        "question": question,
        "products": len(rows),
        "binary": dict(Counter(r["final_binary"] for r in rows)),
        "ordinal": dict(Counter(r["final_ordinal"] for r in rows)),
        "validated": True,
    }
    write_json(base / "summary.json", summary)
    state = json.loads(manifest.read_text(encoding="utf-8"))
    state.update(status="complete", completed_at_utc=now())
    write_json(manifest, state)
    lines = [
        f"# {dataset.upper()} — Pipeline 3 — {question}",
        "",
        f"Products validated: {len(rows)}",
        "",
        "Three separate specialists were followed by a final synthesizing judge; this is not majority voting.",
        "",
        "Binary: " + json.dumps(summary["binary"]),
        "Ordinal: " + json.dumps(summary["ordinal"]),
        "",
        "These are model judgments, not measured predictive accuracy. See prompts.txt for the exact unchanged question. No additional ordinal rubric is imposed.",
        "",
        "Successful-session elapsed sum: "
        + f"{sum(float(m['elapsed_seconds']) for m in metrics):.1f} seconds.",
        "Recorded time includes tools, waiting and reconnects; concurrent sums are not wall-clock time.",
        "Dollar costs are unavailable. Logs/checkpoints retain all attempts.",
    ]
    (base / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "pipelines" / "inputs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pipeline3_system_context_outputs")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wago", "siemens", "smc"],
        default=["wago", "siemens", "smc"],
    )
    # This script deliberately runs only the current question.
    parser.add_argument(
        "--model",
        default="gpt-5.6-sol",
        help="Matches the prior experiment model; must be available to your account.",
    )
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--codex", default=os.environ.get("CODEX_CLI_PATH", "codex"))
    parser.add_argument(
        "--workers",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Concurrent specialists; final judge always waits.",
    )
    parser.add_argument(
        "--limit", type=int, help="First N rows for a separately saved smoke test."
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Seconds per model session."
    )
    parser.add_argument(
        "--retries",
        type=int,
        choices=range(4),
        default=0,
        help="Extra retries for non-quota errors only.",
    )
    parser.add_argument("--retry-delay", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print plan; no model calls/output files.",
    )
    args = parser.parse_args(argv)
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if args.timeout < 1 or args.retry_delay < 0:
        parser.error("timeout must be positive and retry-delay nonnegative")
    args.datasets = list(dict.fromkeys(args.datasets))
    args.questions = ["SYSTEM_CONTEXT"]
    sources = {
        d: input_rows(args.data_dir / f"{d}_product_designations.csv", args.limit)
        for d in args.datasets
    }
    for dataset in args.datasets:
        for question in args.questions:
            print(
                f"PLAN {dataset}/{question}: {len(sources[dataset])} products, 3 specialists + 1 final judge"
            )
    if args.dry_run:
        print(
            f"Dry run complete. Would use {4 * len(args.datasets) * len(args.questions)} model sessions. No calls made."
        )
        return 0
    codex = shutil.which(args.codex) or (
        str(Path(args.codex).resolve()) if Path(args.codex).is_file() else None
    )
    if not codex:
        raise PipelineError(
            "Codex CLI not found. Install/sign in using the official instructions in README.md, or pass --codex PATH."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / ".pipeline3_running.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise PipelineError(
            f"{lock} exists: another run may be active. After confirming it stopped, remove a stale lock and use --resume."
        )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        summaries = []
        for dataset in args.datasets:
            for question in args.questions:
                summaries.append(
                    run_combo(dataset, question, sources[dataset], args, codex)
                )
        write_json(args.output_dir / "pipeline3_summary.json", summaries)
        print(f"COMPLETE: {args.output_dir}")
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed checkpoints are retained; run again with --resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except (PipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
