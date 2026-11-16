"""Run only Pipeline 1 with the user's system-context question, no added decision rubric."""

import argparse
import csv, json, hashlib, os, shutil, subprocess, time, threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS = SCRIPT_DIR.parents[1] / "data" / "pipelines" / "inputs"
QUESTION = (SCRIPT_DIR / "question.txt").read_text(encoding="utf-8").strip()


def output_directory():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for outputs, logs and progress",
    )
    args = parser.parse_args()
    codex = os.environ.get("CODEX_CLI_PATH") or shutil.which("codex")
    if not codex:
        parser.error("Install the Codex CLI or set CODEX_CLI_PATH")
    destination = args.output_dir.resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new output directory")
    return destination


MODEL = "gpt-5.6-sol"
STOP = threading.Event()
LOCK = threading.Lock()
STATUS = {
    "status": "running",
    "question": QUESTION,
    "total_sessions": 9,
    "completed": [],
    "errors": {},
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
STATES = {}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def csvread(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def checkpoint():
    write_json(ROOT / "progress.json", STATUS)


def validate(path, source, kind):
    rows = csvread(path)
    assert len(rows) == len(source)
    assert list(rows[0]) == ["product_id", "ManufacturerProductDesignation", "response"]
    for s, r in zip(source, rows):
        assert all(r[k] == s[k] for k in s)
        assert r["response"].strip()
        if kind == "binary":
            assert r["response"] in ["TRUE", "FALSE"]
        if kind == "ordinal":
            assert r["response"] in ["NONE", "LOW", "MODERATE", "HIGH"]
    return rows


def run(dataset, kind):
    key = dataset + "/" + kind
    if STOP.is_set():
        return
    b, source, metrics = STATES[dataset]
    workspace = b / "sessions" / kind
    for name in ["data", "results"]:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    data = workspace / "data/products.csv"
    csvwrite(data, source)
    output = workspace / "results" / f"experiment_1_{kind}.csv"
    formatting = {
        "binary": "For each product, answer exactly TRUE or FALSE.",
        "ordinal": "For each product, answer exactly NONE, LOW, MODERATE, or HIGH.",
        "qualitative": "For each product, answer in one or two concise sentences.",
    }[kind]
    task = f"""Assess each supplied product using this question:
{QUESTION}

Read all {len(source)} rows of {data}. The only supplied product-specific fields are product_id and ManufacturerProductDesignation. Do not obtain other product-specific data, browse, or read other experiment outputs.

Output format: {formatting}
Create exactly {output} as a UTF-8 CSV with header:
product_id,ManufacturerProductDesignation,response
Preserve both original fields and row order exactly. Write and validate exactly {len(source)} data rows. Modify only the requested output CSV. In the final message, state the row count and validation result."""
    (b / "logs" / f"experiment_1_{kind}_task_prompt.txt").write_text(task)
    codex = os.environ.get("CODEX_CLI_PATH") or shutil.which("codex")
    assert codex, "Codex CLI not found"
    log = b / "logs" / f"experiment_1_{kind}.jsonl"
    error = b / "logs" / f"experiment_1_{kind}_stderr.txt"
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    command = [
        codex,
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--model",
        MODEL,
        "-c",
        'model_reasoning_effort="low"',
        "-C",
        str(workspace),
        "-o",
        str(b / "logs" / f"experiment_1_{kind}_final.txt"),
        "-",
    ]
    print("START", key, flush=True)
    with log.open("w") as f, error.open("w") as e:
        result = subprocess.run(command, input=task, text=True, stdout=f, stderr=e)
    events = []
    for line in log.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    usage = next(
        (e["usage"] for e in reversed(events) if isinstance(e.get("usage"), dict)), {}
    )
    metric = {
        "run": "experiment_1_" + kind,
        "item_count": len(source),
        "started_at_utc": started,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - tick, 3),
        "exit_code": result.returncode,
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": "",
        "model_requested": MODEL,
        "effort_requested": "low",
    }
    write_json(b / "logs" / f"experiment_1_{kind}_metrics.json", metric)
    if result.returncode or not any(e.get("type") == "turn.completed" for e in events):
        details = (
            "\n".join(
                str(e.get("message") or e.get("error"))
                for e in events
                if e.get("type") in ["error", "turn.failed"]
            )
            or error.read_text()
        )
        if any(
            term in details.lower()
            for term in ["usage limit", "too many requests", "429"]
        ):
            STOP.set()
        raise RuntimeError(details)
    rows = validate(output, source, kind)
    shutil.copy2(output, b / "results" / output.name)
    write_json(
        b / "results" / f"experiment_1_{kind}_validation.json",
        {
            "status": "PASS",
            "rows": len(rows),
            "source_fields_and_order_preserved": True,
            "format_labels_valid": True,
        },
    )
    with LOCK:
        metrics[kind] = metric
        csvwrite(
            b / "results/experiment_1_metrics.csv",
            [metrics[k] for k in sorted(metrics)],
        )
        STATUS["completed"].append(key)
        checkpoint()
    print("DONE", key, len(STATUS["completed"]), "/9", flush=True)


if __name__ == "__main__":
    ROOT = output_directory()
    for dataset, n in [("wago", 300), ("siemens", 100), ("smc", 200)]:
        b = ROOT / dataset
        for name in ["data", "results", "logs", "sessions"]:
            (b / name).mkdir(parents=True, exist_ok=True)
        source = INPUTS / f"{dataset}_product_designations.csv"
        rows = csvread(source)
        assert len(rows) == n and len({r["product_id"] for r in rows}) == n
        assert list(rows[0]) == ["product_id", "ManufacturerProductDesignation"]
        shutil.copy2(source, b / "data" / source.name)
        (b / "question.txt").write_text(QUESTION + "\n")
        write_json(
            b / "provenance.json",
            {
                "source_csv": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "products": n,
                "input_columns": list(rows[0]),
                "question": QUESTION,
                "model_requested": MODEL,
                "effort_requested": "low",
                "fresh_run": True,
                "added_decision_rubric": False,
            },
        )
        STATES[dataset] = (b, rows, {})
    (ROOT / "question.txt").write_text(QUESTION + "\n")
    checkpoint()
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {
            pool.submit(run, d, k): d + "/" + k
            for k in ["binary", "ordinal", "qualitative"]
            for d in STATES
        }
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as exc:
                with LOCK:
                    STATUS["errors"][tasks[future]] = str(exc)
                    checkpoint()
                print("ERROR", tasks[future], str(exc), flush=True)
    summaries = []
    for dataset, (b, source, metrics) in STATES.items():
        available = {
            kind: csvread(b / "results" / f"experiment_1_{kind}.csv")
            for kind in ["binary", "ordinal", "qualitative"]
            if (b / "results" / f"experiment_1_{kind}.csv").exists()
        }
        summary = {
            "dataset": dataset,
            "products": len(source),
            "completed_formats": list(available),
            "distributions": {
                kind: dict(Counter(r["response"] for r in rows))
                for kind, rows in available.items()
                if kind != "qualitative"
            },
        }
        summaries.append(summary)
        write_json(b / "summary.json", summary)
        if len(available) == 3:
            combined = []
            for i, r in enumerate(source):
                combined.append(
                    {
                        **r,
                        **{
                            kind: available[kind][i]["response"]
                            for kind in ["binary", "ordinal", "qualitative"]
                        },
                    }
                )
            csvwrite(b / "results/experiment_1_combined.csv", combined)
    STATUS.update(
        status="complete" if len(STATUS["completed"]) == 9 else "incomplete",
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    checkpoint()
    write_json(ROOT / "summary.json", summaries)
    print(STATUS["status"].upper(), json.dumps(summaries), flush=True)
    if STATUS["status"] != "complete":
        raise SystemExit(1)
