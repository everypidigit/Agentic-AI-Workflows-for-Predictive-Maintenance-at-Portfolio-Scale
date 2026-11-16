"""Pipelines 2/3 using the unchanged short system-context question."""

import argparse
import csv, json, hashlib, shutil, os, subprocess, time, threading
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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
LOCK = threading.Lock()
STOP = threading.Event()
STATUS = {
    "status": "running",
    "total_sessions": 27,
    "completed": [],
    "errors": {},
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
S = {}


def read(p):
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csvwrite(p, rows):
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def js(p, v):
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(v, indent=2, ensure_ascii=False))
    temp.replace(p)


def check():
    js(ROOT / "progress.json", STATUS)


def validate(path, source, kind):
    rows = read(path)
    assert len(rows) == len(source)
    header = ["product_id", "ManufacturerProductDesignation"] + (
        {
            "judge": ["binary_vote", "ordinal_vote"],
            "specialist": [
                "evidence_summary",
                "binary_recommendation",
                "ordinal_recommendation",
            ],
            "final": ["final_binary", "final_ordinal", "final_rationale"],
        }[kind]
    )
    assert list(rows[0]) == header
    for s, r in zip(source, rows):
        assert all(r[k] == s[k] for k in s)
        b, o = {
            "judge": ("binary_vote", "ordinal_vote"),
            "specialist": ("binary_recommendation", "ordinal_recommendation"),
            "final": ("final_binary", "final_ordinal"),
        }[kind]
        assert r[b] in ["TRUE", "FALSE"] and r[o] in ["NONE", "LOW", "MODERATE", "HIGH"]
        if kind != "judge":
            text = r["evidence_summary" if kind == "specialist" else "final_rationale"]
            assert text.strip() and len(text.split()) <= (
                18 if kind == "specialist" else 24
            )
    return rows


def execute(job):
    pipeline, dataset, name, kind, role = job
    if STOP.is_set():
        return
    b, source, metrics = S[(pipeline, dataset)]
    workspace = b / "sessions" / name
    for folder in ["data", "results"]:
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    csvwrite(workspace / "data/products.csv", source)
    output = workspace / "results" / f"{name}.csv"
    if kind == "judge":
        instruction = "You are an independent judge. Answer the shared question for every product, returning both binary and ordinal labels. Do not read other judges or any prior pipeline outputs."
        schema = "product_id,ManufacturerProductDesignation,binary_vote,ordinal_vote"
    elif kind == "specialist":
        instruction = (
            "You are an independent "
            + role
            + ". Answer the shared question for every product. Do not read other specialist or prior pipeline outputs. Provide one concise public evidence summary, at most 18 whitespace-separated words, plus binary and ordinal recommendations."
        )
        schema = "product_id,ManufacturerProductDesignation,evidence_summary,binary_recommendation,ordinal_recommendation"
    else:
        shutil.copy2(
            b / "results/experiment_3_specialist_panel.csv",
            workspace / "data/specialist_panel.csv",
        )
        instruction = (
            "You are the final hierarchical judge. Read the three specialist reports in "
            + str(workspace / "data/specialist_panel.csv")
            + ". Synthesize their reports to answer the shared question for each product. Resolve disagreements; do not merely count votes. Provide binary/ordinal labels and a concise public rationale of at most 24 whitespace-separated words. Do not read other pipelines or previous experiments."
        )
        schema = "product_id,ManufacturerProductDesignation,final_binary,final_ordinal,final_rationale"
    task = f"""Shared question (unchanged):
{QUESTION}

{instruction}
Read all {len(source)} rows from {workspace / "data/products.csv"}. Product-specific inputs are only product_id and ManufacturerProductDesignation. Do not obtain other product-specific data or browse. For the final hierarchical judge, the supplied specialist panel is also available.
Binary labels are TRUE or FALSE. Ordinal labels are NONE, LOW, MODERATE, or HIGH.
Create exactly this UTF-8 CSV: {output}
Header: {schema}
Preserve both original input fields and row order exactly. Write and validate exactly {len(source)} data rows. Modify only the requested output CSV. State row count and validation in the final message."""
    (b / "logs" / f"{name}_task_prompt.txt").write_text(task)
    codex = os.environ.get("CODEX_CLI_PATH") or shutil.which("codex")
    assert codex
    log = b / "logs" / f"{name}.jsonl"
    err = b / "logs" / f"{name}_stderr.txt"
    start = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    cmd = [
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
        str(b / "logs" / f"{name}_final.txt"),
        "-",
    ]
    key = f"P{pipeline}/{dataset}/{name}"
    print("START", key, flush=True)
    with log.open("w") as f, err.open("w") as e:
        result = subprocess.run(cmd, input=task, text=True, stdout=f, stderr=e)
    events = []
    for line in log.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    usage = next(
        (e["usage"] for e in reversed(events) if isinstance(e.get("usage"), dict)), {}
    )
    m = {
        "run": name,
        "item_count": len(source),
        "started_at_utc": start,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - tick, 3),
        "exit_code": result.returncode,
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": "",
    }
    js(b / "logs" / f"{name}_metrics.json", m)
    if result.returncode or not any(e.get("type") == "turn.completed" for e in events):
        details = (
            "\n".join(
                str(e.get("message") or e.get("error"))
                for e in events
                if e.get("type") in ["error", "turn.failed"]
            )
            or err.read_text()
        )
        if any(
            x in details.lower() for x in ["usage limit", "too many requests", "429"]
        ):
            STOP.set()
        raise RuntimeError(details)
    validate(output, source, kind)
    shutil.copy2(output, b / "results" / output.name)
    with LOCK:
        metrics[name] = m
        csvwrite(
            b / "results" / f"experiment_{pipeline}_metrics.csv",
            [metrics[k] for k in sorted(metrics)],
        )
        STATUS["completed"].append(key)
        check()
    print("DONE", key, len(STATUS["completed"]), "/27", flush=True)


def batch(jobs):
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(execute, j): j for j in jobs}
        for f in as_completed(tasks):
            try:
                f.result()
            except Exception as exc:
                with LOCK:
                    STATUS["errors"][str(tasks[f])] = str(exc)
                    check()
                print("ERROR", tasks[f], str(exc), flush=True)


def aggregate_p2(dataset):
    b, source, _ = S[(2, dataset)]
    judges = [
        validate(b / "results" / f"experiment_2_judge_{j}.csv", source, "judge")
        for j in range(1, 6)
    ]
    out = []
    for i, s in enumerate(source):
        row = dict(s)
        for kind in ["binary", "ordinal"]:
            votes = [j[i][kind + "_vote"] for j in judges]
            row.update({f"{kind}_vote_judge_{j}": votes[j - 1] for j in range(1, 6)})
            label, count = Counter(votes).most_common(1)[0]
            row[kind + "_majority_vote"] = label if count >= 3 else "NO_MAJORITY"
            row[kind + "_majority_vote_count"] = count
        out.append(row)
    csvwrite(b / "results/experiment_2_majority_vote.csv", out)


def build_panel(dataset):
    b, source, _ = S[(3, dataset)]
    roles = ["evidence", "skeptic", "application"]
    specialists = [
        validate(
            b / "results" / f"experiment_3_specialist_{j}_{r}.csv", source, "specialist"
        )
        for j, r in enumerate(roles, 1)
    ]
    out = []
    for i, s in enumerate(source):
        row = dict(s)
        for j, role in enumerate(roles):
            r = specialists[j][i]
            row.update(
                {
                    role + "_summary": r["evidence_summary"],
                    role + "_binary": r["binary_recommendation"],
                    role + "_ordinal": r["ordinal_recommendation"],
                }
            )
        out.append(row)
    csvwrite(b / "results/experiment_3_specialist_panel.csv", out)


if __name__ == "__main__":
    ROOT = output_directory()
    for pipeline in [2, 3]:
        for dataset, n in [("wago", 300), ("siemens", 100), ("smc", 200)]:
            b = ROOT / f"pipeline_{pipeline}" / dataset
            for folder in ["data", "results", "logs", "sessions"]:
                (b / folder).mkdir(parents=True, exist_ok=True)
            source = INPUTS / f"{dataset}_product_designations.csv"
            rows = read(source)
            assert len(rows) == n
            shutil.copy2(source, b / "data" / source.name)
            (b / "question.txt").write_text(QUESTION + "\n")
            js(
                b / "provenance.json",
                {
                    "products": n,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
                    "question": QUESTION,
                    "model_requested": MODEL,
                    "effort_requested": "low",
                    "product_input_columns": [
                        "product_id",
                        "ManufacturerProductDesignation",
                    ],
                    "added_decision_rubric": False,
                },
            )
            S[(pipeline, dataset)] = (b, rows, {})
    (ROOT / "question.txt").write_text(QUESTION + "\n")
    check()
    jobs = [
        (2, d, f"experiment_2_judge_{j}", "judge", "")
        for j in range(1, 6)
        for d in ["wago", "siemens", "smc"]
    ]
    roles = [
        (
            "evidence",
            "evidence analyst: identify the product role and measurement-relevant evidence for the shared question",
        ),
        (
            "skeptic",
            "skeptical reviewer: examine ambiguity and unsupported assumptions in a monitoring recommendation",
        ),
        (
            "application",
            "maintenance-application analyst: assess monitoring of this component within the production system under the shared question",
        ),
    ]
    jobs += [
        (3, d, f"experiment_3_specialist_{j}_{role}", "specialist", instruction)
        for j, (role, instruction) in enumerate(roles, 1)
        for d in ["wago", "siemens", "smc"]
    ]
    batch(jobs)
    if not STATUS["errors"] and not STOP.is_set():
        for d in ["wago", "siemens", "smc"]:
            aggregate_p2(d)
            build_panel(d)
        batch(
            [
                (3, d, "experiment_3_final_judgment", "final", "")
                for d in ["wago", "siemens", "smc"]
            ]
        )
    STATUS.update(
        status="complete" if len(STATUS["completed"]) == 27 else "incomplete",
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    check()
    print(STATUS["status"].upper(), flush=True)
    if STATUS["status"] != "complete":
        raise SystemExit(1)
