#!/usr/bin/env python3
import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "re_result"
HTML_PATH = ROOT / "apra_dashboard.html"
ATTACK_ROUNDS = 100

LIST_FIELDS = {
    "sampled_ids", "sampled_benign_ids", "sampled_malicious_ids",
    "mad_pass_ids", "mad_reject_ids", "mad_effective_pass_ids", "mad_effective_reject_ids",
    "mad_pass_benign_ids", "mad_pass_malicious_ids", "mad_reject_benign_ids", "mad_reject_malicious_ids",
    "cluster_pass_ids", "cluster_reject_ids", "cluster_effective_pass_ids", "cluster_effective_reject_ids",
    "cluster_pass_benign_ids", "cluster_pass_malicious_ids", "cluster_reject_benign_ids", "cluster_reject_malicious_ids",
    "final_selected_ids", "final_rejected_ids", "final_selected_benign_ids", "final_selected_malicious_ids",
    "final_rejected_benign_ids", "final_rejected_malicious_ids",
}
DICT_FIELDS = {"cluster_scores"}
INT_FIELDS = {
    "epoch", "num_sampled", "num_adversaries", "mad_safety_keep_used", "mad_fallback_used",
    "cluster_best_k", "cluster_selected_cluster", "cluster_fallback_used",
    "final_selected_benign_count", "final_selected_malicious_count",
    "final_rejected_benign_count", "final_rejected_malicious_count",
}
FLOAT_FIELDS = {"mad_median_norm", "mad_mad", "mad_k", "cluster_best_score"}
PRIMARY_COLUMNS = [
    "epoch", "sampled_ids", "sampled_malicious_ids", "final_selected_ids",
    "final_rejected_ids", "screening_rate",
]


def read_csv(path):
    if not path or not path.exists():
        return [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def parse_json(value, fallback):
    if value in ("", None):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_name(directory):
    parts = directory.name.split("_")
    return {
        "id": directory.name,
        "dataset": parts[0] if len(parts) > 0 else "",
        "epochs": parts[1] if len(parts) > 1 else "",
        "created": "_".join(parts[2:4]) if len(parts) > 3 else "",
        "agg_method": parts[4] if len(parts) > 4 else "",
        "attack": "_".join(parts[7:]) if len(parts) > 7 else "",
    }


def is_apra_directory(directory):
    return (directory / "apra_round_summary.csv").exists() and (directory / "apra_client_trace.csv").exists()


def normalize_value(key, value):
    if key in LIST_FIELDS:
        parsed = parse_json(value, [])
        return parsed if isinstance(parsed, list) else []
    if key in DICT_FIELDS:
        parsed = parse_json(value, {})
        return parsed if isinstance(parsed, dict) else {}
    if key in INT_FIELDS:
        return to_int(value)
    if key in FLOAT_FIELDS:
        return to_float(value)
    return value


def normalize_summary(raw_rows):
    rows = []
    for raw in raw_rows:
        row = {key: normalize_value(key, value) for key, value in raw.items()}
        num_sampled = to_int(row.get("num_sampled"), len(row.get("sampled_ids", [])))
        final_selected = row.get("final_selected_ids", []) if isinstance(row.get("final_selected_ids"), list) else []
        sampled_malicious = row.get("sampled_malicious_ids", [])
        rejected_malicious = row.get("final_rejected_malicious_ids", [])
        if not isinstance(sampled_malicious, list):
            sampled_malicious = []
        if not isinstance(rejected_malicious, list):
            rejected_malicious = []
        row["screening_rate"] = len(rejected_malicious) / len(sampled_malicious) if sampled_malicious else None
        row["selected_rate"] = len(final_selected) / num_sampled if num_sampled else None
        row["screened_malicious_count"] = len(rejected_malicious)
        row["sampled_malicious_count"] = len(sampled_malicious)
        row["selected_count"] = len(final_selected)
        rows.append(row)
    return rows


def apra_summary(directory):
    raw_rows, raw_columns = read_csv(directory / "apra_round_summary.csv")
    rows = normalize_summary(raw_rows)
    extra_columns = [c for c in raw_columns if c not in PRIMARY_COLUMNS]
    columns = PRIMARY_COLUMNS + ["screened_malicious_count", "sampled_malicious_count", "selected_count", "selected_rate"] + extra_columns
    seen = set()
    ordered_columns = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered_columns.append(column)
    return rows, ordered_columns, raw_columns


def compute_stats(rows):
    attack_rows = [r for r in rows if 1 <= to_int(r.get("epoch")) <= ATTACK_ROUNDS]

    def aggregate(scope_rows):
        sampled_malicious = sum(len(r.get("sampled_malicious_ids", [])) for r in scope_rows)
        selected_malicious = sum(len(r.get("final_selected_malicious_ids", [])) for r in scope_rows)
        rejected_malicious = sum(len(r.get("final_rejected_malicious_ids", [])) for r in scope_rows)
        sampled = sum(to_int(r.get("num_sampled"), len(r.get("sampled_ids", []))) for r in scope_rows)
        screened = sum(len(r.get("final_rejected_ids", [])) for r in scope_rows)
        return {
            "rounds": len(scope_rows),
            "sampled_clients": sampled,
            "screened_clients": screened,
            "screening_rate": rejected_malicious / sampled_malicious if sampled_malicious else None,
            "sampled_malicious": sampled_malicious,
            "selected_malicious": selected_malicious,
            "rejected_malicious": rejected_malicious,
            "malicious_selected_rate": selected_malicious / sampled_malicious if sampled_malicious else None,
            "malicious_reject_rate": rejected_malicious / sampled_malicious if sampled_malicious else None,
        }

    attack_stats = aggregate(attack_rows)
    all_stats = aggregate(rows)
    return {
        "rounds": len(rows),
        "attack_rounds": ATTACK_ROUNDS,
        "screening_scope": f"epoch<= {ATTACK_ROUNDS}",
        "sampled_clients": attack_stats["sampled_clients"],
        "screened_clients": attack_stats["screened_clients"],
        "screening_rate": attack_stats["screening_rate"],
        "sampled_malicious": attack_stats["sampled_malicious"],
        "selected_malicious": attack_stats["selected_malicious"],
        "rejected_malicious": attack_stats["rejected_malicious"],
        "malicious_selected_rate": attack_stats["malicious_selected_rate"],
        "malicious_reject_rate": attack_stats["malicious_reject_rate"],
        "all_rounds": all_stats,
    }


def scan_experiments():
    found = []
    if not RESULT_ROOT.exists():
        return found
    for directory in sorted(RESULT_ROOT.iterdir()):
        if not directory.is_dir() or not is_apra_directory(directory):
            continue
        rows, _, _ = apra_summary(directory)
        item = parse_name(directory)
        item["has_apra"] = True
        item["stats"] = compute_stats(rows)
        found.append(item)
    found.sort(key=lambda x: (x["attack"], x["created"]))
    return found


def load_experiment(exp_id):
    directory = RESULT_ROOT / exp_id
    if not directory.exists() or not directory.is_dir() or not is_apra_directory(directory):
        return None
    rows, columns, raw_columns = apra_summary(directory)
    return {
        "meta": parse_name(directory),
        "stats": compute_stats(rows),
        "columns": columns,
        "raw_columns": raw_columns,
        "total_rows": len(rows),
    }


def load_rounds(exp_id, page=1, page_size=10):
    directory = RESULT_ROOT / exp_id
    if not directory.exists() or not directory.is_dir() or not is_apra_directory(directory):
        return None
    rows, _, _ = apra_summary(directory)
    page = max(1, to_int(page, 1))
    page_size = min(100, max(1, to_int(page_size, 10)))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "page": page,
        "page_size": page_size,
        "total_rows": len(rows),
        "total_pages": (len(rows) + page_size - 1) // page_size,
        "rows": rows[start:end],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self):
        data = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/":
            self.send_html()
        elif parsed.path == "/api/experiments":
            self.send_json({"experiments": scan_experiments()})
        elif parsed.path == "/api/experiment":
            data = load_experiment(qs.get("id", [""])[0])
            self.send_json(data if data else {"error": "APRA data not found"}, 200 if data else 404)
        elif parsed.path == "/api/rounds":
            data = load_rounds(
                qs.get("id", [""])[0],
                qs.get("page", ["1"])[0],
                qs.get("page_size", ["10"])[0],
            )
            self.send_json(data if data else {"error": "APRA data not found"}, 200 if data else 404)
        else:
            self.send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="APRA round audit dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8925, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"APRA dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
