#!/usr/bin/env python3
"""Apply approved tool/function request specs to fin-kit on `agent` branch."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help="JSON request spec")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.request).read_text(encoding="utf-8"))
    for k in ("tool_name", "module_path", "summary", "code"):
        if not str(spec.get(k, "")).strip():
            raise SystemExit(f"missing required field: {k}")

    if "subprocess" in spec["code"] or "os.system" in spec["code"]:
        raise SystemExit("unsafe code pattern detected")

    target = REPO / spec["module_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")

    req_id = spec.get("request_id", "manual")
    with target.open("a", encoding="utf-8") as f:
        f.write(f"\n\n# ---- agentic function request {req_id} ----\n")
        f.write(spec["code"].rstrip() + "\n")

    log = REPO / "AGENT_REQUESTS.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(
            f"\n## {req_id}\n"
            f"- tool_name: `{spec['tool_name']}`\n"
            f"- module_path: `{spec['module_path']}`\n"
            f"- summary: {spec['summary']}\n"
            f"- status: implemented on agent branch (pending PR review)\n"
        )

    run(["git", "checkout", "-B", "agent"])
    run(["git", "add", spec["module_path"], "AGENT_REQUESTS.md"])
    run(["git", "commit", "-m", f"agent: add requested function {spec['tool_name']} ({req_id})"])
    if args.push:
        run(["git", "push", "-u", "origin", "agent"])

    print("Implemented on agent branch. Open PR and merge after human review.")


if __name__ == "__main__":
    main()
