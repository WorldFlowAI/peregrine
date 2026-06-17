#!/usr/bin/env python3
"""Compare Peregrine token/logit output against llama.cpp debug tools."""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def parse_ints(text):
    return [int(x) for x in re.findall(r"-?\d+", text)]


def peregrine_tokens(peregrine, model, prompt):
    out = run([peregrine, "tokenize", "-m", model, "-p", prompt, "--ids-only"])
    return parse_ints(out.stdout)


def llamacpp_tokens(llama_tokenize, model, prompt):
    out = run([
        llama_tokenize,
        "-m", model,
        "-p", prompt,
        "--ids",
        "--no-escape",
        "--log-disable",
    ])
    return parse_ints(out.stdout)


def peregrine_top_logits(peregrine, model, prompt, top_n):
    out = run([peregrine, "logits", "-m", model, "-p", prompt, "--top", str(top_n)])
    tokens = []
    tops = []
    in_top = False
    for line in out.stdout.splitlines():
        if line.startswith("prompt_tokens:"):
            tokens = parse_ints(line)
        elif line.startswith("top_logits:"):
            in_top = True
        elif in_top and line.strip():
            fields = line.split()
            if len(fields) >= 3:
                tops.append((int(fields[1]), float(fields[2])))
    return tokens, tops


def llamacpp_logits(llama_debug, model, prompt):
    with tempfile.TemporaryDirectory(prefix="peregrine-llama-parity-") as tmp:
        run([
            llama_debug,
            "-m", model,
            "-p", prompt,
            "--save-logits",
            "--logits-output-dir", tmp,
            "--log-disable",
            "--no-warmup",
            "-ctk", "f32",
            "-ctv", "f32",
            "-t", "1",
            "-tb", "1",
        ])
        paths = sorted(pathlib.Path(tmp).glob("llamacpp-*.txt"))
        paths = [p for p in paths if not p.name.endswith("-prompt.txt")]
        if not paths:
            raise RuntimeError("llama-debug did not write a logits text file")
        logits = {}
        for line in paths[0].read_text().splitlines():
            m = re.match(r"\s*(\d+):\s*([-+0-9.eE]+)\s*$", line)
            if m:
                logits[int(m.group(1))] = float(m.group(2))
        if not logits:
            raise RuntimeError(f"no logits parsed from {paths[0]}")
        prompt_file = pathlib.Path(str(paths[0]).removesuffix(".txt") + "-prompt.txt")
        tokens = []
        if prompt_file.exists():
            for line in prompt_file.read_text().splitlines():
                if line.startswith("token ids:"):
                    tokens = parse_ints(line)
                    break
        return tokens, logits


def top_from_logits(logits, top_n):
    return sorted(logits.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def compare_tokens(pg, ref):
    if pg == ref:
        print(f"tokens: ok ({len(pg)} tokens)")
        return True
    print("tokens: mismatch")
    print(f"  peregrine: {pg}")
    print(f"  llama.cpp: {ref}")
    first = next((i for i, pair in enumerate(zip(pg, ref)) if pair[0] != pair[1]), None)
    if first is not None:
        print(f"  first mismatch at index {first}: peregrine={pg[first]} llama.cpp={ref[first]}")
    elif len(pg) != len(ref):
        print(f"  length mismatch: peregrine={len(pg)} llama.cpp={len(ref)}")
    return False


def compare_logits(pg_top, ref_logits, top_n, atol, rtol):
    ref_top = top_from_logits(ref_logits, top_n)
    pg_ids = [tid for tid, _ in pg_top]
    ref_ids = [tid for tid, _ in ref_top]
    ok = True

    if pg_ids == ref_ids:
        print(f"top-{top_n} ids: ok")
    else:
        ok = False
        print(f"top-{top_n} ids: mismatch")
        print(f"  peregrine: {pg_ids}")
        print(f"  llama.cpp: {ref_ids}")

    for tid, pg_val in pg_top:
        if tid not in ref_logits:
            ok = False
            print(f"logit {tid}: missing from llama.cpp output")
            continue
        ref_val = ref_logits[tid]
        tol = atol + rtol * abs(ref_val)
        diff = abs(pg_val - ref_val)
        if diff > tol:
            ok = False
            print(f"logit {tid}: mismatch peregrine={pg_val:.9g} llama.cpp={ref_val:.9g} diff={diff:.3g} tol={tol:.3g}")
    if ok:
        print(f"top-{top_n} logits: ok")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("--peregrine", default="./build-arm64-m3/peregrine")
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--llama-debug")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--tokens-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    ok = True
    pg_tokens = peregrine_tokens(args.peregrine, args.model, args.prompt)
    ref_tokens = llamacpp_tokens(args.llama_tokenize, args.model, args.prompt)
    ok &= compare_tokens(pg_tokens, ref_tokens)

    if not args.tokens_only and args.llama_debug:
        pg_prompt_tokens, pg_top = peregrine_top_logits(args.peregrine, args.model, args.prompt, args.top)
        ref_prompt_tokens, ref_logits = llamacpp_logits(args.llama_debug, args.model, args.prompt)
        if ref_prompt_tokens:
            ok &= compare_tokens(pg_prompt_tokens, ref_prompt_tokens)
        ok &= compare_logits(pg_top, ref_logits, args.top, args.atol, args.rtol)
    elif not args.tokens_only:
        print("logits: skipped (pass --llama-debug to compare)")

    return 0 if ok or args.report_only else 1


if __name__ == "__main__":
    sys.exit(main())
