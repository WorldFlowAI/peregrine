#!/usr/bin/env python3
"""Optional llama.cpp parity test for token IDs and final logits."""

import os
import pathlib
import subprocess
import sys

SKIP = 77


def skip(msg):
    print(f"skip: {msg}")
    return SKIP


def env_path(name):
    value = os.environ.get(name)
    if not value:
        return None
    path = pathlib.Path(value)
    return path if path.exists() else None


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <peregrine-cli>", file=sys.stderr)
        return 2

    peregrine = pathlib.Path(sys.argv[1]).resolve()
    model = env_path("PG_PARITY_MODEL")
    llama_tokenize = env_path("PG_LLAMA_CPP_TOKENIZE")
    llama_debug = env_path("PG_LLAMA_CPP_DEBUG")
    prompt = os.environ.get("PG_PARITY_PROMPT", "Once upon a time")
    top = os.environ.get("PG_PARITY_TOP", "8")

    if not peregrine.exists():
        return skip("peregrine CLI was not built")
    if model is None:
        return skip("set PG_PARITY_MODEL to a GGUF model")
    if llama_tokenize is None:
        return skip("set PG_LLAMA_CPP_TOKENIZE to llama-tokenize")
    if llama_debug is None:
        return skip("set PG_LLAMA_CPP_DEBUG to llama-debug")

    helper = pathlib.Path(__file__).resolve().parents[1] / "tools" / "llama_cpp_parity.py"
    cmd = [
        sys.executable,
        str(helper),
        "-m", str(model),
        "-p", prompt,
        "--peregrine", str(peregrine),
        "--llama-tokenize", str(llama_tokenize),
        "--llama-debug", str(llama_debug),
        "--top", top,
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
