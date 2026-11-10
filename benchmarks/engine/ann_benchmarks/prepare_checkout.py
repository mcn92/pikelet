#!/usr/bin/env python3
"""Install Pikelet's adapter and native binding into an ANN-Benchmarks checkout."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shlex
import shutil
import subprocess
import sys
import sysconfig


HERE = pathlib.Path(__file__).resolve().parent
PIKELET_ROOT = HERE.parent.parent.parent


def run(command: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, cwd=cwd, check=True)


def patch_angular_distance(checkout: pathlib.Path) -> None:
    distance_py = checkout / "ann_benchmarks" / "distance.py"
    text = distance_py.read_text(encoding="utf-8")
    if "def angular(a, b):" not in text:
        marker = "\ndef euclidean(a, b):\n    return norm(a - b)\n"
        replacement = marker + """
def angular(a, b):
    an = norm(a)
    bn = norm(b)
    if an == 0 and bn == 0:
        return 0.0
    if an == 0 or bn == 0:
        return 1.0
    return 1 - np.dot(a, b) / (an * bn)
"""
        if marker not in text:
            raise SystemExit(f"Could not patch angular distance in {distance_py}")
        text = text.replace(marker, replacement, 1)
    old = '''    "angular": Metric(
        distance=lambda a, b: 1 - np.dot(a, b) / (norm(a) * norm(b)),
        distance_valid=lambda a: True
    ),'''
    new = '''    "angular": Metric(
        distance=angular,
        distance_valid=lambda a: True
    ),'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(f"Could not install angular distance hook in {distance_py}")
    distance_py.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=pathlib.Path, help="Path to an ANN-Benchmarks checkout")
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    if not (checkout / "ann_benchmarks" / "algorithms" / "base" / "module.py").is_file():
        raise SystemExit(f"Not an ANN-Benchmarks checkout: {checkout}")

    target = checkout / "ann_benchmarks" / "algorithms" / "pikelet"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("module.py", "config.yml", "Dockerfile"):
        shutil.copy2(HERE / "pikelet" / name, target / name)
    patch_angular_distance(checkout)

    includes = shlex.split(
        subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).strip()
    )
    extension = sysconfig.get_config_var("EXT_SUFFIX")
    output = checkout / f"pikelet_py{extension}"
    command = [
        args.cxx,
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-DPIKELET_ENABLE_AVX2_SIMD",
        "-mavx2",
        "-mfma",
        *includes,
        f"-I{PIKELET_ROOT / 'packages' / 'pikelet-wasm' / 'engine'}",
        str(PIKELET_ROOT / "benchmarks" / "engine" / "vibe" / "pikelet_py.cpp"),
        "-o",
        str(output),
    ]
    run(command)

    check = (
        "import pikelet_py; "
        "assert hasattr(pikelet_py, 'PikeletIndex'); "
        "print('Pikelet binding:', pikelet_py.__file__)"
    )
    run([sys.executable, "-c", check], cwd=checkout)
    print(f"Installed adapter at {target}")


if __name__ == "__main__":
    main()
