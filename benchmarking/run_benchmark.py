"""Run any configured benchmark with the shared optimization wrapper.

Examples:
    python benchmarking/run_benchmark.py --model ArioulNet --dataset mango_new
    python benchmarking/run_benchmark.py --model ResNet18_1D --dataset wheat
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.configurations import get_spec
from benchmarking.global_wrapper import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run a configured spectral benchmark")
    parser.add_argument("--model", required=True, choices=[
        "ArioulNet", "Hmar_Net", "DeepSpectraCNN", "ResNet18_1D",
        "ResNet34_1D", "ResNet50_1D", "ResNet101_1D", "ViT_1D",
    ])
    parser.add_argument("--dataset", required=True, choices=["mango_new", "ossl", "wheat"])
    parser.add_argument("--data-path", help="Override the dataset path from configurations.py")
    args = parser.parse_args()
    spec = get_spec(args.model, args.dataset)
    if args.data_path:
        spec.data_path = args.data_path
    run_benchmark(spec)


if __name__ == "__main__":
    main()
