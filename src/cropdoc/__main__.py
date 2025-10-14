import argparse
import sys
from .train import run as train_run

def main():
    parser = argparse.ArgumentParser(prog="cropdoc", description="CropDoc trainer and tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train dual-head model")
    p_train.add_argument("--bin", required=True, type=str, help="Path to BIN packs directory (contains train/val with s000 shard)")
    p_train.add_argument("--cls", required=True, type=str, help="Path to CLS packs directory (contains train/val with s000 shard)")
    p_train.add_argument("--out", required=True, type=str, help="Output runs dir")
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--batch", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--workers", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--no_compile", action="store_true", help="Force disable torch.compile")
    p_train.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.cmd == "train":
        return train_run(args)
    else:
        print("Unknown command", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
