
import sys
from . import build as _build, train as _train, infer as _infer, viz as _viz
def main():
    if len(sys.argv) <= 1:
        print("Usage: python -m cropdoc <build|train|infer|viz> [args...]")
        raise SystemExit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "build": _build.main()
    elif cmd == "train": _train.main()
    elif cmd == "infer": _infer.main()
    elif cmd == "viz": _viz.main()
    else:
        print(f"Unknown subcommand: {cmd}"); raise SystemExit(2)
if __name__ == "__main__":
    main()
