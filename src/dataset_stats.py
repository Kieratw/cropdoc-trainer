

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEALTHY_NAMES = {
    "healthy",
    "healthyleaf",
    "healthyleaf",
    "zdrowa",
    "zdrowe",
    "zdrowy",
    "healthy leaf",
}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_meta(out_dir: Path, split: str) -> Optional[Path]:
    """Najpierw typowa ścieżka: cls/<split>/s000/meta.json.
    Potem fallback: szukaj dowolnego meta.json w cls/<split>/*/meta.json.
    """
    p = out_dir / "cls" / split / "s000" / "meta.json"
    if p.exists():
        return p

    base = out_dir / "cls" / split
    if base.exists():
        hits = sorted(base.glob("*/meta.json"))
        if hits:
            return hits[0]
    return None


def _count_test_images(out_dir: Path) -> int:
    root = out_dir / "test_images" / "cls"
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if _is_image(p))


def _infer_classes(out_dir: Path, cfg: Optional[dict], meta_train: Optional[dict], meta_val: Optional[dict]) -> List[str]:
    classes: Optional[List[str]] = None

    if cfg and isinstance(cfg.get("classes"), list) and cfg["classes"]:
        classes = list(cfg["classes"])

    if classes is None and meta_train and isinstance(meta_train.get("classes"), list):
        classes = list(meta_train["classes"])

    if classes is None and meta_val and isinstance(meta_val.get("classes"), list):
        classes = list(meta_val["classes"])

    if classes is None:
        # Ostatecznie: katalogi klas w test_images/cls
        root = out_dir / "test_images" / "cls"
        if root.exists():
            classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

    return classes or []


def _healthy_present(classes: List[str]) -> bool:
    for c in classes:
        if c is None:
            continue
        key = str(c).strip().lower()
        if key in HEALTHY_NAMES:
            return True
        if key.startswith("healthy"):
            return True
        if key.startswith("zdrow"):
            return True
    return False


def _split_count_from_meta(meta: Optional[dict]) -> Optional[int]:
    if not meta:
        return None
    n = meta.get("num_samples")
    try:
        return int(n)
    except Exception:
        return None


def _labels_counts(out_dir: Path, split: str, classes: List[str]) -> Optional[Dict[str, int]]:
    """Zlicz etykiety na podstawie index.npy (train/val).
    Zwraca dict: {class_name: count}.
    """
    base = out_dir / "cls" / split
    if not base.exists():
        return None

    idx_paths = list(base.glob("*/index.npy"))
    if not idx_paths:
        idx_paths = list(base.glob("*/labels.npy"))
    if not idx_paths:
        return None

    p = idx_paths[0]
    try:
        y = np.load(p)
        y = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y, minlength=len(classes))
        return {classes[i]: int(counts[i]) for i in range(min(len(classes), len(counts)))}
    except Exception:
        return None


def _test_counts(out_dir: Path) -> Optional[Dict[str, int]]:
    root = out_dir / "test_images" / "cls"
    if not root.exists():
        return None
    out: Dict[str, int] = {}
    for cdir in [d for d in root.iterdir() if d.is_dir()]:
        out[cdir.name] = sum(1 for p in cdir.rglob("*") if _is_image(p))
    return out


def collect_stats_for_out_dir(out_dir: Path, per_class: bool = False) -> dict:
    out_dir = out_dir.resolve()
    cfg = _read_json(out_dir / "builder_config.json")

    meta_train = _read_json(_find_meta(out_dir, "train") or Path(""))
    meta_val = _read_json(_find_meta(out_dir, "val") or Path(""))

    train_n = _split_count_from_meta(meta_train)
    val_n = _split_count_from_meta(meta_val)
    test_n = _count_test_images(out_dir)

    classes = _infer_classes(out_dir, cfg, meta_train, meta_val)
    num_classes = len(classes)
    num_disease_classes = max(0, num_classes - (1 if _healthy_present(classes) else 0))

    total_known_parts = [x for x in [train_n, val_n, test_n] if x is not None]
    total_n = int(sum(total_known_parts)) if total_known_parts else None

    result = {
        "name": out_dir.name,
        "path": str(out_dir),
        "dataset": (cfg.get("dataset") if cfg else None) or out_dir.name,
        "schema": (cfg.get("schema") if cfg else None),
        "classes": classes,
        "num_classes_total": num_classes,
        "num_disease_classes": num_disease_classes,
        "images_train": train_n,
        "images_val": val_n,
        "images_test": test_n,
        "images_total": total_n,
    }

    if per_class and classes:
        trc = _labels_counts(out_dir, "train", classes)
        vac = _labels_counts(out_dir, "val", classes)
        tec = _test_counts(out_dir)
        result["per_class"] = {
            "train": trc,
            "val": vac,
            "test": tec,
        }

    return result


def find_out_dirs(root: Path) -> List[Path]:
    root = root.resolve()
    outs = []
    for cfg_path in root.rglob("builder_config.json"):
        out_dir = cfg_path.parent
        outs.append(out_dir)
    # dedup, zachowaj kolejność
    seen = set()
    uniq = []
    for p in outs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _fmt_int(x: Optional[int]) -> str:
    return "-" if x is None else f"{int(x)}"


def print_summary_table(stats: List[dict]) -> None:
    if not stats:
        print("Brak znalezionych builder_config.json. Podaj poprawny --root albo --dirs.")
        return

    # sort po dataset dla czytelności
    stats = sorted(stats, key=lambda d: str(d.get("dataset") or d.get("name")))

    headers = ["dataset", "schema", "klas", "chorób", "train", "val", "test", "SUMA"]
    rows = []
    for s in stats:
        rows.append([
            str(s.get("dataset", "?")),
            str(s.get("schema") or "-"),
            str(s.get("num_classes_total", "-")),
            str(s.get("num_disease_classes", "-")),
            _fmt_int(s.get("images_train")),
            _fmt_int(s.get("images_val")),
            _fmt_int(s.get("images_test")),
            _fmt_int(s.get("images_total")),
        ])

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(sep: str = "-"):
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    print(line("-"))
    print("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line("="))
    for r in rows:
        print("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line("-"))


def print_per_class(stats: List[dict]) -> None:
    for s in sorted(stats, key=lambda d: str(d.get("dataset") or d.get("name"))):
        pc = s.get("per_class")
        if not pc:
            continue
        print(f"\n[{s.get('dataset')}] per-class (train/val/test)")
        classes = s.get("classes") or []
        tr = pc.get("train") or {}
        va = pc.get("val") or {}
        te = pc.get("test") or {}

        # kolumny: class | train | val | test | sum
        widths = [max(5, max((len(str(c)) for c in classes), default=5)), 7, 7, 7, 7]
        print("  " + "class".ljust(widths[0]) + "  train".rjust(widths[1]) + "  val".rjust(widths[2]) + "  test".rjust(widths[3]) + "  sum".rjust(widths[4]))
        for c in classes:
            a = int(tr.get(c, 0) or 0)
            b = int(va.get(c, 0) or 0)
            d = int(te.get(c, 0) or 0)
            sm = a + b + d
            print("  " + str(c).ljust(widths[0]) + f"  {a:>{widths[1]}}  {b:>{widths[2]}}  {d:>{widths[3]}}  {sm:>{widths[4]}}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Policz statystyki zdjęć i klas dla packów builderów.")
    ap.add_argument("--root", type=str, default=None, help="Katalog nadrzędny, w którym są podkatalogi z builder_config.json")
    ap.add_argument("--dirs", nargs="*", default=None, help="Lista konkretnych katalogów wyjściowych builderów")
    ap.add_argument("--per-class", action="store_true", help="Wypisz dodatkowo rozkład per klasa (train/val/test)")
    ap.add_argument("--json-out", type=str, default=None, help="Zapisz wynik jako JSON do pliku")
    args = ap.parse_args()

    out_dirs: List[Path] = []
    if args.dirs:
        out_dirs += [Path(p) for p in args.dirs]
    if args.root:
        out_dirs += find_out_dirs(Path(args.root))

    # dedup
    seen = set()
    uniq = []
    for p in out_dirs:
        pp = p.resolve()
        if pp not in seen:
            uniq.append(pp)
            seen.add(pp)
    out_dirs = uniq

    stats = []
    for d in out_dirs:
        if not (d / "builder_config.json").exists():
            # jeśli user podał zły katalog, spróbuj znaleźć głębiej
            hits = list(d.rglob("builder_config.json"))
            if hits:
                d = hits[0].parent
            else:
                continue
        stats.append(collect_stats_for_out_dir(d, per_class=bool(args.per_class)))

    print_summary_table(stats)
    if args.per_class:
        print_per_class(stats)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nZapisano JSON: {Path(args.json_out).resolve()}")


if __name__ == "__main__":
    main()
