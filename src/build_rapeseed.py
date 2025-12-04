import argparse, json, random, re, io, csv, hashlib, shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageFile, ImageEnhance
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_IMG_SIZE = 380
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.10
DEFAULT_SEED = 42
DEFAULT_HASH_THRESHOLD = 3

HEALTHY_KEYS = {"syl", "healthy", "zdrowe", "zdrowy"}


# --------------------- WYKRES ASCII ---------------------

def print_distribution_chart(counts: Dict[str, int]):
    """Rysuje prosty wykres słupkowy w terminalu pokazujący liczności klas."""
    print("\n" + "=" * 65)
    print(" 📊  ROZKŁAD KLAS (CO MAMY DO PRZETWORZENIA)")
    print("=" * 65)

    if not counts:
        print(" (Brak danych)")
        return

    max_val = max(counts.values())
    total = sum(counts.values())
    max_width = 30  # szerokość paska

    # Sortujemy od najliczniejszej
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(count / max_val * max_width)
        bar = "█" * bar_len
        perc = (count / total) * 100
        # Formatowanie: Nazwa | Pasek | Liczba (Procent)
        print(f" {cls:>15} |{bar:<{max_width}}| {count:>6} fot. ({perc:5.1f}%)")

    print("-" * 65)
    print(f" {'SUMA':>15} |{'':<{max_width}}| {total:>6} fot. (100.0%)")
    print("=" * 65 + "\n")


# --------------------- HASH / DEDUP ---------------------

def _read_bytes_for_hash(p: Path) -> Optional[bytes]:
    try:
        return p.read_bytes()
    except Exception:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=95)
                return buf.getvalue()
        except Exception:
            return None


def _sha1(p: Path) -> Optional[str]:
    b = _read_bytes_for_hash(p)
    return hashlib.sha1(b).hexdigest() if b is not None else None


def _ahash64(p: Path, size: int = 8) -> Optional[int]:
    """Prosty average-hash 8x8 → 64-bit int."""
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32)
            m = a.mean()
            bits = (a >= m).astype("uint8").flatten()
            out = 0
            for b in bits:
                out = (out << 1) | int(b)
            return int(out)
    except Exception:
        return None


def _ham64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))


class DedupGuard:
    """
    Pilnuje, żeby w teście nie było:
    - identycznych plików jak w train/val (sha1)
    - prawie identycznych obrazów (aHash 8x8, Hamming <= threshold)
    + opcjonalnie deduplikuje duplikaty wewnątrz samego testu.
    """

    def __init__(self, threshold: int, within_test: bool, log_csv: Optional[Path]):
        self.th = int(threshold)
        self.within = bool(within_test)
        self.log_csv = log_csv
        self.idx_sha = set()
        self.idx_ph: List[int] = []

    def add_pairs(self, pairs: List[Tuple[Path, str]]) -> None:
        # --- ZMIANA: Dodano tqdm ---
        for p, _ in tqdm(pairs, desc="[Dedup] Indeksowanie bazy (Train/Val)", unit="img", colour='green'):
            p = Path(p)
            if not p.exists():
                continue
            s = _sha1(p)
            h = _ahash64(p)
            if s:
                self.idx_sha.add(s)
            if h is not None:
                self.idx_ph.append(h)

    def filter_test(self, test_pairs: List[Tuple[Path, str]]):
        kept: List[Tuple[Path, str]] = []
        removed: List[Tuple[Path, str, str]] = []

        # 1) względem train/val
        # --- ZMIANA: Dodano tqdm ---
        for p, c in tqdm(test_pairs, desc="[Dedup] Filtrowanie Testu", unit="img", colour='green'):
            p = Path(p)
            if not p.exists():
                removed.append((p, c, "missing"))
                continue

            s = _sha1(p)
            h = _ahash64(p)

            if s and s in self.idx_sha:
                removed.append((p, c, "bytes_equal_train_val"))
                continue

            if h is not None and any(_ham64(h, r) <= self.th for r in self.idx_ph):
                removed.append((p, c, f"near_dup_train_val_ham<={self.th}"))
                continue

            kept.append((p, c))

        # 2) dedup wewnątrz testu
        if self.within and kept:
            new_kept: List[Tuple[Path, str]] = []
            seen_s = set()
            seen_h: List[int] = []

            # Tu zazwyczaj jest mało elementów, ale dla spójności można dodać,
            # choć przy małym teście mignie niezauważalnie.
            for p, c in kept:
                s = _sha1(p)
                h = _ahash64(p)

                if s and s in seen_s:
                    removed.append((p, c, "dup_within_test_bytes"))
                    continue
                if h is not None and any(_ham64(h, r) <= self.th for r in seen_h):
                    removed.append((p, c, f"near_dup_within_test_ham<={self.th}"))
                    continue

                if s:
                    seen_s.add(s)
                if h is not None:
                    seen_h.append(h)
                new_kept.append((p, c))

            kept = new_kept

        # log
        if self.log_csv:
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "class", "action", "reason"])
                for p, c in kept:
                    w.writerow([str(p), c, "keep", ""])
                for p, c, r in removed:
                    w.writerow([str(p), c, "drop", r])

        return kept, removed


# --------------------- IO / PACK ---------------------

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def load_img_square(path: Path, size: int) -> Optional[np.ndarray]:
    """Wczytaj obraz, zrób resize z zachowaniem proporcji + center crop do kwadratu."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < h:
                nw, nh = size, int(round(h * size / w))
            else:
                nh, nw = size, int(round(w * size / h))
            im = im.resize((nw, nh), Image.BICUBIC)
            l = (nw - size) // 2
            t = (nh - size) // 2
            im = im.crop((l, t, l + size, t + size))
            arr = np.asarray(im, dtype=np.uint8)  # HWC
            return np.transpose(arr, (2, 0, 1))  # CHW
    except Exception:
        return None


def to_records(pairs: List[Tuple[Path, str]],
               classes: List[str],
               img_size: int,
               split_name: str,
               step_desc: str) -> List[Tuple[np.ndarray, int]]:
    c2i = {c: i for i, c in enumerate(classes)}
    out: List[Tuple[np.ndarray, int]] = []
    # --- ZMIANA: Ulepszony opis paska ---
    pbar = tqdm(pairs, desc=step_desc, unit="img", colour='blue')
    for p, c in pbar:
        x = load_img_square(p, img_size)
        if x is None:
            continue
        y = c2i[c]
        out.append((x, y))
    pbar.close()
    return out


def write_pack(out_dir: Path,
               split: str,
               records: List[Tuple[np.ndarray, int]],
               classes: List[str],
               step_desc: str) -> None:
    if not records:
        print(f"[{split}] brak rekordów, nic nie zapisuję.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    C, H, W = records[0][0].shape
    N = len(records)

    shard = out_dir / "s000"
    shard.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(shard / "images.dat",
                   dtype=np.uint8,
                   mode="w+",
                   shape=(N, C, H, W))
    labels = np.empty((N,), dtype=np.int64)

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)

    # --- ZMIANA: Ulepszony opis paska ---
    for i, (x, y) in enumerate(tqdm(records, desc=step_desc, unit="img", colour='yellow')):
        mm[i] = x
        labels[i] = y

        xf = x.astype(np.float32) / 255.0
        sum_c += xf.reshape(3, -1).sum(axis=1)
        sumsq_c += (xf ** 2).reshape(3, -1).sum(axis=1)

    mm.flush()
    np.save(shard / "index.npy", labels)

    total_pix = N * H * W
    mean = (sum_c / total_pix).tolist()
    var = (sumsq_c / total_pix) - np.square(sum_c / total_pix)
    std = np.sqrt(np.clip(var, 1e-12, None)).tolist()

    meta = {
        "classes": classes,
        "num_samples": int(N),
        "shape": [3, H, W],
        "dtype": "uint8",
        "labels_file": "index.npy",
        "data_file": "images.dat",
        "mean": mean,
        "std": std,
        "split": split,
        "shard": 0,
    }
    (shard / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def copy_test_images(pairs: List[Tuple[Path, str]], dest_root: Path) -> None:
    if not pairs:
        print("[test_images] brak próbek.")
        return

    # --- ZMIANA: Ulepszony opis paska ---
    for p, c in tqdm(pairs, desc="[5/5] Kopiowanie Test Images", unit="img", colour='cyan'):
        cls_dir = dest_root / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        target = cls_dir / p.name
        if target.exists():
            k = 1
            while (cls_dir / f"{p.stem}_{k}{p.suffix}").exists():
                k += 1
            target = cls_dir / f"{p.stem}_{k}{p.suffix}"
        try:
            shutil.copy2(p, target)
        except Exception as e:
            print(f"[copy] {p} → {target}: {e}")


def stratified_split(pairs: List[Tuple[Path, str]],
                     val_ratio: float,
                     test_ratio: float,
                     seed: int):
    rng = random.Random(seed)
    by_class: Dict[str, List[Path]] = {}
    for p, c in pairs:
        by_class.setdefault(c, []).append(p)

    train, val, test = [], [], []

    for c, plist in by_class.items():
        plist = plist[:]
        rng.shuffle(plist)
        n = len(plist)

        n_test = int(round(n * test_ratio))
        n_val = int(round((n - n_test) * val_ratio))

        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n - 1))
            n_val = max(0, min(n_val, n - 1 - n_test))

        test += [(p, c) for p in plist[:n_test]]
        val += [(p, c) for p in plist[n_test:n_test + n_val]]
        train += [(p, c) for p in plist[n_test + n_val:]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# --------------------- SKANOWANIE ŹRÓDŁA ---------------------

HILO_RE = re.compile(r"^\s*([A-Za-z0-9]+)_(Hi|Lo)\s*$", re.IGNORECASE)


def _canonical_class_from_folder_name(name: str) -> str:
    """
    Alt_Hi / Alt_Lo → Alt
    Syl_Hi / Syl_Lo → healthy
    """
    m = HILO_RE.match(name)
    base = m.group(1) if m else name
    key = re.sub(r"[^A-Za-z0-9]+", "", base).lower()
    if key in HEALTHY_KEYS:
        return "healthy"
    return base


def scan_hilo(src: Path):
    """
    Szuka folderów *_Hi oraz *_Lo bez wgłębiania w bardziej chore struktury.
    """
    pairs: List[Tuple[Path, str]] = []
    debug_rows = []

    dirs = [d for d in src.iterdir() if d.is_dir()]
    for d in sorted(dirs, key=lambda x: x.name):
        if not HILO_RE.match(d.name):
            # ignorujemy dziwne katalogi
            continue
        cls = _canonical_class_from_folder_name(d.name)
        cnt = 0
        for p in d.rglob("*"):
            if is_image(p):
                pairs.append((p, cls))
                cnt += 1
        debug_rows.append({"folder": d.name, "class": cls, "count": cnt})

    return pairs, debug_rows


# --------------------- META DLA UI ---------------------

def write_ui_meta(out_dir: Path, classes: List[str]) -> None:
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # healthy zawsze na końcu
    ordered = [c for c in classes if c != "healthy"] + (
        ["healthy"] if "healthy" in classes else []
    )

    translation = {"healthy": "zdrowa"}
    for c in classes:
        translation.setdefault(c, c)

    data = {
        "translation": translation,
        "display_order": ordered,
    }

    (meta_dir / "labels_pl.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------- MAIN ---------------------

def main():
    ap = argparse.ArgumentParser("Rapeseed CLS builder (Hi/Lo, Syl_* → healthy, no offline aug)")
    ap.add_argument("--src", required=True, help="Katalog źródłowy z Alt_Hi, Alt_Lo, Syl_Hi itd.")
    ap.add_argument("--out", required=True, help="Katalog wyjściowy na packi")
    ap.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--hash_threshold", type=int, default=DEFAULT_HASH_THRESHOLD)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- skan ---
    print(f"[scan] źródło: {src}")
    cls_pairs, scan_debug = scan_hilo(src)
    (log_dir / "scan_debug.json").write_text(
        json.dumps(scan_debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not cls_pairs:
        raise SystemExit("Brak obrazów w źródle. Albo ścieżka zła, albo katalogi nie są w formacie *_Hi/_Lo.")

    # klasy: healthy na końcu
    uniq = sorted({c for _, c in cls_pairs if c != "healthy"})
    if any(c == "healthy" for _, c in cls_pairs):
        uniq.append("healthy")
    classes = uniq

    # --- ZMIANA: Wyświetlanie wykresu ---
    counts: Dict[str, int] = {}
    for _, c in cls_pairs:
        counts[c] = counts.get(c, 0) + 1

    print_distribution_chart(counts)

    # --- split ---
    tr, va, te = stratified_split(cls_pairs, args.val_ratio, args.test_ratio, args.seed)

    # --- dedup testu ---
    dg = DedupGuard(
        threshold=args.hash_threshold,
        within_test=True,
        log_csv=log_dir / "dedup_test.csv",
    )
    # Teraz dg.add_pairs ma w środku pasek postępu
    dg.add_pairs(tr + va)
    # Teraz dg.filter_test ma w środku pasek postępu
    te, removed = dg.filter_test(te)

    if removed:
        print(f"[dedup] Z testu usunięto {len(removed)} duplikatów / podobnych.")

    # Dodane opisy kroków [x/5] dla czytelności
    rec_train = to_records(tr, classes, args.img_size, "train", "[3/5] Build Train Pack")
    rec_val = to_records(va, classes, args.img_size, "val", "[4/5] Build Val Pack")

    write_pack(out / "cls" / "train", "train", rec_train, classes, "[3/5] Write Train Pack")
    write_pack(out / "cls" / "val", "val", rec_val, classes, "[4/5] Write Val Pack")

    copy_test_images(te, out / "test_images" / "cls")
    write_ui_meta(out, classes)

    # zapis prostego configu info
    cfg = {
        "dataset": "rapeseed",
        "schema": "hilo",
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "img_size": args.img_size,
        "seed": args.seed,
        "hash_threshold": args.hash_threshold,
        "classes": classes,
    }
    (out / "builder_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nGOTOWE ✅")
    print(f"  train: {len(rec_train)}")
    print(f"  val:   {len(rec_val)}")
    print(f"  test:  {len(te)} (w formie test_images/cls/*)")


if __name__ == "__main__":
    main()