# build_dataset.py — augmentacja TRAIN + eksport oryginałów + CSV indeksy
import argparse, json, random, time, shutil, csv
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== Stałe =====
IMG_SIZE   = 256
VAL_RATIO  = 0.15
TEST_RATIO = 0.10
SEED       = 42
AUG_MULT   = 4  # augmentacje tylko dla TRAIN
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPORT_SPLITS = ("val", "test")  # które splity eksportować

# ===== Utilsy =====
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def banner(s: str):
    print("\n" + "="*len(s))
    print(s)
    print("="*len(s))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ===== Wczytywanie i augmentacja =====
def load_img_square(p: Path, size: int) -> np.ndarray | None:
    try:
        im = Image.open(p).convert("RGB")
        im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        return np.transpose(arr, (2, 0, 1))
    except Exception:
        return None

def aug_image(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    if rng.random() < 0.5: img = ImageOps.mirror(img)
    if rng.random() < 0.2: img = ImageOps.flip(img)
    if rng.random() < 0.6: img = img.rotate(rng.uniform(-10, 10), resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    if rng.random() < 0.6: img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.6: img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.6: img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    out = np.asarray(img, dtype=np.uint8)
    return np.transpose(out, (2, 0, 1))

# ===== Tworzenie rekordów =====
def build_records(pairs: List[Tuple[Path, str]], classes: List[str], aug_mult: int, seed: int, split_name: str):
    rng = random.Random(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    recs = []
    skipped = []
    pbar = tqdm(total=len(pairs) * (1 + max(0, aug_mult)), desc=f"build {split_name}", unit="img")
    for p, c in pairs:
        arr = load_img_square(p, IMG_SIZE)
        if arr is None:
            skipped.append(str(p))
            pbar.update(1)
            continue
        recs.append((arr, c2i[c]))
        pbar.update(1)
        for _ in range(aug_mult):
            recs.append((aug_image(arr, rng), c2i[c]))
            pbar.update(1)
    pbar.close()
    if skipped:
        Path("./_skipped.txt").write_text("\n".join(skipped), encoding="utf-8")
        print(f"[{split_name}] pominięto {len(skipped)} plików → _skipped.txt")
    return recs

# ===== Zapis packów =====
def write_pack(out_dir: Path, split: str, records: List[Tuple[np.ndarray, int]], classes: List[str]):
    ensure_dir(out_dir / "s000")
    if not records:
        print(f"[{split}] brak rekordów, pomijam")
        return
    shard = out_dir / "s000"
    X = np.stack([x for x, _ in records], axis=0)
    y = np.array([int(y) for _, y in records], dtype=np.int64)

    dat_path = shard / "images.dat"
    idx_path = shard / "index.npy"
    meta_path = shard / "meta.json"

    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=X.shape)
    mm[:] = X[:]
    del mm
    np.save(idx_path, y)

    f = X.astype(np.float32) / 255.0
    mean = f.mean(axis=(0, 2, 3))
    std = f.std(axis=(0, 2, 3)).clip(1e-6, None)

    C, H, W = X.shape[1:]
    meta = {
        "classes": classes,
        "num_samples": int(X.shape[0]),
        "shape": [int(C), int(H), int(W)],
        "dtype": "uint8",
        "labels_file": "index.npy",
        "data_file": "images.dat",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "split": split,
        "shard": 0
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

# ===== Binarny mapping =====
def make_binary_from_pairs(pairs: List[Tuple[Path, str]]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    def is_healthy(n: str) -> bool:
        lc = n.lower()
        return any(tok in lc for tok in ["healthy", "control", "normal", "syl", "hi", "lo", "zdrow"])
    out = []
    for p, c in pairs:
        out.append((p, "healthy" if is_healthy(c) else "diseased"))
    return out, ["healthy", "diseased"]

# ===== Eksport oryginałów + zapis CSV =====
def export_original_images(split_pairs: List[Tuple[Path, str]], out_root: Path, split: str):
    base = out_root / "export_images" / split
    ensure_dir(base)
    csv_path = out_root / "export_images" / f"{split}_index.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "full_path"])
        for src_path, cls in tqdm(split_pairs, desc=f"Export {split}", unit="img"):
            dst_dir = base / cls
            ensure_dir(dst_dir)
            dst = dst_dir / src_path.name
            if dst.exists():
                stem, suf = dst.stem, dst.suffix
                k = 1
                while (dst_dir / f"{stem}__{k}{suf}").exists():
                    k += 1
                dst = dst_dir / f"{stem}__{k}{suf}"
            shutil.copy2(src_path, dst)
            rel = dst.relative_to(out_root)
            writer.writerow([dst.name, cls, str(rel)])
    print(f"[CSV] zapisano {csv_path}")

# ===== Splitowanie =====
def simple_split(pairs: List[Tuple[Path, str]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    by_class: Dict[str, List[Path]] = {}
    for p, c in pairs:
        by_class.setdefault(c, []).append(p)
    train, val, test = [], [], []
    for c, plist in by_class.items():
        rng.shuffle(plist)
        n = len(plist)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test += [(p, c) for p in plist[:n_test]]
        val += [(p, c) for p in plist[n_test:n_test + n_val]]
        train += [(p, c) for p in plist[n_test + n_val:]]
    return train, val, test

# ===== Listowanie klas (flat layout) =====
def list_flat(src: Path) -> Tuple[List[Tuple[Path, str]], List[str]]:
    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    pairs = []
    for c in classes:
        for p in (src / c).rglob("*"):
            if p.is_file() and is_image(p):
                pairs.append((p, c))
    return pairs, classes

# ===== main =====
def main():
    ap = argparse.ArgumentParser("build_dataset (augmentacja + CSV index)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    ensure_dir(out / "cls")
    ensure_dir(out / "bin")

    banner("BUILD DATASET (FLAT LAYOUT DETECTED)")
    pairs, classes = list_flat(src)
    tr, va, te = simple_split(pairs, VAL_RATIO, TEST_RATIO, args.seed)

    # MULTICLASS
    banner("MULTICLASS PACKS")
    rec_tr = build_records(tr, classes, AUG_MULT, args.seed, "train(cls)")
    rec_va = build_records(va, classes, 0, args.seed, "val(cls)")
    write_pack(out / "cls" / "train", "train", rec_tr, classes)
    write_pack(out / "cls" / "val", "val", rec_va, classes)
    if te:
        rec_te = build_records(te, classes, 0, args.seed, "test(cls)")
        write_pack(out / "cls" / "test", "test", rec_te, classes)

    # BINARY
    banner("BINARY PACKS")
    bin_pairs, bin_classes = make_binary_from_pairs(pairs)
    trb, vab, teb = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, args.seed)
    rec_trb = build_records(trb, bin_classes, AUG_MULT, args.seed, "train(bin)")
    rec_vab = build_records(vab, bin_classes, 0, args.seed, "val(bin)")
    write_pack(out / "bin" / "train", "train", rec_trb, bin_classes)
    write_pack(out / "bin" / "val", "val", rec_vab, bin_classes)
    if teb:
        rec_teb = build_records(teb, bin_classes, 0, args.seed, "test(bin)")
        write_pack(out / "bin" / "test", "test", rec_teb, bin_classes)

    # EXPORT ORYGINAŁÓW + CSV
    for sp, split_pairs in {"val": va, "test": te}.items():
        if split_pairs:
            export_original_images(split_pairs, out, sp)

    # Zapis builder_config.json
    cfg = {
        "img_size": IMG_SIZE,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "seed": SEED,
        "aug_mult_train": AUG_MULT,
        "outputs": ["cls", "bin"],
        "export_splits": list(EXPORT_SPLITS)
    }
    (out / "builder_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    banner("ZAKOŃCZONO")

if __name__ == "__main__":
    main()
