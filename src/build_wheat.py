# build_flat_packs_jpg_test.py
import argparse, json, random, re, shutil, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

# ---------- konfig domyślny ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEALTHY_RE = re.compile(r"(healthy|control|normal|none|ok)", re.IGNORECASE)

DEFAULT_IMG_SIZE = 256
DEFAULT_SEED = 1337
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.20
DEFAULT_AUG_MULT_CLS = 2
DEFAULT_AUG_MULT_BIN = 2

# ---------- utils ----------
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def is_healthy_name(name: str) -> bool:
    return bool(HEALTHY_RE.search(name))

def list_class_dirs(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])

def list_pairs_flat(src: Path, normalize_healthy: bool) -> List[Tuple[Path, str]]:
    pairs = []
    for cdir in list_class_dirs(src):
        cname = "healthy" if (normalize_healthy and is_healthy_name(cdir.name)) else cdir.name
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                pairs.append((p, cname))
    if not pairs:
        raise SystemExit(f"Nie znaleziono obrazów w {src}")
    return pairs

def stratified_split(
    pairs: List[Tuple[Path, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Stratyfikacja per klasa, z bezpiecznikami dla małych klas."""
    rng = random.Random(seed)
    by_cls: Dict[str, List[Path]] = defaultdict(list)
    for p, c in pairs: by_cls[c].append(p)

    tr, va, te = [], [], []
    for c, plist in by_cls.items():
        plist = plist[:]; rng.shuffle(plist)
        n = len(plist)
        n_te = int(round(n * test_ratio))
        n_va = int(round((n - n_te) * val_ratio))
        # bezpieczniki
        if n_te + n_va >= n:
            n_va = max(0, n - 1 - n_te)
        if n >= 5 and n_te == 0: n_te = 1
        if n >= 8 and n_va == 0: n_va = 1
        if n_te + n_va >= n:
            n_te = max(0, min(n_te, n - 1))
            n_va = max(0, min(n_va, n - 1 - n_te))

        te_list = plist[:n_te]
        va_list = plist[n_te:n_te+n_va]
        tr_list = plist[n_te+n_va:]

        tr += [(p, c) for p in tr_list]
        va += [(p, c) for p in va_list]
        te += [(p, c) for p in te_list]

    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

# ---------- transforms ----------
def tfm_base(size: int):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])

def tfm_aug(size: int):
    return T.Compose([
        T.Resize(int(size*1.05)),
        T.CenterCrop(int(size*1.05)),
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.ColorJitter(0.12, 0.12, 0.12, 0.04),
        T.RandomRotation(12),
        T.ToTensor()
    ])

# ---------- zapis paczek (images.dat + index.npy + meta.json) ----------
@dataclass
class PackSpec:
    out_dir: Path
    split: str
    class_names: List[str]
    aug_mult: int
    img_size: int

def write_pack(spec: PackSpec, pairs: List[Tuple[Path, str]]):
    C, H, W = 3, spec.img_size, spec.img_size
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    # liczba rekordów po augmentacji (tylko train replikujemy)
    n = sum(spec.aug_mult if spec.split == "train" else 1 for _ in pairs)

    data_path = spec.out_dir / "images.dat"
    idx_path  = spec.out_dir / "index.npy"
    meta_path = spec.out_dir / "meta.json"

    mm = np.memmap(data_path, dtype=np.uint8, mode="w+", shape=(n, C, H, W))
    ys = np.empty((n,), dtype=np.int64)

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    total_pix = 0

    t_base = tfm_base(spec.img_size)
    t_aug  = tfm_aug(spec.img_size)

    cursor = 0
    for img_path, cname in tqdm(pairs, desc=f"build {spec.out_dir.parent.name}/{spec.out_dir.name} [{spec.split}]", unit="img"):
        y = spec.class_names.index(cname)
        reps = spec.aug_mult if spec.split == "train" else 1
        for _ in range(reps):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                t = (t_aug if spec.split == "train" else t_base)(im)  # float [0,1], CHW
            arr_f = t.numpy()
            sum_c += arr_f.reshape(3, -1).sum(1)
            sumsq_c += (arr_f**2).reshape(3, -1).sum(1)
            total_pix += H*W

            arr_u8 = (arr_f * 255.0).clip(0, 255).astype(np.uint8)
            mm[cursor] = arr_u8; ys[cursor] = y; cursor += 1

    mm.flush(); np.save(idx_path, ys)

    mean = (sum_c / total_pix).tolist()
    var  = (sumsq_c / total_pix) - np.square(sum_c / total_pix)
    std  = np.sqrt(np.clip(var, 1e-12, None)).tolist()

    meta = {
        "classes": spec.class_names,
        "num_samples": int(cursor),
        "shape": [3, H, W],
        "dtype": "uint8",
        "labels_file": "index.npy",
        "data_file": "images.dat",
        "mean": mean,
        "std": std,
        "split": spec.split,
        "shard": 0
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ---------- BIN pomocnicze ----------
def to_bin_pairs(pairs: List[Tuple[Path, str]]):
    out = []
    for p, c in pairs:
        out.append((p, "healthy" if (c.lower()=="healthy" or is_healthy_name(c)) else "diseased"))
    return out, ["healthy", "diseased"]

# ---------- test_images: tylko JPG ----------
def ensure_jpg_from_any(src_path: Path, dst_path: Path, quality: int = 95):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst_path, format="JPEG", quality=quality, optimize=True)

def unique_name(dst_dir: Path, base: str) -> str:
    name = f"{base}.jpg"
    if not (dst_dir / name).exists():
        return name
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}.jpg"

def export_jpg_tree(pairs: List[Tuple[Path, str]], out_root: Path, quality: int = 95):
    for src, cls in tqdm(pairs, desc=f"copy JPG → {out_root.name}", unit="img"):
        dst_dir = out_root / cls
        dst_name = unique_name(dst_dir, src.stem)
        ensure_jpg_from_any(src, dst_dir / dst_name, quality=quality)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Flat → packs(train/val) + JPG test (no leakage)")
    ap.add_argument("--src", required=True, help="Płaski root z klasami (foldery klas)")
    ap.add_argument("--out", required=True, help="Folder wyjściowy (powstanie cls/, bin/, test_images/)")
    ap.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--aug_mult_cls", type=int, default=DEFAULT_AUG_MULT_CLS)
    ap.add_argument("--aug_mult_bin", type=int, default=DEFAULT_AUG_MULT_BIN)
    ap.add_argument("--normalize_healthy", action="store_true",
                    help="Zamień Healthy/HealthyLeaf/Control/Normal/... na 'healthy' (zalecane)")
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    set_seeds(args.seed)
    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) odczyt i normalizacja nazw
    all_pairs = list_pairs_flat(src, normalize_healthy=args.normalize_healthy)
    classes_cls = sorted(set(c for _, c in all_pairs))
    print("Klasy CLS:", classes_cls)

    # 2) stratyfikowany split
    tr_cls, va_cls, te_cls = stratified_split(all_pairs, args.val_ratio, args.test_ratio, args.seed)
    print(f"Rozmiary: train={len(tr_cls)}  val={len(va_cls)}  test={len(te_cls)}")

    # 3) PACZKI CLS: tylko train/val (testu nie pakujemy!)
    write_pack(PackSpec(out/"cls"/"train"/"s000", "train", classes_cls, args.aug_mult_cls, args.img_size), tr_cls)
    if va_cls:
        write_pack(PackSpec(out/"cls"/"val"/"s000",   "val",   classes_cls, 1, args.img_size), va_cls)

    # 4) PACZKI BIN: train/val z (train+val); zero przecieku testu
    bin_tr_pairs, bin_classes = to_bin_pairs(tr_cls)
    bin_va_pairs, _           = to_bin_pairs(va_cls)
    write_pack(PackSpec(out/"bin"/"train"/"s000", "train", bin_classes, args.aug_mult_bin, args.img_size), bin_tr_pairs)
    write_pack(PackSpec(out/"bin"/"val"/"s000",   "val",   bin_classes, 1,                 args.img_size), bin_va_pairs)

    # 5) TEST: tylko JPG 1:1 (CLS i BIN)
    if te_cls:
        export_jpg_tree(te_cls, out/"test_images"/"cls", quality=args.jpg_quality)
        te_bin_pairs, _ = to_bin_pairs(te_cls)
        export_jpg_tree(te_bin_pairs, out/"test_images"/"bin", quality=args.jpg_quality)

    # 6) builder_config + podsumowanie
    cfg = {
        "img_size": args.img_size,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "aug_mult_cls": args.aug_mult_cls,
        "aug_mult_bin": args.aug_mult_bin,
        "classes_cls": classes_cls,
        "classes_bin": ["healthy", "diseased"],
        "notes": [
            "Packs: CLS(train/val), BIN(train/val); TEST tylko JPG w test_images/",
            "BIN trenujemy wyłącznie na train+val (no leakage).",
            "Opcjonalna normalizacja healthy → 'healthy' (--normalize_healthy)."
        ]
    }
    (out/"builder_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    summ = {
        "train": len(tr_cls), "val": len(va_cls), "test": len(te_cls),
        "classes": classes_cls
    }
    (out/"split_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    print("GOTOWE ✅  Wyjście:", out.as_posix())
    if te_cls:
        print("Test JPG:", (out/"test_images").as_posix())

if __name__ == "__main__":
    main()
