import argparse, json, random, re, shutil, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


HASH_THRESHOLD = 2

import hashlib, io, csv
from PIL import ImageFile as _IFX
_IFX.LOAD_TRUNCATED_IMAGES = True

def _read_bytes_for_hash(p: Path):
    try:
        return p.read_bytes()
    except Exception:
        try:
            from PIL import Image
            with Image.open(p) as im:
                im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=95)
                return buf.getvalue()
        except Exception:
            return None

def _sha1(p: Path):
    b = _read_bytes_for_hash(p)
    if b is None: return None
    import hashlib as _hh
    return _hh.sha1(b).hexdigest()

def _ahash64(p: Path, size: int = 8):
    try:
        from PIL import Image
        import numpy as _np
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            a = _np.asarray(im, dtype=_np.float32)
            m = a.mean()
            bits = (a >= m).astype("uint8").flatten()
            out = 0
            for b in bits:
                out = (out << 1) | int(b)
            return int(out)
    except Exception:
        return None

def _ham64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1<<64)-1)).count("1"))

class DedupGuard:
    def __init__(self, threshold: int = HASH_THRESHOLD, within_test: bool = True, log_csv: Path|None = None):
        self.th = int(threshold)
        self.within = bool(within_test)
        self.log_csv = log_csv
        self.idx_sha = set()
        self.idx_ph = []  # perceptual hashes

    def add_pairs(self, pairs):
        for p, _ in pairs:
            if not Path(p).exists(): 
                continue
            s = _sha1(p)
            h = _ahash64(p)
            if s: self.idx_sha.add(s)
            if h is not None: self.idx_ph.append(h)

    def filter_test(self, test_pairs):
        kept = []
        removed = []
        # vs train/val
        for p, c in test_pairs:
            pp = Path(p)
            if not pp.exists():
                removed.append((p, c, "missing")); continue
            s = _sha1(pp); h = _ahash64(pp)
            if s and s in self.idx_sha:
                removed.append((p, c, "bytes_equal(train/val)")); continue
            if h is not None and any(_ham64(h, r) <= self.th for r in self.idx_ph):
                removed.append((p, c, f"near_dup(train/val)_ham<={self.th}")); continue
            kept.append((p, c))
        # within test
        if self.within and kept:
            seen_s = set(); seen_h = []
            out = []
            for p, c in kept:
                s = _sha1(Path(p)); h = _ahash64(Path(p))
                if s and s in seen_s:
                    removed.append((p, c, "dup_within_test_bytes")); continue
                if h is not None and any(_ham64(h, r) <= self.th for r in seen_h):
                    removed.append((p, c, f"near_dup_within_test_ham<={self.th}")); continue
                if s: seen_s.add(s)
                if h is not None: seen_h.append(h)
                out.append((p, c))
            kept = out
        if self.log_csv:
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["path","class","action","reason"])
                for p,c in kept: w.writerow([str(p), c, "keep", ""])
                for p,c,r in removed: w.writerow([str(p), c, "drop", r])
        return kept, removed
# === /NO-LEAK GUARD ===



# ===== Domyślne =====
IMG_SIZE   = 256
VAL_RATIO  = 0.15
SEED       = 42
AUG_MULT   = 0
IMG_EXTS   = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ===== Utils =====
def banner(txt): print("\n" + "="*8 + " " + txt + " " + "="*8)

def load_img_square(path: Path, size: int) -> np.ndarray | None:
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
            return np.transpose(arr, (2, 0, 1))   # CHW
    except Exception:
        return None

def aug_image(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    # proste, szybkie augmenty
    if rng.random() < 0.6:
        img = img.rotate(rng.uniform(-10, 10), resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    if rng.random() < 0.6:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.6:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.6:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    out = np.asarray(img, dtype=np.uint8)
    return np.transpose(out, (2, 0, 1))

def write_pack(out_dir: Path, split: str, records: List[Tuple[np.ndarray, int]], classes: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"[{split}] brak próbek, pomijam")
        return
    C, H, W = records[0][0].shape
    N = len(records)
    part = out_dir / "s000"
    part.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(part / "images.dat", dtype=np.uint8, mode="w+", shape=(N, C, H, W))
    labels = np.empty((N,), dtype=np.int64)

    sum_c = np.zeros(3); sumsq_c = np.zeros(3)
    for i, (x, y) in enumerate(tqdm(records, desc=f"pack {split}", unit="img")):
        mm[i] = x; labels[i] = y
        xf = x.astype(np.float32) / 255.0
        sum_c += xf.reshape(3, -1).mean(1)
        sumsq_c += (xf.reshape(3, -1) ** 2).mean(1)
    mm.flush(); np.save(part / "index.npy", labels)

    mean = (sum_c / N)
    var  = (sumsq_c / N) - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-6))
    (part / "meta.json").write_text(json.dumps({
        "classes": classes,
        "num_samples": int(N),
        "shape": [int(C), int(H), int(W)],
        "dtype": "uint8",
        "labels_file": "index.npy",
        "data_file": "images.dat",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "split": split,
        "shard": 0
    }, indent=2, ensure_ascii=False), encoding="utf-8")

def copy_test_images(pairs: List[Tuple[Path, str]], dest_root: Path, tag: str):
    if not pairs:
        print(f"[test→img/{tag}] brak próbek, pomijam")
        return
    for p, c in tqdm(pairs, desc=f"copy test→img/{tag}", unit="img"):
        cls_dir = dest_root / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        target = cls_dir / p.name
        if target.exists():
            stem, suf = p.stem, p.suffix; k = 1
            while (cls_dir / f"{stem}_{k}{suf}").exists():
                k += 1
            target = cls_dir / f"{stem}_{k}{suf}"
        try:
            shutil.copy2(p, target)
        except Exception as e:
            print(f"[copy] nie udało się skopiować {p} → {target}: {e}")

# ===== Tomato PlantVillage (CV) =====
def _norm_tomato_cls(name: str) -> str:
    n = name.lower()
    n = n.replace("tomato___", "").replace("tomato__", "").replace("tomato_", "")
    n = n.replace("__", "_").replace(" ", "_").replace("-", "_")
    if "healthy" in n: return "healthy"
    if "bacterial" in n and "spot" in n: return "bacterial_spot"
    if "early" in n and "blight" in n:   return "early_blight"
    if "late" in n and "blight" in n:    return "late_blight"
    if "leaf" in n and "mold" in n:      return "leaf_mold"
    if "septoria" in n:                  return "septoria_leaf_spot"
    if "spider" in n or "mite" in n:     return "spider_mites"
    if "target" in n and "spot" in n:    return "target_spot"
    if "yellow" in n and "curl" in n:    return "yellow_leaf_curl_virus"
    if "mosaic" in n:                    return "mosaic_virus"
    return n

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def resolve_cv_dir(src: Path, cv: int | None) -> Path:
    # Jeśli wskazano już fold (ma Train/Test), użyj go
    for cand in [src, src/"Train", src/"train"]:
        if (src/"Train").exists() and (src/"Test").exists():
            return src
    if (src/"Train").exists() and (src/"Test").exists():
        return src
    # Inaczej szukaj Cross-validationX
    if cv is None: cv = 1
    names = [f"Cross-validation{cv}", f"Cross_validation{cv}", f"CrossValidation{cv}"]
    for n in names:
        d = src / n
        if d.exists() and d.is_dir():
            return d
    # fallback: pierwszy pasujący
    for d in sorted(src.iterdir()):
        if d.is_dir() and re.search(r"cross[-_ ]?validation", d.name, re.I):
            return d
    raise FileNotFoundError(f"Nie znalazłem folderu Cross-validation w: {src}")

def list_tomato_cv(cv_dir: Path) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[str]]:
    # wykryj Train/Test (różne warianty wielkości liter)
    def pick(name):
        for n in [name, name.lower(), name.upper(), name.capitalize()]:
            d = cv_dir / n
            if d.exists() and d.is_dir(): return d
        return None

    d_train = pick("Train") or pick("training") or pick("train")
    d_test  = pick("Test")  or pick("testing")  or pick("test")
    if not d_train or not d_test:
        raise RuntimeError(f"W {cv_dir} nie widzę Train/Test. Struktura musi być: Cross-validationX/Train, Test.")

    cls_set = set()
    train_pairs: List[Tuple[Path, str]] = []
    test_pairs:  List[Tuple[Path, str]] = []

    # Train
    for cls_dir in sorted([d for d in d_train.iterdir() if d.is_dir()]):
        cname = _norm_tomato_cls(cls_dir.name)
        cls_set.add(cname)
        for p in cls_dir.rglob("*"):
            if is_image(p):
                train_pairs.append((p, cname))

    # Test
    for cls_dir in sorted([d for d in d_test.iterdir() if d.is_dir()]):
        cname = _norm_tomato_cls(cls_dir.name)
        cls_set.add(cname)
        for p in cls_dir.rglob("*"):
            if is_image(p):
                test_pairs.append((p, cname))

    return train_pairs, test_pairs, sorted(cls_set)

def split_train_val(train_pairs: List[Tuple[Path, str]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_cls: Dict[str, List[Path]] = {}
    for p, c in train_pairs:
        by_cls.setdefault(c, []).append(p)
    tr: List[Tuple[Path, str]] = []
    va: List[Tuple[Path, str]] = []
    for c, plist in by_cls.items():
        plist = sorted(plist); rng.shuffle(plist)
        n = len(plist); n_val = int(round(n * val_ratio))
        va += [(p, c) for p in plist[:n_val]]
        tr += [(p, c) for p in plist[n_val:]]
    return tr, va

def to_records(pairs: List[Tuple[Path, str]], classes: List[str], aug_mult: int, seed: int, split_name: str):
    rng = random.Random(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    recs: List[Tuple[np.ndarray, int]] = []
    pbar = tqdm(total=len(pairs) * (1 + max(0, aug_mult)), desc=f"build {split_name}", unit="img")
    for p, c in pairs:
        arr = load_img_square(p, IMG_SIZE)
        if arr is None:
            pbar.update(1)
            continue
        recs.append((arr, c2i[c])); pbar.update(1)
        for _ in range(max(0, aug_mult)):
            recs.append((aug_image(arr, rng), c2i[c])); pbar.update(1)
    pbar.close()
    return recs

def make_binary_pairs(pairs: List[Tuple[Path, str]]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    out = []
    for p, c in pairs:
        out.append((p, "healthy" if c == "healthy" else "diseased"))
    return out, ["healthy", "diseased"]

# ===== Main =====
def main():
    global IMG_SIZE  # ← to musi być pierwsze w funkcji zanim użyjesz IMG_SIZE

    ap = argparse.ArgumentParser(
        "PlantVillage Tomato (Cross-validation) → memmap tensors (multiclass + binary) + test_images copy"
    )
    ap.add_argument("--src", required=True, help="Root z Cross-validation* LUB bezpośrednio folder Cross-validationX")
    ap.add_argument("--out", required=True, help="Folder wyjściowy na packs i test_images")
    ap.add_argument("--cv", type=int, default=None, help="Numer folda (1..5), jeśli podajesz root z wieloma folderami")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)  # teraz legalnie użyte
    ap.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--aug-mult", type=int, default=AUG_MULT, help="Ile dodatkowych augmentów na obraz TRAIN (0=brak)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    IMG_SIZE = int(args.img_size)  # nadpisujemy globalną zgodnie z parametrem

    src = Path(args.src); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"cls").mkdir(parents=True, exist_ok=True)
    (out/"bin").mkdir(parents=True, exist_ok=True)

    test_img_root = out / "test_images"
    test_img_cls  = test_img_root / "cls"
    test_img_bin  = test_img_root / "bin"

    cv_dir = resolve_cv_dir(src, args.cv)
    print(f"Używam folda: {cv_dir}")

    banner("Skanuję PlantVillage Tomato")
    train_pairs_all, test_pairs_all, cls_classes = list_tomato_cv(cv_dir)
    print(f"Klasy ({len(cls_classes)}): {cls_classes}")
    print(f"Train IMG: {len(train_pairs_all)} | Test IMG: {len(test_pairs_all)}")

    tr, va = split_train_val(train_pairs_all, args.val_ratio, args.seed)

    
    # NO-LEAK: filtruj test względem train/val
    _dg = DedupGuard(log_csv=out/'_logs'/'dedup_tomato_cls.csv')
    _dg.add_pairs(tr + va)
    test_pairs_all, _ = _dg.filter_test(test_pairs_all)
    banner("CLS → train/val")
    write_pack(out/"cls"/"train", "train",
               to_records(tr, cls_classes, args.aug_mult, args.seed, "train(cls)"), cls_classes)
    write_pack(out/"cls"/"val", "val",
               to_records(va, cls_classes, 0, args.seed, "val(cls)"), cls_classes)

    banner("BIN → train/val")
    bin_tr_pairs, bin_classes = make_binary_pairs(tr)
    bin_va_pairs, _           = make_binary_pairs(va)
    write_pack(out/"bin"/"train", "train",
               to_records(bin_tr_pairs, bin_classes, args.aug_mult, args.seed, "train(bin)"), bin_classes)
    write_pack(out/"bin"/"val", "val",
               to_records(bin_va_pairs,  bin_classes, 0, args.seed, "val(bin)"),   bin_classes)

    banner("Kopiuję test_images (oryginały)")
    copy_test_images(test_pairs_all, test_img_cls, tag="cls")
    bin_test_pairs, _ = make_binary_pairs(test_pairs_all)
    copy_test_images(bin_test_pairs,  test_img_bin, tag="bin")

    cfg = {
        "img_size": IMG_SIZE,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "aug_mult_train": args.aug_mult,
        "outputs": ["cls", "bin"],
        "notes": [
            "train/val zbudowane WYŁĄCZNIE z 'Train' wybranego folda.",
            "Test kopiowany jako JPG do out/test_images/{cls,bin}.",
            "Klasy znormalizowane; 'BIN' = healthy vs diseased."
        ],
        "diagnostic": {
            "cv_dir": cv_dir.as_posix(),
            "num_train": len(train_pairs_all),
            "num_test": len(test_pairs_all),
            "classes": cls_classes
        }
    }
    (out / "builder_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    banner("GOTOWE ✅")
    print("CLS:", (out/'cls'/'train').as_posix(), (out/'cls'/'val').as_posix())
    print("BIN:", (out/'bin'/'train').as_posix(), (out/'bin'/'val').as_posix())
    print("IMG test:", (test_img_cls).as_posix(), (test_img_bin).as_posix())

if __name__ == "__main__":
    main()
