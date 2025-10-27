import argparse, json, random, time, re, shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


HASH_THRESHOLD = 2  # 0 identyczne, 2 rozsądnie, 4 agresywnie

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





IMG_SIZE   = 256
VAL_RATIO  = 0.15
TEST_RATIO = 0.10
SEED       = 42
AUG_MULT   = 0
IMG_EXTS   = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ===== Utils =====
def banner(txt): print("\\n" + "="*8 + " " + txt + " " + "="*8)

def load_img_square(path: Path, size: int) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < h:  nw, nh = size, int(round(h * size / w))
            else:      nh, nw = size, int(round(w * size / h))
            im = im.resize((nw, nh), Image.BICUBIC)
            l = (nw - size) // 2; t = (nh - size) // 2
            im = im.crop((l, t, l + size, t + size))
            arr = np.asarray(im, dtype=np.uint8)  # HWC
            return np.transpose(arr, (2,0,1))     # CHW
    except Exception:
        return None

def aug_image(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(np.transpose(arr, (1,2,0)))
    #if rng.random() < 0.5: img = ImageOps.mirror(img)
    #if rng.random() < 0.2: img = ImageOps.flip(img)
    if rng.random() < 0.6:
        img = img.rotate(rng.uniform(-10,10), resample=Image.BICUBIC, expand=False, fillcolor=(0,0,0))
    if rng.random() < 0.6: img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9,1.1))
    if rng.random() < 0.6: img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9,1.1))
    if rng.random() < 0.6: img = ImageEnhance.Color(img).enhance(rng.uniform(0.9,1.1))
    out = np.asarray(img, dtype=np.uint8)
    return np.transpose(out, (2,0,1))

def write_pack(out_dir: Path, split: str, records: List[Tuple[np.ndarray,int]], classes: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"[{split}] brak próbek, pomijam")
        return
    C,H,W = records[0][0].shape
    N = len(records)
    part = out_dir / "s000"
    part.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(part/"images.dat", dtype=np.uint8, mode="w+", shape=(N,C,H,W))
    labels = np.empty((N,), dtype=np.int64)

    sum_c = np.zeros(3); sumsq_c = np.zeros(3)
    for i,(x,y) in enumerate(tqdm(records, desc=f"pack {split}", unit="img")):
        mm[i] = x; labels[i] = y
        xf = x.astype(np.float32)/255.0
        sum_c += xf.reshape(3,-1).mean(1)
        sumsq_c += (xf.reshape(3,-1)**2).mean(1)
    mm.flush(); np.save(part/"index.npy", labels)

    mean = (sum_c/N)
    var  = (sumsq_c/N) - mean**2
    std  = np.sqrt(np.maximum(var, 1e-6))
    (part/"meta.json").write_text(json.dumps({
        "classes": classes,
        "num_samples": int(N),
        "shape": [int(C),int(H),int(W)],
        "dtype": "uint8",
        "labels_file": "index.npy",
        "data_file": "images.dat",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "split": split,
        "shard": 0
    }, indent=2, ensure_ascii=False), encoding="utf-8")

def copy_test_images(pairs: List[Tuple[Path,str]], dest_root: Path, tag: str):
    if not pairs:
        print(f"[test→img/{tag}] brak próbek, pomijam")
        return
    for p, c in tqdm(pairs, desc=f"copy test→img/{tag}", unit="img"):
        cls_dir = dest_root / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        target = cls_dir / p.name
        if target.exists():
            stem, suf = p.stem, p.suffix
            k = 1
            while (cls_dir / f"{stem}_{k}{suf}").exists():
                k += 1
            target = cls_dir / f"{stem}_{k}{suf}"
        try:
            shutil.copy2(p, target)
        except Exception as e:
            print(f"[copy] nie udało się skopiować {p} → {target}: {e}")

# ====== POTATO listing ======
def _norm_cls(name: str) -> str:
    n = name.lower()
    if re.search(r"healthy|control|ok|none|normal", n):
        return "healthy"
    if re.search(r"early.?bl(ight|t)", n) or n in {"earlyblt","earlyblight","eb"}:
        return "earlyblight"
    if re.search(r"late.?bl(ight|t)", n) or n in {"lateblt","lateblight","lb"}:
        return "lateblight"
    return name  # cokolwiek innego zostaw w spokoju

def list_potato(src: Path) -> Tuple[List[Tuple[Path,str]], List[str]]:
    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    pairs = []
    normed = set()
    for c in tqdm(classes, desc="scan classes", unit="cls"):
        c_norm = _norm_cls(c)
        normed.add(c_norm)
        for p in (src/c).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                pairs.append((p, c_norm))
    classes_out = sorted(normed)
    return pairs, classes_out

# ===== Splitting =====
def simple_split(items: List[Tuple[Path,str]], val_ratio: float, test_ratio: float, seed:int):
    rng=random.Random(seed)
    by_class: Dict[str, List[Path]] = {}
    for p,c in items: by_class.setdefault(c,[]).append(p)
    train=[]; val=[]; test=[]
    for c,plist in by_class.items():
        plist = sorted(plist); rng.shuffle(plist)
        n=len(plist); n_test=int(round(n*test_ratio)); n_val=int(round(n*val_ratio))
        test += [(p,c) for p in plist[:n_test]]
        val  += [(p,c) for p in plist[n_test:n_test+n_val]]
        train+= [(p,c) for p in plist[n_test+n_val:]]
    return train,val,test

def build_records(pairs: List[Tuple[Path,str]], classes: List[str], aug_mult:int, seed:int, split_name:str):
    rng = random.Random(seed)
    c2i = {c:i for i,c in enumerate(classes)}
    total = len(pairs) * (1 + max(0, aug_mult))
    recs=[]; skipped=[]
    pbar = tqdm(total=total, desc=f"build {split_name}", unit="img")
    for p,c in pairs:
        arr = load_img_square(p, IMG_SIZE)
        if arr is None:
            skipped.append(str(p))
            pbar.update(1)
            continue
        recs.append((arr, c2i[c])); pbar.update(1)
        for _ in range(aug_mult):
            recs.append((aug_image(arr, rng), c2i[c])); pbar.update(1)
    pbar.close()
    if skipped:
        log = Path("skipped_" + split_name.replace("/", "_") + ".txt")
        try:
            log.write_text("\\n".join(skipped), encoding="utf-8")
            print(f"[{split_name}] SKIPPED: {len(skipped)} plików. Lista → {log}")
        except Exception:
            print(f"[{split_name}] SKIPPED: {len(skipped)} plików (nie udało się zapisać logu).")
    return recs

def make_binary_from_pairs(pairs: List[Tuple[Path,str]]) -> Tuple[List[Tuple[Path,str]], List[str]]:
    out=[]
    for p,c in pairs:
        out.append((p, "healthy" if c == "healthy" else "diseased"))
    return out, ["healthy","diseased"]

# ===== Main =====
def main():
    ap = argparse.ArgumentParser("POTATO → memmap tensors (multiclass + binary) + kopia testowych IMG")
    ap.add_argument("--src", required=True, help="Folder z klasami (healthy, earlyblt, lateblt itp.)")
    ap.add_argument("--out", required=True, help="Folder wyjściowy na packs i test_images")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    (out/"cls").mkdir(parents=True, exist_ok=True)
    (out/"bin").mkdir(parents=True, exist_ok=True)

    # Foldery na kopie testowych obrazów
    test_img_root = out/"test_images"
    test_img_cls  = test_img_root/"cls"
    test_img_bin  = test_img_root/"bin"

    banner("POTATO: skan i split")
    items, cls_classes = list_potato(src)
    tr,va,te = simple_split(items, VAL_RATIO, TEST_RATIO, SEED)

    
    # NO-LEAK: filtruj test względem train/val
    _dg = DedupGuard(log_csv=out/'_logs'/'dedup_potato_cls.csv')
    _dg.add_pairs(tr + va)
    te, _ = _dg.filter_test(te)
# ====== CLS ======
    banner("CLS → train/val[/test]")
    write_pack(out/"cls"/"train","train", build_records(tr, cls_classes, AUG_MULT, SEED, "train(cls)"), cls_classes)
    write_pack(out/"cls"/"val",  "val",   build_records(va, cls_classes, 0,       SEED, "val(cls)"),   cls_classes)
    if te:
        write_pack(out/"cls"/"test","test", build_records(te, cls_classes, 0,     SEED, "test(cls)"),  cls_classes)

    # ====== BIN ======
    banner("BIN → train/val[/test]")
    bin_pairs_all, bin_classes = make_binary_from_pairs(items)
    trb, vab, teb = simple_split(bin_pairs_all, VAL_RATIO, TEST_RATIO, SEED)
    
    # NO-LEAK: filtruj test bin względem train/val
    _dgb = DedupGuard(log_csv=out/'_logs'/'dedup_potato_bin.csv')
    _dgb.add_pairs(trb + vab)
    teb, _ = _dgb.filter_test(teb)
    write_pack(out/"bin"/"train","train", build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)"), bin_classes)
    write_pack(out/"bin"/"val",  "val",   build_records(vab, bin_classes, 0,       SEED, "val(bin)"),   bin_classes)
    if teb:
        write_pack(out/"bin"/"test","test", build_records(teb, bin_classes, 0,     SEED, "test(bin)"),  bin_classes)

    # ====== Kopie IMG z TESt splitu (żeby ewaluować bez dotykania train/val) ======
    banner("Kopiuję test_images (oryginały)")
    copy_test_images(te,  test_img_cls, tag="cls")
    copy_test_images(teb, test_img_bin, tag="bin")

    # ====== Zapis konfiguracji ======
    (out/"builder_config.json").write_text(json.dumps({
        "img_size": IMG_SIZE, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO,
        "seed": SEED, "aug_mult_train": AUG_MULT,
        "outputs": ["cls","bin"],
        "note": "train/val zbudowane TYLKO z własnych splitów; test_images to surowe kopie z podziału test."
    }, indent=2), encoding="utf-8")

    mins = (time.time()-time.time())/60  # nie śledzimy dokładnie czasu tutaj
    banner("GOTOWE")
    print(f"CLS packs: {(out/'cls'/'train').as_posix()}, {(out/'cls'/'val').as_posix()}, {(out/'cls'/'test').as_posix() if (out/'cls'/'test').exists() else '—'}")
    print(f"BIN packs: {(out/'bin'/'train').as_posix()}, {(out/'bin'/'val').as_posix()}, {(out/'bin'/'test').as_posix() if (out/'bin'/'test').exists() else '—'}")
    print(f"IMG test:  {test_img_cls.as_posix()}, {test_img_bin.as_posix()}")

if __name__ == "__main__":
    main()
