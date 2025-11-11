# build_rapeseed.py
# RZEPAK (CLS) → packs + test_images + NO-LEAK + PL meta
# Obsługa Hi/Lo: Alt_Hi + Alt_Lo → Alt; tylko Syl_* → healthy
# Dodatki:
#  - scan_debug.json w _logs/ z mapowaniem folder→klasa
#  - dedup testu (bytes + aHash64, próg Hamming = 3)
#  - packi: cls/train/s000, cls/val/s000, test_images/cls
#  - labels_pl.json z prostą kolejnością (healthy na końcu)

import argparse, json, random, re, io, csv, hashlib, shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageFile, ImageEnhance
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- stałe ----------
IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
IMG_SIZE = 380
VAL_RATIO = 0.15
TEST_RATIO = 0.10
SEED = 42
HASH_THRESHOLD = 3
AUG_MULT = 3

# tylko te prefiksy traktujemy jako “healthy”
HEALTHY_KEYS = {"syl", "healthy", "zdrowe", "zdrowy"}

# ---------- hash/dedup ----------
def _read_bytes_for_hash(p: Path):
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

def _sha1(p: Path):
    b = _read_bytes_for_hash(p)
    return hashlib.sha1(b).hexdigest() if b is not None else None

def _ahash64(p: Path, size: int = 8):
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32); m = a.mean()
            bits = (a >= m).astype("uint8").flatten()
            out = 0
            for b in bits: out = (out << 1) | int(b)
            return int(out)
    except Exception:
        return None

def _ham64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1<<64)-1)).count("1"))

class DedupGuard:
    def __init__(self, threshold: int = HASH_THRESHOLD, within_test: bool = True, log_csv: Optional[Path] = None):
        self.th = int(threshold); self.within = bool(within_test); self.log_csv = log_csv
        self.idx_sha = set(); self.idx_ph = []
    def add_pairs(self, pairs: List[Tuple[Path,str]]):
        for p,_ in pairs:
            p = Path(p)
            if not p.exists(): continue
            s = _sha1(p); h = _ahash64(p)
            if s: self.idx_sha.add(s)
            if h is not None: self.idx_ph.append(h)
    def filter_test(self, test_pairs: List[Tuple[Path,str]]):
        kept, removed = [], []
        for p, c in test_pairs:
            pp = Path(p)
            if not pp.exists(): removed.append((p,c,"missing")); continue
            s = _sha1(pp); h = _ahash64(pp)
            if s and s in self.idx_sha: removed.append((p,c,"bytes_equal(train/val)")); continue
            if h is not None and any(_ham64(h, r) <= self.th for r in self.idx_ph):
                removed.append((p,c,f"near_dup(train/val)_ham<={self.th}")); continue
            kept.append((p,c))
        if self.within and kept:
            seen_s, seen_h = set(), []
            out = []
            for p,c in kept:
                s = _sha1(Path(p)); h = _ahash64(Path(p))
                if s and s in seen_s: removed.append((p,c,"dup_within_test_bytes")); continue
                if h is not None and any(_ham64(h, r) <= self.th for r in seen_h):
                    removed.append((p,c,f"near_dup_within_test_ham<={self.th}")); continue
                if s: seen_s.add(s)
                if h is not None: seen_h.append(h)
                out.append((p,c))
            kept = out
        if self.log_csv:
            self.log_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["path","class","action","reason"])
                for p,c in kept: w.writerow([str(p), c, "keep", ""])
                for p,c,r in removed: w.writerow([str(p), c, "drop", r])
        return kept, removed

# ---------- IO/pack ----------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def load_img_square(path: Path, size: int) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w,h = im.size
            if w < h: nw,nh=size,int(round(h*size/w))
            else:     nh,nw=size,int(round(w*size/h))
            im = im.resize((nw,nh), Image.BICUBIC)
            l=(nw-size)//2; t=(nh-size)//2
            im = im.crop((l,t,l+size,t+size))
            arr = np.asarray(im, dtype=np.uint8)
            return np.transpose(arr,(2,0,1))  # CHW
    except Exception:
        return None

def to_records(pairs: List[Tuple[Path,str]], classes: List[str], aug_mult: int, seed: int, split_name: str):
    rng = random.Random(seed); c2i = {c:i for i,c in enumerate(classes)}; out = []
    pbar = tqdm(total=len(pairs)*(1+max(0,aug_mult)), desc=f"build {split_name}", unit="img")
    for p,c in pairs:
        x = load_img_square(p, IMG_SIZE)
        if x is None: pbar.update(1); continue
        y = c2i[c]; out.append((x,y)); pbar.update(1)
        for _ in range(max(0,aug_mult)):
            img = Image.fromarray(np.transpose(x,(1,2,0)))
            if rng.random()<0.6: img = img.rotate(rng.uniform(-10,10), resample=Image.BICUBIC, expand=False, fillcolor=(0,0,0))
            if rng.random()<0.5: img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9,1.1))
            if rng.random()<0.5: img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9,1.1))
            if rng.random()<0.5: img = ImageEnhance.Color(img).enhance(rng.uniform(0.9,1.1))
            aug = np.asarray(img, dtype=np.uint8)
            aug = np.transpose(aug,(2,0,1))
            out.append((aug,y)); pbar.update(1)
    pbar.close(); return out

def write_pack(out_dir: Path, split: str, records, classes: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"[{split}] brak próbek → pomijam"); return
    C,H,W = records[0][0].shape; N = len(records)
    shard = out_dir/"s000"; shard.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(shard/"images.dat", dtype=np.uint8, mode="w+", shape=(N,C,H,W))
    labels = np.empty((N,), dtype=np.int64)
    sum_c = np.zeros(3, dtype=np.float64); sumsq_c = np.zeros(3, dtype=np.float64)
    for i,(x,y) in enumerate(tqdm(records, desc=f"pack {out_dir.parent.name}/{split}", unit="img")):
        mm[i]=x; labels[i]=y
        xf = x.astype(np.float32)/255.0
        sum_c += xf.reshape(3,-1).sum(1)
        sumsq_c += (xf**2).reshape(3,-1).sum(1)
    mm.flush(); np.save(shard/"index.npy", labels)
    total_pix = N*H*W
    mean = (sum_c/total_pix).tolist()
    var  = (sumsq_c/total_pix) - np.square(sum_c/total_pix)
    std  = np.sqrt(np.clip(var, 1e-12, None)).tolist()
    (shard/"meta.json").write_text(json.dumps({
        "classes":classes,"num_samples":int(N),"shape":[3,H,W],"dtype":"uint8",
        "labels_file":"index.npy","data_file":"images.dat","mean":mean,"std":std,
        "split":split,"shard":0
    }, indent=2, ensure_ascii=False), encoding="utf-8")

def copy_test_images(pairs: List[Tuple[Path,str]], dest_root: Path):
    if not pairs:
        print("[test→img] brak próbek"); return
    for p,c in tqdm(pairs, desc="copy test→img", unit="img"):
        cls_dir = dest_root/c; cls_dir.mkdir(parents=True, exist_ok=True)
        target = cls_dir/p.name
        if target.exists():
            k=1
            while (cls_dir/f"{p.stem}_{k}{p.suffix}").exists(): k+=1
            target = cls_dir/f"{p.stem}_{k}{p.suffix}"
        try: shutil.copy2(p, target)
        except Exception as e: print(f"[copy] {p} → {target}: {e}")

def stratified_split(pairs: List[Tuple[Path,str]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    by: Dict[str, List[Path]] = {}
    for p,c in pairs: by.setdefault(c,[]).append(p)
    train,val,test=[],[],[]
    for c,plist in by.items():
        plist = plist[:]; rng.shuffle(plist)
        n = len(plist)
        n_test = int(round(n*test_ratio))
        n_val  = int(round((n-n_test)*val_ratio))
        if n_test+n_val >= n:
            n_test = max(0, min(n_test, n-1))
            n_val  = max(0, min(n_val,  n-1-n_test))
        test += [(p,c) for p in plist[:n_test]]
        val  += [(p,c) for p in plist[n_test:n_test+n_val]]
        train+= [(p,c) for p in plist[n_test+n_val:]]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train,val,test

# ---------- meta do UI ----------
PL_META_BASE = {"translation": {"healthy":"zdrowa"}, "display_order": []}

def write_ui_meta(out_dir: Path, classes: List[str]):
    meta_dir = out_dir/"meta"; meta_dir.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in classes if c != "healthy"] + (["healthy"] if "healthy" in classes else [])
    data = dict(PL_META_BASE)
    tr = dict(PL_META_BASE["translation"])
    for c in classes:
        tr.setdefault(c, c)
    data["translation"] = tr
    data["display_order"] = ordered
    (meta_dir/"labels_pl.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------- skanery ----------
HILO_RE = re.compile(r"^\s*([A-Za-z0-9]+)_(Hi|Lo)\s*$", re.IGNORECASE)

def _canon_from_base(base: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "", base).lower()
    return "healthy" if key in HEALTHY_KEYS else base

def looks_like_hilo(src: Path) -> bool:
    for d in src.iterdir():
        if d.is_dir() and HILO_RE.match(d.name):
            return True
    return False

def scan_hilo(src: Path):
    """Zwraca (pairs, debug) gdzie pairs = [(path, klasa)], debug = lista wpisów z licznikiem."""
    pairs: List[Tuple[Path,str]] = []
    debug_rows = []
    for d in sorted([x for x in src.iterdir() if x.is_dir()]):
        m = HILO_RE.match(d.name)
        if not m: continue
        base = m.group(1).strip()
        canonical = _canon_from_base(base)
        cnt = 0
        for p in d.rglob("*"):
            if is_image(p):
                pairs.append((p, canonical)); cnt += 1
        debug_rows.append({"folder": d.name, "base": base, "canonical": canonical, "count": cnt})
    return pairs, debug_rows

def scan_legacy(src: Path):
    roots = []
    for name in ["Mono","MuRo","Vir","Ins","Ntr","healthy"]:
        d = src/name
        if d.exists() and d.is_dir(): roots.append(d)
    pairs=[]; debug_rows=[]
    for r in roots:
        for cdir in [d for d in r.iterdir() if d.is_dir()]:
            cname = cdir.name.strip()
            canonical = _canon_from_base(cname)
            cnt=0
            for p in cdir.rglob("*"):
                if is_image(p): pairs.append((p, canonical)); cnt+=1
            debug_rows.append({"folder": f"{r.name}/{cdir.name}", "base": cname, "canonical": canonical, "count": cnt})
    return pairs, debug_rows

# ---------- main ----------
def main():
    global IMG_SIZE
    ap = argparse.ArgumentParser("Rapeseed builder (Hi/Lo merge; Syl_* → healthy)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--aug_mult", type=int, default=AUG_MULT)
    ap.add_argument("--hash_threshold", type=int, default=HASH_THRESHOLD)
    args = ap.parse_args()

    IMG_SIZE = int(args.img_size)
    src = Path(args.src); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out/"_logs"; log_dir.mkdir(parents=True, exist_ok=True)

    # skan
    if looks_like_hilo(src):
        print("[scan] Wykryto Hi/Lo. Łączenie par; Syl_* → healthy.")
        cls_pairs, scan_debug = scan_hilo(src)
        schema = "hilo"
    else:
        print("[scan] Wykryto legacy (Mono/MuRo/...).")
        cls_pairs, scan_debug = scan_legacy(src)
        schema = "legacy"

    # zapis debug
    (log_dir/"scan_debug.json").write_text(json.dumps(scan_debug, indent=2, ensure_ascii=False), encoding="utf-8")

    if not cls_pairs:
        raise SystemExit("Brak obrazów w źródle. Pustką modelu nie nakarmisz.")

    # klasy (healthy zawsze na końcu)
    uniq = sorted({c for _,c in cls_pairs if c != "healthy"})
    if any(c == "healthy" for _,c in cls_pairs):
        uniq.append("healthy")
    classes = uniq[:]

    # sanity
    counts: Dict[str,int] = {}
    for _,c in cls_pairs: counts[c] = counts.get(c,0) + 1
    print("[scan] Klasy:")
    for c in classes:
        print(f"   {c:>12}: {counts.get(c,0)}")
    if len(classes) == 1 and classes[0] == "healthy":
        print("\n[UWAGA] Wyszła tylko klasa 'healthy'. Sprawdź scan_debug.json. "
              "To zwykle oznacza, że wszystkie foldery miały prefiks Syl_* albo ścieżkę źródła wskazałeś nie tam, gdzie trzeba.")

    # split + dedup testu
    tr, va, te = stratified_split(cls_pairs, args.val_ratio, args.test_ratio, args.seed)
    dg = DedupGuard(threshold=args.hash_threshold, within_test=True, log_csv=(log_dir/"dedup_test.csv"))
    dg.add_pairs(tr + va)
    te, removed = dg.filter_test(te)
    if removed:
        print(f"[dedup] Z testu usunięto {len(removed)} duplikatów/podobnych.")

    # budowa
    write_pack(out/"cls"/"train","train", to_records(tr, classes, args.aug_mult, args.seed, "train"), classes)
    write_pack(out/"cls"/"val",  "val",   to_records(va, classes, 0,             args.seed, "val"),   classes)
    copy_test_images(te, out/"test_images"/"cls")
    write_ui_meta(out, classes)

    # config
    (out/"builder_config.json").write_text(json.dumps({
        "dataset":"rapeseed", "schema": schema,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
        "img_size": IMG_SIZE, "seed": args.seed,
        "aug_mult_train": args.aug_mult, "hash_threshold": args.hash_threshold,
        "classes": classes
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GOTOWE ✅", out.as_posix())

if __name__ == "__main__":
    main()
