import argparse, json, random, io, csv, hashlib, re, sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageFile, ImageEnhance
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== Konfig domyślny =====
IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
IMG_SIZE = 380
VAL_RATIO = 0.15
TEST_RATIO = 0.10
SEED = 42
AUG_MULT = 0         # offline augment WYŁĄCZONY
HASH_THRESHOLD = 2   # 0 identyczne, 2 rozsądnie, 4 agresywnie

# ===== NO-LEAK GUARD (bytes + aHash) =====
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
# ===== /NO-LEAK =====

# ===== Mapowania etykiet =====
CLASSES_A7 = ["Healthy","LeafBlight","Rust","PowderyMildew","FusariumFootRot","WheatBlast","BlackPoint"]
CLASSES_B5 = ["HealthyLeaf","LeafBlight","FusariumFootRot","WheatBlast","BlackPoint"]

MAP_WFD_TO_A7 = {
    "healthy": "Healthy",
    "leaf_rust": "Rust",
    "stem_rust": "Rust",
    "yellow_rust": "Rust",
    "powdery_mildew": "PowderyMildew",
    "septoria": "LeafBlight",
    "seedlings": None,  # drop
}
MAP_FOLDER_TO_A7 = {
    "HealthyLeaf": "Healthy",
    "LeafBlight": "LeafBlight",
    "FusariumFootRot": "FusariumFootRot",
    "WheatBlast": "WheatBlast",
    "BlackPoint": "BlackPoint",
}
MAP_WFD_TO_B5 = {
    "healthy": "HealthyLeaf",
    "leaf_rust": "LeafBlight",
    "stem_rust": "LeafBlight",
    "yellow_rust": "LeafBlight",
    "powdery_mildew": "LeafBlight",
    "septoria": "LeafBlight",
    "seedlings": None,
}
MAP_FOLDER_TO_B5 = {
    "HealthyLeaf": "HealthyLeaf",
    "LeafBlight": "LeafBlight",
    "FusariumFootRot": "FusariumFootRot",
    "WheatBlast": "WheatBlast",
    "BlackPoint": "BlackPoint",
}
WFD_ONEHOT_COLS = {"healthy","leaf_rust","powdery_mildew","seedlings","septoria","stem_rust","yellow_rust"}

def canon_cls_label(label: str, schema: str) -> str:
    if schema == "B5":
        return "HealthyLeaf" if label.lower() in {"healthy","healthyleaf"} else label
    return "Healthy" if label.lower() in {"healthy","healthyleaf"} else label

def map_label(schema: str, source: str, label: str) -> Optional[str]:
    l = label.strip()
    if source == "WFD":
        return (MAP_WFD_TO_A7 if schema=="A7" else MAP_WFD_TO_B5).get(l, None)
    else:
        return (MAP_FOLDER_TO_A7 if schema=="A7" else MAP_FOLDER_TO_B5).get(l, None)

# ===== Utils IO / obrazy =====
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_pairs_from_folder_tree(root: Path) -> List[Tuple[Path,str]]:
    pairs=[]
    for cdir in [d for d in root.iterdir() if d.is_dir()]:
        cname=cdir.name.strip()
        for p in cdir.rglob("*"):
            if is_image(p): pairs.append((p,cname))
    if not pairs:
        raise SystemExit(f"Brak obrazów w strukturze folderowej: {root}")
    return pairs

# Multi-key index dla dopasowania nazw z CSV do plików
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+","", s.strip().lower())

def build_multikey_index(root: Path):
    idx = {"exact":defaultdict(list), "lc":defaultdict(list), "stem":defaultdict(list), "stemlc":defaultdict(list), "norm":defaultdict(list)}
    for p in root.rglob("*"):
        if not is_image(p): continue
        b, s = p.name, p.stem
        idx["exact"][b].append(p)
        idx["lc"][b.lower()].append(p)
        idx["stem"][s].append(p)
        idx["stemlc"][s.lower()].append(p)
        idx["norm"][_norm(b)].append(p); idx["norm"][_norm(s)].append(p)
    return idx

def _candidate_bases(raw: str) -> List[str]:
    r = raw.strip().strip('"').strip("'").replace("\\","/")
    base = Path(r).name
    out = [base]
    if "_" in base:
        out.insert(0, base.rsplit("_",1)[-1])
        out.append(base.split("_",1)[1])
    if "-" in base:
        out.append(base.split("-",1)[1])
    m = re.search(r"([0-9a-fA-F]{12,})$", Path(base).stem)
    if m: out += [m.group(1)+Path(base).suffix, m.group(1)]
    seen=set(); uniq=[]
    for b in out:
        if b and b not in seen: uniq.append(b); seen.add(b)
    return uniq

def find_in_index(raw: str, root: Path, index) -> Optional[Path]:
    p = (root / raw)
    if p.exists() and is_image(p): return p
    for b in _candidate_bases(raw):
        stem = Path(b).stem
        for bucket, key in [("exact",b),("lc",b.lower()),("stem",stem),("stemlc",stem.lower()),("norm",_norm(b)),("norm",_norm(stem))]:
            lst = index[bucket].get(key)
            if lst: return lst[0]
    return None

# ===== CSV wczytanie z autodetekcją =====
def read_csv_smart(csv_path: Path):
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    delim = "," if txt.count(",") >= txt.count(";") else ";"
    reader = csv.DictReader(txt.splitlines(), delimiter=delim)
    if not reader.fieldnames:
        raise SystemExit(f"CSV bez nagłówka: {csv_path}")
    return list(reader), reader.fieldnames, delim

def detect_col(header: List[str], prefer: List[str], fallback: List[str]) -> Optional[str]:
    H = [h.strip().lower() for h in header]
    for w in prefer:
        if w.lower() in H: return header[H.index(w.lower())]
    for h in header:
        if any(tok in h.lower() for tok in fallback): return h
    return None

def _truthy(x) -> Optional[float]:
    s = str(x).strip().lower()
    if s == "": return None
    if s in {"1","true","t","yes","y","x","✓"}: return 1.0
    if s in {"0","false","f","no","n"}: return 0.0
    try: return float(s.replace(",","."))  # number-ish
    except Exception: return 1.0

def read_pairs_from_csv(csv_path: Path, root: Path, img_col: Optional[str], cls_col: Optional[str], diag: Dict):
    rows, header, delim = read_csv_smart(csv_path)
    img_c = img_col or detect_col(header, ["image","filename","file","img","path","image_name","name","image_id"], ["file","image","img","path","name","id"])
    cls_c = cls_col or detect_col(header, ["label","class","category","disease","target"], ["label","class","category","disease","target"])
    wide_cols = [] if cls_c else [c for c in header if c.strip().lower() in WFD_ONEHOT_COLS]
    wide = bool(wide_cols)

    if img_c is None or (cls_c is None and not wide):
        raise SystemExit(f"Nie wykryto kolumn (image/label). Nagłówki: {header}")

    index = build_multikey_index(root)
    pairs=[]; unmatched=[]; multi=0; zero=0
    for row in rows:
        raw_name = str(row.get(img_c,"")).strip()
        if not raw_name: continue

        if not wide:
            lab = str(row.get(cls_c,"")).strip()
            if not lab: continue
        else:
            scores=[]
            for c in wide_cols:
                v = _truthy(row.get(c,""))
                if v is not None and v>0: scores.append((c,float(v)))
            if not scores: zero+=1; continue
            sick = [(c,s) for c,s in scores if c.strip().lower()!="healthy"]
            lab = max(sick or scores, key=lambda x:x[1])[0]
            if len(sick or scores) > 1: multi+=1

        p = find_in_index(raw_name, root, index)
        if p is None: unmatched.append(raw_name); continue
        pairs.append((p, lab))

    diag["csv"] = {
        "delimiter": delim, "header": header,
        "detected_img_col": img_c, "detected_cls_col": None if wide else cls_c,
        "wide_mode": wide, "wide_cols": wide_cols,
        "rows": len(rows), "matched": len(pairs),
        "unmatched": len(unmatched), "unmatched_sample": list(dict.fromkeys(unmatched))[:50],
        "wide_multi_hits": multi, "wide_zero_hits": zero
    }
    return pairs

# ===== Augmenty / pack =====
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
            arr = np.asarray(im, dtype=np.uint8)  # HWC
            return np.transpose(arr,(2,0,1))      # CHW
    except Exception:
        return None

def to_records(pairs: List[Tuple[Path,str]], classes: List[str], aug_mult: int, seed: int, split_name: str):
    rng = random.Random(seed); c2i = {c:i for i,c in enumerate(classes)}; out = []
    pbar = tqdm(total=len(pairs)*(1+max(0,aug_mult)), desc=f"build {split_name}", unit="img")
    for p,c in pairs:
        arr = load_img_square(p, IMG_SIZE)
        if arr is None: pbar.update(1); continue
        y = c2i[c]; out.append((arr,y)); pbar.update(1)
        # aug_mult będzie 0 dla train → pętla nie ruszy
        for _ in range(max(0,aug_mult)):
            img = Image.fromarray(np.transpose(arr,(1,2,0)))
            if rng.random()<0.6: img = img.rotate(rng.uniform(-10,10), resample=Image.BICUBIC, expand=False, fillcolor=(0,0,0))
            if rng.random()<0.5: img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9,1.1))
            if rng.random()<0.5: img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9,1.1))
            if rng.random()<0.5: img = ImageEnhance.Color(img).enhance(rng.uniform(0.9,1.1))
            aug = np.asarray(img, dtype=np.uint8); aug = np.transpose(aug,(2,0,1))
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

def export_jpg_tree(pairs: List[Tuple[Path,str]], dest_root: Path, quality: int = 95):
    for p,c in tqdm(pairs, desc="copy test→img", unit="img"):
        cls_dir = dest_root/c; cls_dir.mkdir(parents=True, exist_ok=True)
        target = cls_dir/p.name
        if target.exists():
            k=1
            while (cls_dir/f"{p.stem}_{k}{p.suffix}").exists(): k+=1
            target = cls_dir/f"{p.stem}_{k}{p.suffix}"
        try:
            with Image.open(p) as im:
                if im.mode in ("RGBA","LA"):
                    bg = Image.new("RGB", im.size, (255,255,255))
                    bg.paste(im.convert("RGBA"), mask=im.split()[-1]); im = bg
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(target, format="JPEG", quality=quality, optimize=True)
        except Exception as e:
            print(f"[copy] {p} → {target}: {e}")

# ===== Split =====
def stratified_split(pairs: List[Tuple[Path,str]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    by: Dict[str, List[Path]] = defaultdict(list)
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

# ===== Main =====
def main():
    global IMG_SIZE
    ap = argparse.ArgumentParser("PSZENICA → merge folder + WFD CSV → packs + test JPG (NO-LEAK)")
    ap.add_argument("--src_a", required=True, help="Folder-tree dataset (klasa = katalog)")
    ap.add_argument("--src_b", required=True, help="WFD root (obrazy)")
    ap.add_argument("--csv",    required=True, help="WFD CSV (single-col label lub one-hot)")
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--cls_col", default=None)
    ap.add_argument("--out", required=True)

    # aliasy myślnik/podkreślenie
    ap.add_argument("--img-size","--img_size", dest="img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-ratio","--val_ratio", dest="val_ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--test-ratio","--test_ratio", dest="test_ratio", type=float, default=TEST_RATIO)
    ap.add_argument("--aug-mult","--aug_mult", dest="aug_mult", type=int, default=AUG_MULT)  # ignorowane dla train
    ap.add_argument("--hash-threshold","--hash_threshold", dest="hash_threshold", type=int, default=HASH_THRESHOLD)
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--schema", choices=["A7","B5"], default="A7")
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args()

    IMG_SIZE = int(args.img_size)
    random.seed(args.seed); np.random.seed(args.seed)

    # 1) wczytaj pary
    pairs_folder_raw = list_pairs_from_folder_tree(Path(args.src_a))
    diag: Dict = {}
    pairs_wfd_raw    = read_pairs_from_csv(Path(args.csv), Path(args.src_b), args.img_col, args.cls_col, diag)

    # 2) mapowanie do klas i kanonizacja
    pairs_folder=[]
    for p,lab in pairs_folder_raw:
        m = map_label(args.schema, "FOLDER", lab)
        if m is not None: pairs_folder.append((p, canon_cls_label(m, args.schema)))
    pairs_wfd=[]
    for p,lab in pairs_wfd_raw:
        m = map_label(args.schema, "WFD", lab)
        if m is not None: pairs_wfd.append((p, canon_cls_label(m, args.schema)))

    # 3) merge + sanity
    all_pairs = list(dict.fromkeys(pairs_folder + pairs_wfd))
    classes = (CLASSES_A7 if args.schema=="A7" else CLASSES_B5)
    bad = {c for _,c in all_pairs if c not in classes}
    if bad:
        raise SystemExit(f"Nieznane etykiety: {sorted(bad)}; oczekiwane {classes}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    if args.dryrun or not all_pairs:
        (out_root/"diagnostic.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
        sample = diag.get("csv",{}).get("unmatched_sample",[])
        if sample:
            (out_root/"unmatched_sample.csv").write_text("raw_name\n"+"\n".join(sample), encoding="utf-8")
        if not all_pairs:
            raise SystemExit("0 par po dopasowaniu — patrz diagnostic.json.")
        print(f"[DRYRUN] Diagnoza w {out_root}"); return

    # 4) split + NO-LEAK na teście
    tr,va,te = stratified_split(all_pairs, args.val_ratio, args.test_ratio, args.seed)
    (out_root/"_logs").mkdir(parents=True, exist_ok=True)
    dg = DedupGuard(threshold=args.hash_threshold, within_test=True, log_csv=out_root/"_logs"/"dedup_wheat_cls.csv")
    dg.add_pairs(tr + va); te, _ = dg.filter_test(te)
    print(f"Rozmiary: train={len(tr)}  val={len(va)}  test={len(te)}")

    # 5) packs – train BEZ offline augmentu
    (out_root/"cls").mkdir(parents=True, exist_ok=True)
    write_pack(out_root/"cls"/"train", "train", to_records(tr, classes, 0,           args.seed, "train(cls)"), classes)
    write_pack(out_root/"cls"/"val",   "val",   to_records(va, classes, 0,           args.seed, "val(cls)"),   classes)

    # 6) test JPG
    if te:
        export_jpg_tree(te, out_root/"test_images"/"cls", quality=args.jpg_quality)

    # 7) meta
    (out_root/"meta").mkdir(parents=True, exist_ok=True)
    (out_root/"meta"/"labels_pl.json").write_text(json.dumps({
        "translation": {c:c for c in classes},
        "display_order": classes
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_root/"builder_config.json").write_text(json.dumps({
        "dataset":"wheat","schema": args.schema,"classes": classes,"img_size": IMG_SIZE,
        "seed": args.seed,"val_ratio": args.val_ratio,"test_ratio": args.test_ratio,
        "aug_mult": 0, "hash_threshold": args.hash_threshold,
        "notes":[
            "CSV: single-col albo one-hot. Dopasowanie nazw: ostatni '_', stem, znormalizowane.",
            "NO-LEAK: bytes+phash z progowaniem i odszumieniem w teście.",
            "Brak augmentacji offline; augmentacje tylko w trakcie treningu."
        ],
        "diagnostic": diag
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("GOTOWE ✅", out_root.as_posix())

if __name__ == "__main__":
    main()
