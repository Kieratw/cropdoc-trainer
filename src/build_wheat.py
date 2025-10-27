
# === NO-LEAK GUARD (bytes + aHash) ===
HASH_THRESHOLD = 2  # 0 identyczne, 2 rozsądnie, 4 agresywnie
from pathlib import Path
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

# -*- coding: utf-8 -*-
# build_wheat.py — scala zbiór folderowy + WFD (CSV) do wspólnych klas,
# zapisuje packs (CLS/BIN: train/val) i test JPG (bez przecieku).
# Wspiera schematy: A7 (7 klas) oraz B5 (5 klas), seedlings = DROP.
#
# Obsługa CSV:
#  - tryb "klasy w jednej kolumnie" (label/class/category/disease/target)
#  - tryb "szeroki" (one-hot): nagłówki ['img','healthy','leaf_rust', ...]
#    wybieramy kolumnę z wartością > 0, preferując chorobę nad healthy.
#
# Heurystyki dopasowania plików:
#  - najpierw bierze wszystko PO OSTATNIM '_' (ucina prefiksy),
#  - potem oryginał, po pierwszym '_' i '-', ogon cyfr/hex (>=12),
#  - indeksuje exact/lower/stem/normalized.

import argparse, csv, hashlib, json, random, re, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

# --------------------- mapowania ---------------------
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
    "HealthyLeaf": "Healthy",  # folderowa zdrowa → Healthy
    "LeafBlight": "LeafBlight",
    "FusariumFootRot": "FusariumFootRot",
    "WheatBlast": "WheatBlast",
    "BlackPoint": "BlackPoint",
}
CLASSES_A7 = ["Healthy","LeafBlight","Rust","PowderyMildew","FusariumFootRot","WheatBlast","BlackPoint"]

MAP_WFD_TO_B5 = {
    "healthy": "HealthyLeaf",
    "leaf_rust": "LeafBlight",
    "stem_rust": "LeafBlight",
    "yellow_rust": "LeafBlight",
    "powdery_mildew": "LeafBlight",
    "septoria": "LeafBlight",
    "seedlings": None,  # drop
}
MAP_FOLDER_TO_B5 = {
    "HealthyLeaf": "HealthyLeaf",
    "LeafBlight": "LeafBlight",
    "FusariumFootRot": "FusariumFootRot",
    "WheatBlast": "WheatBlast",
    "BlackPoint": "BlackPoint",
}
CLASSES_B5 = ["HealthyLeaf","LeafBlight","FusariumFootRot","WheatBlast","BlackPoint"]

# WFD znane kolumny (one-hot)
WFD_LABEL_COLUMNS = ["healthy","leaf_rust","powdery_mildew","seedlings","septoria","stem_rust","yellow_rust"]

# --------------------- konfig ---------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMG_SIZE = 256
DEFAULT_SEED = 1337
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.20
DEFAULT_AUG_MULT_CLS = 2
DEFAULT_AUG_MULT_BIN = 2

# --------------------- utils ---------------------
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_class_dirs(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])

def list_pairs_from_folder_tree(src: Path) -> List[Tuple[Path, str]]:
    pairs = []
    for cdir in list_class_dirs(src):
        cname = cdir.name.strip()
        for p in cdir.rglob("*"):
            if is_image(p):
                pairs.append((p, cname))
    if not pairs:
        raise SystemExit(f"Brak obrazów w strukturze folderowej: {src}")
    return pairs

# --- indeks plików z wieloma kluczami dopasowania ---
def build_multikey_index(root: Path):
    idx_exact: Dict[str, List[Path]] = defaultdict(list)     # exact basename
    idx_lc:    Dict[str, List[Path]] = defaultdict(list)     # lowercase basename
    idx_stem:  Dict[str, List[Path]] = defaultdict(list)     # stem
    idx_stemlc:Dict[str, List[Path]] = defaultdict(list)     # lowercase stem
    idx_norm:  Dict[str, List[Path]] = defaultdict(list)     # znormalizowane ([_-\s] -> "")

    for p in root.rglob("*"):
        if not is_image(p): continue
        b = p.name
        s = p.stem
        bl = b.lower()
        sl = s.lower()
        bn = re.sub(r"[\s_\-]+", "", b.lower())
        sn = re.sub(r"[\s_\-]+", "", s.lower())

        idx_exact[b].append(p)
        idx_lc[bl].append(p)
        idx_stem[s].append(p)
        idx_stemlc[sl].append(p)
        idx_norm[bn].append(p)
        idx_norm[sn].append(p)

    return {
        "exact": idx_exact,
        "lc": idx_lc,
        "stem": idx_stem,
        "stemlc": idx_stemlc,
        "norm": idx_norm,
    }

def _candidate_bases(raw: str) -> List[Tuple[str, str]]:
    """
    Generuje kandydatów (base, ext), priorytetowo:
    1) wariant po OSTATNIM '_' (ucina wszystko przed ostatnim '_' łącznie z nim),
    2) oryginał,
    3) wariant po PIERWSZYM '_' i po PIERWSZYM '-',
    4) ogon hex/cyfrowy (>=12).
    """
    r = raw.strip().strip('"').strip("'").replace("\\", "/")
    base = Path(r).name
    stem, ext = Path(base).stem, Path(base).suffix

    cand: List[Tuple[str, str]] = []

    # 1) po OSTATNIM '_': "pref1_pref2_ABC.jpg" -> "ABC.jpg"
    if "_" in base:
        right_last = base.rsplit("_", 1)[-1]
        if right_last:
            cand.append((right_last, ext))
            cand.append((right_last, ""))

    # 2) oryginał
    cand.append((base, ext))

    # 3) po PIERWSZYM '_' i '-'
    if "_" in base:
        right_first = base.split("_", 1)[1]
        if right_first:
            cand.append((right_first, ext))
    if "-" in base:
        right_dash = base.split("-", 1)[1]
        if right_dash:
            cand.append((right_dash, ext))

    # 4) ogon hex/cyfrowy (>=12), np. 0000000001ffffff
    m = re.search(r"([0-9a-fA-F]{12,})$", stem)
    if m:
        tail = m.group(1)
        cand.append((tail + ext, ext))
        cand.append((tail, ""))

    # unikalizacja kolejności
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for b, e in cand:
        key = (b.lower(), e.lower())
        if key not in seen and b:
            seen.add(key)
            uniq.append((b, e))
    return uniq

def find_in_index(raw: str, root: Path, index: Dict[str, Dict[str, List[Path]]]) -> Optional[Path]:
    # 0) spróbuj traktować jako ścieżkę względem root
    r = raw.strip().strip('"').strip("'").replace("\\", "/")
    p = (root / r)
    if p.exists() and is_image(p):
        return p

    # 1) generuj kandydatów
    cands = _candidate_bases(raw)

    # 2) dopasowanie po wielu kluczach
    for base, _ext in cands:
        stem = Path(base).stem
        keys = [
            ("exact", base),
            ("lc", base.lower()),
            ("stem", stem),
            ("stemlc", stem.lower()),
            ("norm", re.sub(r"[\s_\-]+", "", base.lower())),
            ("norm", re.sub(r"[\s_\-]+", "", stem.lower())),
        ]
        for bucket, key in keys:
            lst = index[bucket].get(key)
            if lst:
                return lst[0]
    return None

# --- CSV czytanie z autodetekcją separatora ---
def read_csv_smart(csv_path: Path):
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    # heurystyka separatora
    delim = "," if txt.count(",") >= txt.count(";") else ";"
    reader = csv.DictReader(txt.splitlines(), delimiter=delim)
    if reader.fieldnames is None:
        raise SystemExit(f"CSV bez nagłówka: {csv_path}")
    rows = list(reader)
    return rows, reader.fieldnames, delim

def detect_col(header: List[str], pref: List[str], fallback_tokens: List[str]) -> Optional[str]:
    hl = [h.strip().lower() for h in header]
    for cand in pref:
        if cand.lower() in hl:
            return header[hl.index(cand.lower())]
    for h in header:
        l = h.lower()
        if any(tok in l for tok in fallback_tokens):
            return h
    return None

def _truthy(v: str) -> Optional[float]:
    """Zwraca True-ish jako float gdy 1/true/x itd., False-ish gdy 0/pusty, None gdy nieczytelne."""
    if v is None: return None
    s = str(v).strip().lower()
    if s == "": return None
    # liczbowe
    try:
        f = float(s.replace(",", "."))
        return f
    except Exception:
        pass
    if s in {"1","true","t","yes","y","x","✓"}: return 1.0
    if s in {"0","false","f","no","n"}: return 0.0
    # coś jest, ale nie wiemy — potraktuj jak 1.0
    return 1.0

# --------------------- wczytanie CSV -> (Path, label) ---------------------
def read_pairs_from_csv(csv_path: Path, root: Path,
                        img_col: Optional[str], cls_col: Optional[str],
                        dry_log: Dict) -> List[Tuple[Path, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Nie znaleziono CSV: {csv_path}")

    rows, header, delim = read_csv_smart(csv_path)

    # detekcja kolumn
    img_c = img_col or detect_col(header,
                                  ["image","filename","file","img","path","image_name","name","image_id","img_name"],
                                  ["file","image","img","path","name","id"])
    cls_c = cls_col or detect_col(header,
                                  ["label","class","category","disease","target"],
                                  ["label","class","category","disease","target"])

    # tryb "szeroki" jeśli nie ma jednokolumnowej etykiety
    wide_mode = False
    wide_cols: List[str] = []
    if cls_c is None:
        wide_cols = [c for c in header if c.strip().lower() in WFD_LABEL_COLUMNS]
        wide_mode = len(wide_cols) > 0

    if img_c is None or (cls_c is None and not wide_mode):
        raise SystemExit(f"Nie wykryto kolumn (image/label). Nagłówek: {header}")

    index = build_multikey_index(root)
    pairs: List[Tuple[Path, str]] = []
    unmatched: List[str] = []
    wide_multi_hits = 0
    wide_zero_hits = 0

    for row in rows:
        raw_name = str(row.get(img_c, "")).strip()
        if not raw_name:
            continue

        if not wide_mode:
            raw_cls  = str(row.get(cls_c, "")).strip()
            if not raw_cls:
                continue
            chosen_label = raw_cls
        else:
            # wybór etykiety z one-hot
            scores = []
            for c in wide_cols:
                v = _truthy(row.get(c, ""))
                if v is not None and v > 0:
                    scores.append((c, float(v)))
            if not scores:
                wide_zero_hits += 1
                continue

            # preferuj chorych nad healthy gdy jest konflikt
            non_healthy = [(c, s) for c, s in scores if c.strip().lower() != "healthy"]
            if non_healthy:
                chosen_label = max(non_healthy, key=lambda x: x[1])[0]
                if len(non_healthy) > 1:
                    wide_multi_hits += 1
            else:
                chosen_label = "healthy"
                if len(scores) > 1:
                    wide_multi_hits += 1

        # seedlings drop
        if chosen_label.strip().lower() == "seedlings":
            continue

        p = find_in_index(raw_name, root, index)
        if p is None:
            unmatched.append(raw_name)
            continue
        pairs.append((p, chosen_label))

    # log diagnostyczny
    dry_log["csv"] = {
        "delimiter": delim,
        "header": header,
        "detected_img_col": img_c,
        "detected_cls_col": cls_c if not wide_mode else None,
        "wide_mode": wide_mode,
        "wide_cols": wide_cols,
        "rows": len(rows),
        "matched": len(pairs),
        "unmatched": len(unmatched),
        "unmatched_sample": list(dict.fromkeys(unmatched))[:50],
        "wide_multi_hits": wide_multi_hits,
        "wide_zero_hits": wide_zero_hits,
    }

    return pairs

# --------------------- normalizacja etykiet CLS ---------------------
def map_label(schema: str, source: str, label: str) -> Optional[str]:
    l = label.strip()
    if source == "WFD":
        if schema == "A7": return MAP_WFD_TO_A7.get(l, None)
        else:               return MAP_WFD_TO_B5.get(l, None)
    else:
        if schema == "A7": return MAP_FOLDER_TO_A7.get(l, None)
        else:               return MAP_FOLDER_TO_B5.get(l, None)

def canon_cls_label(label: str, schema: str) -> str:
    """Wyrównuje 'healthy'/'HealthyLeaf' do jednej nazwy zależnie od schematu."""
    if not label:
        return label
    l = label.strip()
    if schema == "B5":
        return "HealthyLeaf" if l.lower() in {"healthy","healthyleaf"} else l
    else:  # A7
        return "Healthy" if l.lower() in {"healthy","healthyleaf"} else l

# --------------------- split & pack ---------------------
def dedup_pairs(pairs: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    seen: Dict[Path, str] = {}
    out: List[Tuple[Path, str]] = []
    for p, c in pairs:
        if p not in seen:
            seen[p] = c; out.append((p, c))
    return out

def stratified_split(pairs: List[Tuple[Path, str]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    by_cls: Dict[str, List[Path]] = defaultdict(list)
    for p, c in pairs: by_cls[c].append(p)

    tr, va, te = [], [], []
    for c, plist in by_cls.items():
        plist = plist[:]; rng.shuffle(plist)
        n = len(plist)
        n_te = int(round(n * test_ratio))
        n_va = int(round((n - n_te) * val_ratio))
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
                t = (t_aug if spec.split == "train" else t_base)(im)
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

def to_bin_pairs(pairs: List[Tuple[Path, str]]):
    out = []
    for p, c in pairs:
        out.append((p, "healthy" if c.lower().startswith("healthy") else "diseased"))
    return out, ["healthy", "diseased"]

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
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = unique_name(dst_dir, src.stem)
        ensure_jpg_from_any(src, dst_dir / dst_name, quality=quality)

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser("Merge folder-tree + WFD+CSV → packs(train/val) + JPG test (no leakage)")
    ap.add_argument("--src_a", required=True, help="Zbiór folderowy (każda klasa to folder)")
    ap.add_argument("--src_b", required=True, help="Zbiór WFD (drzewo obrazów)")
    ap.add_argument("--csv",    required=True, help="CSV dla WFD: nazwa pliku -> klasa lub one-hot kolumny")
    ap.add_argument("--img_col", default=None, help="Nazwa kolumny z nazwą pliku (opcjonalnie wymuś)")
    ap.add_argument("--cls_col", default=None, help="Nazwa kolumny z etykietą (opcjonalnie wymuś)")

    ap.add_argument("--out", required=True, help="Folder wyjściowy")
    ap.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--aug_mult_cls", type=int, default=DEFAULT_AUG_MULT_CLS)
    ap.add_argument("--aug_mult_bin", type=int, default=DEFAULT_AUG_MULT_BIN)
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--schema", choices=["A7","B5"], default="A7",
                    help="A7 = 7 klas; B5 = 5 klas jak w folderach; seedlings zawsze DROP.")
    ap.add_argument("--dryrun", action="store_true",
                    help="Nie buduje paczek, tylko diagnozuje CSV i dopasowania plików.")

    args = ap.parse_args()
    set_seeds(args.seed)

    dry_log: Dict = {}

    # 1) wczytaj pary
    pairs_folder_raw = list_pairs_from_folder_tree(Path(args.src_a))
    pairs_wfd_raw    = read_pairs_from_csv(Path(args.csv), Path(args.src_b),
                                           args.img_col, args.cls_col,
                                           dry_log)

    # 2) mapuj do wspólnych klas + kanonizuj nazwy
    pairs_folder: List[Tuple[Path, str]] = []
    for p, lab in pairs_folder_raw:
        mapped = map_label(args.schema, "FOLDER", lab)
        if mapped is not None:
            mapped = canon_cls_label(mapped, args.schema)
            pairs_folder.append((p, mapped))

    pairs_wfd: List[Tuple[Path, str]] = []
    for p, lab in pairs_wfd_raw:
        mapped = map_label(args.schema, "WFD", lab)
        if mapped is not None:
            mapped = canon_cls_label(mapped, args.schema)
            pairs_wfd.append((p, mapped))

    all_pairs = dedup_pairs(pairs_folder + pairs_wfd)

    # 2b) walidacja etykiet CLS
    classes = CLASSES_A7 if args.schema == "A7" else CLASSES_B5
    bad = {c for _, c in all_pairs if c not in classes}
    if bad:
        raise SystemExit(f"Nieznane etykiety w CLS: {sorted(bad)}; oczekiwane {classes}")

    # log diagnostyczny
    dry_log["counts"] = {
        "folder_raw": len(pairs_folder_raw),
        "wfd_raw": len(pairs_wfd_raw),
        "folder_kept": len(pairs_folder),
        "wfd_kept": len(pairs_wfd),
        "combined": len(all_pairs),
        "classes": classes,
    }

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.dryrun or not all_pairs:
        unmatched = dry_log.get("csv", {}).get("unmatched_sample", [])
        if unmatched:
            (out_dir/"unmatched_sample.csv").write_text("raw_name\n" + "\n".join(unmatched), encoding="utf-8")
        (out_dir/"diagnostic.json").write_text(json.dumps(dry_log, indent=2, ensure_ascii=False), encoding="utf-8")

        if not all_pairs:
            raise SystemExit(f"CSV zaczytany, ale 0 par po dopasowaniu. "
                             f"Diagnoza: {out_dir/'diagnostic.json'}  "
                             f"Przykłady niedopasowanych: {out_dir/'unmatched_sample.csv'}")

        print(f"[DRYRUN] Diagnoza zapisana do: {out_dir}")
        return

    print("Docelowe klasy:", classes)

    # 3) split
    tr_cls, va_cls, te_cls = stratified_split(all_pairs, args.val_ratio, args.test_ratio, args.seed)
    
    # NO-LEAK: filtruj test względem train/val
    _dg = DedupGuard(log_csv=out_dir/'_logs'/'dedup_wheat_cls.csv')
    _dg.add_pairs(tr_cls + va_cls)
    te_cls, _ = _dg.filter_test(te_cls)
    print(f"Rozmiary: train={len(tr_cls)}  val={len(va_cls)}  test={len(te_cls)}")

    # 4) PACZKI CLS: train/val
    write_pack(PackSpec(out_dir/"cls"/"train"/"s000", "train", classes, args.aug_mult_cls, args.img_size), tr_cls)
    if va_cls:
        write_pack(PackSpec(out_dir/"cls"/"val"/"s000",   "val",   classes, 1, args.img_size), va_cls)

    # 5) PACZKI BIN: train/val z (train+val)
    bin_tr_pairs, bin_classes = to_bin_pairs(tr_cls + va_cls)
    write_pack(PackSpec(out_dir/"bin"/"train"/"s000", "train", bin_classes, args.aug_mult_bin, args.img_size), bin_tr_pairs)

    # 6) TEST: JPG
    if te_cls:
        export_jpg_tree(te_cls, out_dir/"test_images"/"cls", quality=args.jpg_quality)
        te_bin_pairs, _ = to_bin_pairs(te_cls)
        export_jpg_tree(te_bin_pairs, out_dir/"test_images"/"bin", quality=args.jpg_quality)

    cfg = {
        "schema": args.schema,
        "classes": classes,
        "img_size": args.img_size,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "aug_mult_cls": args.aug_mult_cls,
        "aug_mult_bin": args.aug_mult_bin,
        "notes": [
            "WFD 'seedlings' pominięte.",
            "BIN z train+val, test tylko JPG.",
            "CSV: autodetekcja separatora i kolumn; tryb one-hot ['img', 'healthy', ...] wspierany.",
            "Dopasowanie nazw: po ostatnim '_' + inne heurystyki.",
            "Kanonizacja healthy ↔ HealthyLeaf zależnie od schematu.",
        ],
        "diagnostic": dry_log,
    }
    (out_dir/"builder_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GOTOWE ✅  Wyjście:", out_dir.as_posix())

if __name__ == "__main__":
    main()
