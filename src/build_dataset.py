# make_dataset.py — buduje równolegle: packs/cls/* oraz packs/bin/*
import argparse, json, random, time, re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ===== Domyślne =====
IMG_SIZE   = 256
VAL_RATIO  = 0.15
TEST_RATIO = 0.00
SEED       = 42
AUG_MULT   = 2                 # augmenty tylko dla TRAIN; 0 = brak
IMG_EXTS   = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ===== Utils =====
def banner(txt): print("\n" + "="*8 + " " + txt + " " + "="*8)

def has_images(p: Path) -> bool:
    return any(pp.is_file() and pp.suffix.lower() in IMG_EXTS for pp in p.rglob("*"))

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
        # uszkodzony / nieobraz / dziwne rozszerzenie udające jpg → pomiń
        return None


def aug_image(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(np.transpose(arr, (1,2,0)))
    if rng.random() < 0.5: img = ImageOps.mirror(img)
    if rng.random() < 0.2: img = ImageOps.flip(img)
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

# ===== Layout detection =====
def detect_layout(src: Path) -> str:
    if (src/"train").exists() or (src/"val").exists() or (src/"validation").exists() or (src/"test").exists():
        return "presplit_generic"
    if (src/"Splitted_Dataset").exists() and (src/"Splitted_Dataset"/"Train").exists():
        return "wheat_splitted"
    if any(d.is_dir() and d.name.lower().startswith("cross-validation") for d in src.iterdir()):
        return "tomato_cv"
    group_names = {"Mono","MuRoHi","MuRoLo","Syl_Hi","Syl_Lo"}
    if any((src/g).exists() for g in group_names):
        return "rapeseed_depth2"
    if any(d.is_dir() for d in src.iterdir()):
        return "flat1"
    return "unknown"

# ===== Listing & mapping =====
def list_paths_with_progress(base: Path, classes: List[str]) -> List[Tuple[Path,str]]:
    pairs=[]
    for c in tqdm(classes, desc="scan classes", unit="cls"):
        for p in (base/c).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                pairs.append((p, c))
    return pairs

def list_flat1(src: Path) -> Tuple[List[Tuple[Path,str]], List[str]]:
    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    pairs = list_paths_with_progress(src, classes)
    return pairs, classes

def list_wheat_splitted(src: Path) -> Dict[str, List[Tuple[Path,str]]]:
    root = src/"Splitted_Dataset"
    splits = {"train":"Train","val":"Validation","test":"Test"}
    out={}
    for key,folder in splits.items():
        sp = root/folder
        if not sp.exists(): continue
        classes = [d.name for d in sp.iterdir() if d.is_dir()]
        items = list_paths_with_progress(sp, classes)
        # normalizacja healthy
        out[key] = [(p, ("healthy" if c.lower().startswith("healthy") else c)) for p,c in items]
    return out

def list_tomato_cv(src: Path, fold:int=1) -> Dict[str, List[Tuple[Path,str]]]:
    cv_root = src/f"Cross-validation{fold}"
    train_root = cv_root/"Train"
    test_root  = cv_root/"Test"
    if not train_root.exists() or not test_root.exists():
        raise SystemExit(f"Brak Cross-validation{fold}/Train lub Test")
    out={}
    for key,sp in {"train":train_root,"val":test_root}.items():
        classes = [d.name for d in sp.iterdir() if d.is_dir()]
        out[key] = list_paths_with_progress(sp, classes)
    return out

# ---- RZEPAK: budowa par dla multiclass i binary ----
def rapeseed_pairs_mono_multiclass(src: Path) -> Tuple[List[Tuple[Path,str]], List[str]]:
    mono = src/"Mono"
    if not mono.exists():
        return [], []
    # Mono/<Alt_Hi|Lo> -> Alt itd.
    base_names = []
    for d in mono.iterdir():
        if d.is_dir():
            base_names.append(d.name.split("_")[0].title())
    classes = sorted(set(base_names))
    pairs=[]
    for d in tqdm([x for x in mono.iterdir() if x.is_dir()], desc="scan Mono", unit="cls"):
        base = d.name.split("_")[0].title()
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                pairs.append((p, base))
    return pairs, classes

def rapeseed_pairs_binary(src: Path) -> Tuple[List[Tuple[Path,str]], List[str]]:
    pairs=[]
    # Syl_* -> healthy
    for g in ["Syl_Hi","Syl_Lo"]:
        gp = src/g
        if gp.exists():
            for p in gp.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    pairs.append((p, "healthy"))
    # Mono + MuRoHi + MuRoLo -> diseased
    for g in ["Mono","MuRoHi","MuRoLo"]:
        gp = src/g
        if gp.exists():
            for p in gp.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    pairs.append((p, "diseased"))
    return pairs, ["healthy","diseased"]

# ===== Splits =====
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
            # „zużyj” 1 krok paska, bo tej bazowej próbki nie będzie (augmentów też nie robimy)
            pbar.update(1)
            continue
        recs.append((arr, c2i[c])); pbar.update(1)
        for _ in range(aug_mult):
            recs.append((aug_image(arr, rng), c2i[c])); pbar.update(1)
    pbar.close()

    if skipped:
        log = Path("skipped_" + split_name.replace("/", "_") + ".txt")
        try:
            log.write_text("\n".join(skipped), encoding="utf-8")
            print(f"[{split_name}] SKIPPED: {len(skipped)} plików. Lista → {log}")
        except Exception:
            print(f"[{split_name}] SKIPPED: {len(skipped)} plików (nie udało się zapisać logu).")
    return recs

# ===== Bin-mapper dla innych datasetów =====
def is_healthy_name(c: str) -> bool:
    return bool(re.search(r"(healthy|control|normal|none|ok)", c, re.IGNORECASE))

def make_binary_from_pairs(pairs: List[Tuple[Path,str]]) -> Tuple[List[Tuple[Path,str]], List[str]]:
    out=[]
    for p,c in pairs:
        out.append((p, "healthy" if is_healthy_name(c) else "diseased"))
    return out, ["healthy","diseased"]

# ===== Main =====
def main():
    ap = argparse.ArgumentParser("Dataset → memmap tensors (multiclass + binary)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    (out/"cls").mkdir(parents=True, exist_ok=True)
    (out/"bin").mkdir(parents=True, exist_ok=True)

    banner("DETECT LAYOUT")
    layout = detect_layout(src)
    print(f"[detect] {layout}")

    start = time.time()

    if layout == "rapeseed_depth2":
        # MULTICLASS (Mono only)
        banner("RAPESEED → MULTICLASS (Mono only)")
        cls_pairs, cls_classes = rapeseed_pairs_mono_multiclass(src)
        if cls_pairs:
            tr,va,_ = simple_split(cls_pairs, VAL_RATIO, TEST_RATIO, SEED)
            rec_tr = build_records(tr, cls_classes, AUG_MULT, SEED, "train(cls)")
            rec_va = build_records(va, cls_classes, 0, SEED, "val(cls)")
            write_pack(out/"cls"/"train", "train", rec_tr, cls_classes)
            write_pack(out/"cls"/"val",   "val",   rec_va, cls_classes)
        else:
            print("[cls] Mono puste — pomijam multiklasę dla rzepaku.")

        # BINARY (Syl healthy, reszta diseased)
        banner("RAPESEED → BINARY (healthy vs diseased)")
        bin_pairs, bin_classes = rapeseed_pairs_binary(src)
        trb,vab,_ = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, SEED)
        rec_trb = build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)")
        rec_vab = build_records(vab, bin_classes, 0, SEED, "val(bin)")
        write_pack(out/"bin"/"train", "train", rec_trb, bin_classes)
        write_pack(out/"bin"/"val",   "val",   rec_vab, bin_classes)

    elif layout == "wheat_splitted":
        banner("WHEAT → MULTICLASS")
        S = list_wheat_splitted(src)
        cls_classes = sorted({c for _,c in S.get("train",[])})
        rec_tr = build_records(S.get("train",[]), cls_classes, AUG_MULT, SEED, "train(cls)")
        rec_va = build_records(S.get("val",[]),   cls_classes, 0, SEED, "val(cls)")
        write_pack(out/"cls"/"train","train", rec_tr, cls_classes)
        write_pack(out/"cls"/"val",  "val",   rec_va, cls_classes)
        if "test" in S and S["test"]:
            rec_te = build_records(S["test"], cls_classes, 0, SEED, "test(cls)")
            write_pack(out/"cls"/"test", "test", rec_te, cls_classes)

        banner("WHEAT → BINARY")
        all_pairs = S.get("train",[]) + S.get("val",[]) + S.get("test",[])
        bin_pairs, bin_classes = make_binary_from_pairs(all_pairs)
        trb,vab,_ = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, SEED)
        write_pack(out/"bin"/"train","train", build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)"), bin_classes)
        write_pack(out/"bin"/"val",  "val",   build_records(vab, bin_classes, 0, SEED, "val(bin)"), bin_classes)

    elif layout == "tomato_cv":
        banner("TOMATO → MULTICLASS (CV fold=1)")
        S = list_tomato_cv(src, fold=1)
        cls_classes = sorted({c for _,c in S["train"]})
        rec_tr = build_records(S["train"], cls_classes, AUG_MULT, SEED, "train(cls)")
        rec_va = build_records(S["val"],   cls_classes, 0, SEED, "val(cls)")
        write_pack(out/"cls"/"train","train", rec_tr, cls_classes)
        write_pack(out/"cls"/"val",  "val",   rec_va, cls_classes)

        banner("TOMATO → BINARY")
        bin_pairs, bin_classes = make_binary_from_pairs(S["train"] + S["val"])
        trb,vab,_ = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, SEED)
        write_pack(out/"bin"/"train","train", build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)"), bin_classes)
        write_pack(out/"bin"/"val",  "val",   build_records(vab, bin_classes, 0, SEED, "val(bin)"), bin_classes)

    elif layout == "presplit_generic":
        banner("PRESPLIT → MULTICLASS")
        def list_split(sp: Path):
            classes = sorted([d.name for d in sp.iterdir() if d.is_dir()])
            pairs=[]
            for c in tqdm(classes, desc=f"scan {sp.name}", unit="cls"):
                for p in (sp/c).rglob("*"):
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        pairs.append((p,c))
            return pairs, classes
        tr_pairs, cls_classes = list_split(src/"train")
        rec_tr = build_records(tr_pairs, cls_classes, AUG_MULT, SEED, "train(cls)")
        write_pack(out/"cls"/"train","train", rec_tr, cls_classes)
        va_dir = src/("val" if (src/"val").exists() else "validation") if ((src/"val").exists() or (src/"validation").exists()) else None
        if va_dir:
            va_pairs,_ = list_split(va_dir)
            write_pack(out/"cls"/"val","val", build_records(va_pairs, cls_classes, 0, SEED, "val(cls)"), cls_classes)
        if (src/"test").exists():
            te_pairs,_ = list_split(src/"test")
            write_pack(out/"cls"/"test","test", build_records(te_pairs, cls_classes, 0, SEED, "test(cls)"), cls_classes)

        banner("PRESPLIT → BINARY")
        all_pairs = tr_pairs + (va_pairs if (va_dir) else []) + (te_pairs if (src/"test").exists() else [])
        bin_pairs, bin_classes = make_binary_from_pairs(all_pairs)
        trb,vab,_ = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, SEED)
        write_pack(out/"bin"/"train","train", build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)"), bin_classes)
        write_pack(out/"bin"/"val",  "val",   build_records(vab, bin_classes, 0, SEED, "val(bin)"), bin_classes)

    elif layout == "flat1":
        banner("FLAT1 → MULTICLASS")
        items, cls_classes = list_flat1(src)
        tr,va,te = simple_split(items, VAL_RATIO, TEST_RATIO, SEED)
        write_pack(out/"cls"/"train","train", build_records(tr, cls_classes, AUG_MULT, SEED, "train(cls)"), cls_classes)
        write_pack(out/"cls"/"val","val",     build_records(va, cls_classes, 0, SEED, "val(cls)"), cls_classes)
        if te:
            write_pack(out/"cls"/"test","test", build_records(te, cls_classes, 0, SEED, "test(cls)"), cls_classes)

        banner("FLAT1 → BINARY")
        bin_pairs, bin_classes = make_binary_from_pairs(items)
        trb,vab,_ = simple_split(bin_pairs, VAL_RATIO, TEST_RATIO, SEED)
        write_pack(out/"bin"/"train","train", build_records(trb, bin_classes, AUG_MULT, SEED, "train(bin)"), bin_classes)
        write_pack(out/"bin"/"val",  "val",   build_records(vab, bin_classes, 0, SEED, "val(bin)"), bin_classes)

    else:
        raise SystemExit("Nie rozpoznano układu. Pokaż strukturę katalogów albo popraw ścieżkę.")

    # zapis globalnego info
    (out/"builder_config.json").write_text(json.dumps({
        "img_size": IMG_SIZE, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO,
        "seed": SEED, "aug_mult_train": AUG_MULT,
        "outputs": ["cls","bin"]
    }, indent=2), encoding="utf-8")

    mins = (time.time()-start)/60
    banner(f"GOTOWE w {mins:.1f} min")
    print(f"CLS packs: {(out/'cls'/'train').as_posix()}, {(out/'cls'/'val').as_posix() if (out/'cls'/'val').exists() else '—'}")
    print(f"BIN packs: {(out/'bin'/'train').as_posix()}, {(out/'bin'/'val').as_posix() if (out/'bin'/'val').exists() else '—'}")

if __name__ == "__main__":
    main()
