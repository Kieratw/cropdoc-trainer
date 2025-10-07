from pathlib import Path
import shutil, random, re, json

random.seed(42)

# Gdzie masz dane
SRC_POTATO = Path("data_raw/potato")
SRC_CORN   = Path("data_raw/corn")

OUT = Path("data_merged")
SPLIT = (0.7, 0.15, 0.15)   # train/val/test

def norm(s):
    return re.sub(r"\s+|[-_]+"," ", s.lower()).strip()

# Tokeny chorób (bez nazwy rośliny)
CORN_MAP = {
    "corn__common_rust":     [("common",), ("rust",)],
    "corn__gray_leaf_spot":  [("gray","grey"), ("leaf",), ("spot",)],
    "corn__blight":          [("blight",)],
    "corn__healthy":         [("healthy","healty")],   # alias na literówkę
}
POTATO_MAP = {
    "potato__early_blight":  [("early",), ("blight",)],
    "potato__late_blight":   [("late",),  ("blight",)],
    "potato__healthy":       [("healthy","healty")],
}

def match_tokens(name, token_groups):
    n = norm(name)
    return all(any(tok in n for tok in group) for group in token_groups)

def find_class_dirs(root: Path, token_groups):
    """Szukaj podfolderów w danym cropie (corn/potato), które pasują do tokenów choroby."""
    dirs = []
    if not root.exists():
        return dirs
    for d in root.rglob("*"):
        if d.is_dir() and match_tokens(d.name, token_groups):
            imgs = [p for p in d.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
            if len(imgs) >= 5:
                dirs.append(d)
    return dirs

def collect_images(dirlist):
    """Zbierz obrazki z wielu folderów, usuń duplikaty po (nazwa, rozmiar)."""
    imgs, seen = [], set()
    for d in dirlist:
        for p in d.rglob("*"):
            if p.suffix.lower() not in [".jpg",".jpeg",".png"]:
                continue
            key = (p.name.lower(), p.stat().st_size)
            if key in seen:
                continue
            seen.add(key)
            imgs.append(p)
    random.shuffle(imgs)
    return imgs

def stratified_split(items, split):
    n = len(items)
    n_tr = int(n*split[0]); n_val = int(n*split[1])
    return items[:n_tr], items[n_tr:n_tr+n_val], items[n_tr+n_val:]

def dump_subset(imgs, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        shutil.copy2(p, dst/p.name)

def main():
    # (opcjonalnie) wyczyść output, jeśli chcesz zacząć od zera:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ["train","val","test"]:
        (OUT/s).mkdir(parents=True, exist_ok=True)

    stats = {}

    # --- C O R N ---
    for canon, toks in CORN_MAP.items():
        dirs = find_class_dirs(SRC_CORN, toks)
        imgs = collect_images(dirs)
        if not imgs:
            print(f"[WARN] no images for class: {canon}")
            continue
        tr, va, te = stratified_split(imgs, SPLIT)
        dump_subset(tr, OUT/"train"/canon)
        dump_subset(va, OUT/"val"/canon)
        dump_subset(te, OUT/"test"/canon)
        stats[canon] = {"train":len(tr),"val":len(va),"test":len(te)}
        print(f"{canon:>25}: {len(tr)}/{len(va)}/{len(te)}")

    # --- P O T A T O ---
    for canon, toks in POTATO_MAP.items():
        dirs = find_class_dirs(SRC_POTATO, toks)
        imgs = collect_images(dirs)
        if not imgs:
            print(f"[WARN] no images for class: {canon}")
            continue
        tr, va, te = stratified_split(imgs, SPLIT)
        dump_subset(tr, OUT/"train"/canon)
        dump_subset(va, OUT/"val"/canon)
        dump_subset(te, OUT/"test"/canon)
        stats[canon] = {"train":len(tr),"val":len(va),"test":len(te)}
        print(f"{canon:>25}: {len(tr)}/{len(va)}/{len(te)}")

    (OUT/"stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nDone → data_merged/{train,val,test}/<plant__disease>/*.jpg")

if __name__ == "__main__":
    main()