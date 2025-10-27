from __future__ import annotations

import csv, json, random, sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import numpy as np
from PIL import Image
import imagehash


CONFIG = {

    "TEST_DIRS": [
        r"D:\inzynierka\Datasety_do_treningu\potato\test_images",
        r"D:\inzynierka\Datasety_do_treningu\rapeseed\test_images",
        r"D:\inzynierka\Datasety_do_treningu\tomato\test_images",
        r"D:\inzynierka\Datasety_do_treningu\wheat\test_images",
    ],

    "PACKS": [
        r"D:\inzynierka\Datasety_do_treningu"
    ],

    "THRESHOLD": 6,       # 5..8 sensowne
    "HASH_SIZE": 8,

    "MAX_TEST_PER_DIR": None,   # np. 2000
    "MAX_PACK": None,           # np. 5000

    "WITHIN_TEST": False,

    "OUT": r"D:\wyniki\near_dups_ALL.csv",
    "SAVE_DEBUG": None,  # np. r"D:\wyniki\debug_png"
}


IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def phash_hex(img: Image.Image, hash_size: int = 8) -> str:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    h = imagehash.phash(img, hash_size=hash_size)
    return str(h)

def hex_to_int(h: str) -> int:
    return int(h, 16)

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

class BKNode:
    __slots__ = ("key", "items", "children")
    def __init__(self, key: int, item):
        self.key = key
        self.items = [item]
        self.children = {}

class BKTree:
    def __init__(self):
        self.root: Optional[BKNode] = None
    def insert(self, key: int, item):
        if self.root is None:
            self.root = BKNode(key, item); return
        node = self.root
        while True:
            d = hamming(key, node.key)
            if d == 0:
                node.items.append(item); return
            child = node.children.get(d)
            if child is None:
                node.children[d] = BKNode(key, item); return
            node = child
    def query(self, key: int, threshold: int):
        out = []
        node = self.root
        if node is None: return out
        stack = [node]
        while stack:
            node = stack.pop()
            d = hamming(key, node.key)
            if d <= threshold:
                for it in node.items:
                    out.append((d, it))
            lo, hi = max(0, d - threshold), d + threshold
            for dd, child in node.children.items():
                if lo <= dd <= hi:
                    stack.append(child)
        return out

def find_shards(packs_roots: List[Path]) -> List[Path]:
    shards = []
    for root in packs_roots:
        root = Path(root)
        for meta in root.rglob("meta.json"):
            shards.append(meta.parent)
    return shards

def iter_pack_images(shard_dir: Path, max_samples: Optional[int] = None):
    meta_path = shard_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    num = int(meta["num_samples"])
    C, H, W = map(int, meta["shape"])
    dtype = np.dtype(meta.get("dtype", "uint8"))
    data_path = shard_dir / meta.get("data_file", "images.dat")
    split = meta.get("split", "unknown")
    shard_name = shard_dir.name

    expected_bytes = num * C * H * dtype.itemsize * W
    actual_bytes = data_path.stat().st_size
    if expected_bytes != actual_bytes:
        raise RuntimeError(f"Rozmiar pliku {data_path} ({actual_bytes}) != oczekiwany ({expected_bytes}).")

    arr = np.memmap(data_path, dtype=dtype, mode="r", shape=(num, C, H, W))
    indices = range(num)
    if max_samples is not None and max_samples < num:
        indices = list(indices)
        random.shuffle(indices)
        indices = indices[:max_samples]

    for i in indices:
        img_chw = np.array(arr[i])
        img = np.transpose(img_chw, (1,2,0))
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        pil = Image.fromarray(img) if img.ndim == 3 else Image.fromarray(np.repeat(img, 3, axis=2))
        yield ((split, shard_name, int(i)), pil)
    del arr

def scan_build_tree_from_packs(packs_roots, hash_size, max_pack):
    print("[*] Buduję indeks z TRAIN/VAL (packi)...")
    shards = find_shards(packs_roots)
    if not shards:
        print("[ERR] Nie znaleziono shardów (meta.json).", file=sys.stderr); sys.exit(2)
    tree = BKTree()
    total = 0
    for shard in shards:
        try:
            for (split, shard_name, idx), img in iter_pack_images(shard, max_samples=max_pack):
                hh = phash_hex(img, hash_size=hash_size)
                hi = hex_to_int(hh)
                tree.insert(hi, (split, shard_name, idx, hh))
                total += 1
                if total % 5000 == 0:
                    print(f"  ... wczytano {total} z packów")
        except Exception as e:
            print(f"[WARN] Pomijam shard {shard}: {e}", file=sys.stderr)
    print(f"[*] Indeks gotowy. Obrazów w indexie: {total}")
    return tree

def iter_test_paths_one_dir(root_dir: Path):
    for p in Path(root_dir).rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p

def process_one_test_dir(dir_path, tree, cfg, writer, out_rows_counter):
    root = Path(dir_path)
    print(f"[*] === Test dir: {root} ===")
    max_test = cfg["MAX_TEST_PER_DIR"]
    threshold = cfg["THRESHOLD"]
    hash_size = cfg["HASH_SIZE"]
    within_test = cfg["WITHIN_TEST"]
    debug_dir = cfg["SAVE_DEBUG"]
    debug_dir = Path(debug_dir) if debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    test_tree = BKTree() if within_test else None
    seen = 0

    for p in iter_test_paths_one_dir(root):
        try:
            with Image.open(p) as img:
                img.load()
                hh = phash_hex(img, hash_size=hash_size)
        except Exception as e:
            print(f"[WARN] Nie mogę otworzyć {p}: {e}", file=sys.stderr)
            continue
        hi = hex_to_int(hh)

        hits = tree.query(hi, threshold)
        for dist, item in hits:
            split, shard_name, idx, hh_pack = item
            writer.writerow({
                "test_path": str(p),
                "pack_split": split,
                "pack_shard": shard_name,
                "pack_index": idx,
                "dist": dist,
                "hash_test_hex": hh,
                "hash_pack_hex": hh_pack
            })
            out_rows_counter[0] += 1
            if debug_dir:
                try:
                    with Image.open(p) as dbg_img:
                        dbg_img.save(debug_dir / f"TEST__{p.stem}__dist{dist}.png")
                except Exception as e:
                    print(f"[WARN] Debug save fail {p}: {e}", file=sys.stderr)

        if within_test:
            hits2 = test_tree.query(hi, threshold)
            for dist, item in hits2:
                path2, hh2 = item
                if path2 == str(p):
                    continue
                writer.writerow({
                    "test_path": str(p),
                    "pack_split": "WITHIN_TEST",
                    "pack_shard": Path(path2).parent.name,
                    "pack_index": Path(path2).name,
                    "dist": dist,
                    "hash_test_hex": hh,
                    "hash_pack_hex": hh2
                })
                out_rows_counter[0] += 1
            test_tree.insert(hi, (str(p), hh))

        seen += 1
        if seen % 1000 == 0:
            print(f"  ... {seen} obrazów w {root}")
        if max_test is not None and seen >= max_test:
            print(f"  [*] limit MAX_TEST_PER_DIR={max_test} osiągnięty w {root}")
            break

def main():
    cfg = CONFIG
    out_path = Path(cfg["OUT"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = scan_build_tree_from_packs(cfg["PACKS"], cfg["HASH_SIZE"], cfg["MAX_PACK"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_path","pack_split","pack_shard","pack_index","dist","hash_test_hex","hash_pack_hex"])
        writer.writeheader()
        out_rows_counter = [0]
        for td in cfg["TEST_DIRS"]:
            process_one_test_dir(td, tree, cfg, writer, out_rows_counter)
    print(f"[*] Zapisano {out_rows_counter[0]} wierszy do {out_path}")
    print("[*] Koniec.")

if __name__ == "__main__":
    main()
