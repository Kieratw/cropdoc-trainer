import argparse, json
from pathlib import Path
from typing import List, Tuple

import build_dataset as BD


HASH_THRESHOLD = 2
import hashlib, io, csv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
    if b is None: return None
    return hashlib.sha1(b).hexdigest()
def _ahash64(p: Path, size: int = 8):
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size), Image.BICUBIC)
            import numpy as np
            a = np.asarray(im, dtype=np.float32)
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
        self.th = int(threshold); self.within = bool(within_test); self.log_csv = log_csv
        self.idx_sha = set(); self.idx_ph = []
    def add_pairs(self, pairs):
        for p, _ in pairs:
            from pathlib import Path as _P
            p = _P(p)
            if not p.exists(): continue
            s = _sha1(p); h = _ahash64(p)
            if s: self.idx_sha.add(s)
            if h is not None: self.idx_ph.append(h)
    def filter_test(self, test_pairs):
        kept, removed = [], []
        for p, c in test_pairs:
            from pathlib import Path as _P
            pp = _P(p)
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
                from pathlib import Path as _P
                s = _sha1(_P(p)); h = _ahash64(_P(p))
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

def main():
    ap = argparse.ArgumentParser("Rapeseed (per-plant) builder with no-leak test")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--aug-mult", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hash-threshold", type=int, default=HASH_THRESHOLD)
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    (out/"cls").mkdir(parents=True, exist_ok=True); (out/"bin").mkdir(parents=True, exist_ok=True)
    test_img_root = out/"test_images"; test_img_cls = test_img_root/"cls"; test_img_bin = test_img_root/"bin"

    # CLS (Mono)
    cls_pairs, cls_classes = BD.rapeseed_pairs_mono_multiclass(src)
    if cls_pairs:
        tr, va, te = BD.simple_split(cls_pairs, args.val_ratio, args.test_ratio, args.seed)
        dg = DedupGuard(threshold=args.hash_threshold, within_test=True, log_csv=(out/"_logs"/"dedup_rapeseed_cls.csv"))
        dg.add_pairs(tr + va); te, _ = dg.filter_test(te)
        BD.write_pack(out/"cls"/"train","train", BD.build_records(tr, cls_classes, args.aug_mult, args.seed, "train(cls)"), cls_classes)
        BD.write_pack(out/"cls"/"val","val",     BD.build_records(va, cls_classes, 0,             args.seed, "val(cls)"),   cls_classes)
        BD.copy_test_images(te, test_img_cls, "cls")

    # BIN (Syl healthy vs reszta)
    bin_pairs, bin_classes = BD.rapeseed_pairs_binary(src)
    if bin_pairs:
        trb, vab, teb = BD.simple_split(bin_pairs, args.val_ratio, args.test_ratio, args.seed)
        dgb = DedupGuard(threshold=args.hash_threshold, within_test=True, log_csv=(out/"_logs"/"dedup_rapeseed_bin.csv"))
        dgb.add_pairs(trb + vab); teb, _ = dgb.filter_test(teb)
        BD.write_pack(out/"bin"/"train","train", BD.build_records(trb, bin_classes, args.aug_mult, args.seed, "train(bin)"), bin_classes)
        BD.write_pack(out/"bin"/"val","val",     BD.build_records(vab, bin_classes, 0,             args.seed, "val(bin)"),   bin_classes)
        BD.copy_test_images(teb, test_img_bin, "bin")

    (out/"builder_config.json").write_text(json.dumps({
        "dataset":"rapeseed",
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio,
        "seed": args.seed, "aug_mult_train": args.aug_mult, "hash_threshold": args.hash_threshold
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
