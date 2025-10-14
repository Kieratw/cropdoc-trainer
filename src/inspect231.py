import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def infer_images_dat_shape(meta, data_path: Path):
    # meta["shape"] assumed [C,H,W]
    C, H, W = meta["shape"]
    item_bytes = int(C) * int(H) * int(W)
    fsize = os.path.getsize(data_path)
    if fsize % item_bytes != 0:
        raise ValueError(f"images.dat size {fsize} is not divisible by C*H*W={item_bytes}")
    N = fsize // item_bytes
    return int(N), int(C), int(H), int(W)

def memmap_images(data_path: Path, shape):
    # shape = (N,C,H,W)
    return np.memmap(data_path.as_posix(), dtype=np.uint8, mode="r", shape=shape)

def to_pil_from_CHW(arr):
    # arr: (C,H,W) uint8 -> PIL expects HWC
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    img = np.transpose(arr, (1,2,0))
    return Image.fromarray(img)

def inspect_pack(pack_dir: Path, show=3, save_samples=None):
    pack_dir = pack_dir.resolve()
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack dir does not exist: {pack_dir}")

    print("Inspecting:", pack_dir)
    meta_p = pack_dir / "meta.json"
    index_p = pack_dir / "index.npy"
    # default data file name per builder
    data_p = pack_dir / "images.dat"

    meta = load_json(meta_p)
    if meta is None:
        raise FileNotFoundError(f"Missing meta.json in {pack_dir}")
    print("\nmeta.json:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # optional global builder config (one level up or two levels)
    cfg_candidates = [pack_dir.parent.parent / "builder_config.json", pack_dir.parent.parent.parent / "builder_config.json"]
    for c in cfg_candidates:
        cfg = load_json(c)
        if cfg is not None:
            print("\nFound builder_config.json at", c)
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
            break

    # resolve filenames from meta if present
    data_name = meta.get("data_file", "images.dat")
    labels_name = meta.get("labels_file", "index.npy")
    data_p = pack_dir / data_name
    index_p = pack_dir / labels_name

    print("\nFiles we will use:")
    print(" data:", data_p)
    print(" labels:", index_p)

    if not data_p.exists():
        raise FileNotFoundError(f"No data file: {data_p}")
    if not index_p.exists():
        raise FileNotFoundError(f"No labels file: {index_p}")

    # infer N,C,H,W
    N, C, H, W = infer_images_dat_shape(meta, data_p)
    print(f"\nInferred shape -> N={N}, C={C}, H={H}, W={W}")

    # load memmap and labels
    mm = memmap_images(data_p, (N, C, H, W))
    labels = np.load(index_p, mmap_mode="r")
    print("labels shape:", labels.shape, " dtype:", labels.dtype)

    if len(mm) != len(labels):
        print("WARNING: lengths differ! images:", len(mm), " labels:", len(labels))

    # class names
    classes = meta.get("classes", None)
    if classes is None:
        print("No classes key in meta.json")
        classes = [str(i) for i in range(int(labels.max())+1)]
    print("\nClasses (from meta.json):", classes)

    # counts
    unique, counts = np.unique(np.array(labels).astype(int), return_counts=True)
    counts_map = {int(u): int(c) for u,c in zip(unique, counts)}
    print("\nLabel counts (index:count):")
    for idx, cnt in sorted(counts_map.items()):
        name = classes[idx] if idx < len(classes) else f"IDX_{idx}"
        print(f"  {idx} ({name}): {cnt}")

    # show / save several samples (first ones and some random)
    sample_indices = list(range(min(3, N)))
    # add some random ones if requested
    if show > 3:
        rs = np.random.default_rng(42).choice(N, max(0, show-3), replace=False).tolist()
        sample_indices += [i for i in rs if i not in sample_indices]

    print("\nSample indices to show:", sample_indices)

    PIL_images = []
    for i in sample_indices:
        arr = np.array(mm[i], copy=True)  # (C,H,W)
        pil = to_pil_from_CHW(arr)
        label_idx = int(labels[i])
        label_name = classes[label_idx] if label_idx < len(classes) else str(label_idx)
        title = f"idx={i} label={label_idx} ({label_name})"
        PIL_images.append((pil, title))
        if save_samples:
            os.makedirs(save_samples, exist_ok=True)
            out_path = Path(save_samples) / f"sample_{i}_lbl{label_idx}.png"
            pil.save(out_path)
            print("Saved sample to", out_path)

    # show using matplotlib
    cols = len(PIL_images)
    if cols > 0:
        fig, axs = plt.subplots(1, cols, figsize=(4*cols,4))
        if cols == 1:
            axs = [axs]
        for ax, (img, title) in zip(axs, PIL_images):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack_dir", required=True, help="Path to the s000 folder containing meta.json, images.dat, index.npy")
    ap.add_argument("--show", type=int, default=5, help="How many samples to display/save")
    ap.add_argument("--save_samples", default=None, help="Directory to save sample PNGs (optional)")
    args = ap.parse_args()
    inspect_pack(Path(args.pack_dir), show=args.show, save_samples=args.save_samples)