import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from tqdm import tqdm

import matplotlib.pyplot as plt

IM_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------- model -----------------
def create_model(arch: str, num_classes: int):
    a = arch.lower()
    if a in {"mobilenetv3", "mobilenetv3_large", "mobilenetv3-large"}:
        m = tvm.mobilenet_v3_large(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if a in {"convnext_tiny", "convnext-tiny", "convnext"}:
        m = tvm.convnext_tiny(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if a in {"efficientnet_b0", "efficientnet"}:
        m = tvm.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise SystemExit(f"Nieznana architektura: {arch}")


# ----------------- norm -----------------
def try_autodetect_meta(images_dir: Path) -> Optional[Path]:
    """
    Jeśli images_dir = .../<dataset_root>/test_images/cls,
    to meta szukamy w <dataset_root>/cls/train/s000/meta.json
    """
    try:
        if images_dir.name == "cls" and images_dir.parent.name == "test_images":
            dataset_root = images_dir.parent.parent
            candidate = dataset_root / "cls" / "train" / "s000" / "meta.json"
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


def load_norm(meta_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray, str]:
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        mean = np.array(meta.get("mean", IMAGENET_MEAN), dtype=np.float32)
        std = np.array(meta.get("std", IMAGENET_STD), dtype=np.float32)
        return mean, std, "meta.json"
    return (
        np.array(IMAGENET_MEAN, dtype=np.float32),
        np.array(IMAGENET_STD, dtype=np.float32),
        "imagenet",
    )


# ----------------- data -----------------
def read_images_from_folder(root: Path) -> List[Path]:
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IM_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs


def infer_has_labels(img_paths: List[Path], class_names: List[str]) -> bool:
    """
    Uznajemy, że mamy GT, jeśli każdy obraz jest w folderze o nazwie klasy.
    """
    for p in img_paths:
        if p.parent.name not in class_names:
            return False
    return True


class FolderDataset(Dataset):
    def __init__(
        self,
        img_paths: List[Path],
        class_to_idx: Dict[str, int],
        img_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        has_labels: bool,
    ):
        self.img_paths = img_paths
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.has_labels = has_labels

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
        arr = arr.transpose(2, 0, 1)  # [3,H,W]
        arr = (arr - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        x = self._load_image(p)
        if self.has_labels:
            y = self.class_to_idx[p.parent.name]
        else:
            y = -1
        return x, y, str(p)




class PathDataset(Dataset):
    """Dataset zwracający tylko etykietę (jeśli jest) i ścieżkę. Używany w trybie TTA."""

    def __init__(
        self,
        img_paths: List[Path],
        class_to_idx: Dict[str, int],
        has_labels: bool,
    ):
        self.img_paths = img_paths
        self.class_to_idx = class_to_idx
        self.has_labels = has_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        if self.has_labels:
            y = self.class_to_idx[p.parent.name]
        else:
            y = -1
        return y, str(p)

# ----------------- plotting: confusion -----------------
def plot_confusion(
    cm: np.ndarray,
    classes: List[str],
    out_path: Path,
    normalize: bool = False,
    title: Optional[str] = None,
    show: bool = True,
):
    cm_plot = cm.copy()
    if normalize:
        cm_plot = cm_plot.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
    else:
        cm_plot = cm_plot.astype(np.int64)
        fmt = "d"

    n = len(classes)
    side = max(6, min(1.0 * n, 16))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Prawdziwa klasa",
        xlabel="Predykcja",
    )
    if title:
        ax.set_title(title, pad=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(n):
        for j in range(n):
            val = format(cm_plot[i, j], fmt)
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Zapisano: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------- plotting: przykładowe obrazki -----------------
def make_gallery(
    examples,
    classes: List[str],
    out_path: Path,
    title: str,
    show: bool = True,
):
    """
    examples: lista dictów:
      {
        "path": <str>,
        "gt":   <int lub None>,
        "pred": <int>,
        "conf": <float>
      }
    """
    if not examples:
        return

    n = len(examples)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for idx, ex in enumerate(examples):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        p = Path(ex["path"])
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            ax.axis("off")
            continue

        ax.imshow(img)
        ax.axis("off")

        gt = ex.get("gt", None)
        pred = ex["pred"]
        conf = ex["conf"]

        if gt is not None and gt >= 0:
            gt_name = classes[int(gt)]
            pred_name = classes[int(pred)]
            ttl = f"GT: {gt_name}\nPred: {pred_name} ({conf:.2f})"
        else:
            pred_name = classes[int(pred)]
            ttl = f"Pred: {pred_name} ({conf:.2f})"

        ax.set_title(ttl, fontsize=8)

    # wyłącz puste osie
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Zapisano: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_visual_examples(
    save_dir: Path,
    paths: List[str],
    preds: np.ndarray,
    probs: np.ndarray,
    classes: List[str],
    gts: Optional[np.ndarray] = None,
    vis_limit: int = 40,
    show: bool = True,
):
    """
    Tworzy 1–2 galerie:
      - misclassified_examples.png (jeśli mamy GT i są błędy)
      - correct_examples.png (kilka poprawnych / top conf)
    """
    paths = list(paths)
    n = len(paths)
    if n == 0:
        return

    preds = np.asarray(preds, dtype=int)

    if gts is not None:
        gts = np.asarray(gts, dtype=int)
        wrong_idx = np.where(preds != gts)[0]
        correct_idx = np.where(preds == gts)[0]
    else:
        gts = None
        wrong_idx = np.arange(n)
        correct_idx = np.array([], dtype=int)

    # błędy – sortujemy po pewności malejąco
    if len(wrong_idx) > 0:
        wrong_conf = probs[wrong_idx, preds[wrong_idx]]
        order = np.argsort(-wrong_conf)
        wrong_idx = wrong_idx[order][:vis_limit]

        wrong_examples = []
        for i in wrong_idx:
            wrong_examples.append(
                {
                    "path": paths[i],
                    "gt": int(gts[i]) if gts is not None else None,
                    "pred": int(preds[i]),
                    "conf": float(probs[i, preds[i]]),
                }
            )

        out_wrong = save_dir / "misclassified_examples.png"
        make_gallery(wrong_examples, classes, out_wrong, title="Błędne predykcje", show=show)

    # poprawne – weźmy kilka najbardziej pewnych
    if len(correct_idx) > 0:
        corr_conf = probs[correct_idx, preds[correct_idx]]
        order = np.argsort(-corr_conf)
        correct_idx = correct_idx[order][:vis_limit]

        corr_examples = []
        for i in correct_idx:
            corr_examples.append(
                {
                    "path": paths[i],
                    "gt": int(gts[i]) if gts is not None else None,
                    "pred": int(preds[i]),
                    "conf": float(probs[i, preds[i]]),
                }
            )

        out_corr = save_dir / "correct_examples.png"
        make_gallery(corr_examples, classes, out_corr, title="Poprawne predykcje (najpewniejsze)", show=show)




# ----------------- TTA (opcjonalnie) -----------------
def _pil_to_tensor(img: Image.Image, img_size: int, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Dokładnie to samo pre-processing co w FolderDataset._load_image."""
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    arr = arr.transpose(2, 0, 1)  # [3,H,W]
    arr = (arr - mean[:, None, None]) / std[:, None, None]
    return torch.from_numpy(arr)


def _tta_stack_from_path(path: str, img_size: int, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Zwraca tensor [4,3,H,W] dla 4 wariantów TTA:
    1) oryginał -> resize
    2) flip horizontal -> resize
    3) center crop (80%) -> resize
    4) darken (80% jasności) -> resize
    """
    img0 = Image.open(path).convert("RGB")

    # 1) Original
    t1 = _pil_to_tensor(img0, img_size, mean, std)

    # 2) Flip Horizontal
    if hasattr(Image, "Transpose"):
        img_flip = img0.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        img_flip = img0.transpose(Image.FLIP_LEFT_RIGHT)
    t2 = _pil_to_tensor(img_flip, img_size, mean, std)

    # 3) Center Crop & Resize (80% oryginalnego obrazu)
    w, h = img0.size
    cw = max(1, int(round(0.8 * w)))
    ch = max(1, int(round(0.8 * h)))
    left = int(round((w - cw) / 2.0))
    top = int(round((h - ch) / 2.0))
    img_crop = img0.crop((left, top, left + cw, top + ch))
    t3 = _pil_to_tensor(img_crop, img_size, mean, std)

    # 4) Darken (80% jasności)
    img_dark = ImageEnhance.Brightness(img0).enhance(0.8)
    t4 = _pil_to_tensor(img_dark, img_size, mean, std)

    return torch.stack([t1, t2, t3, t4], dim=0)

# ----------------- eval -----------------
@torch.no_grad()
def run_eval(
    ckpt_path: Path,
    images_dir: Path,
    meta_path_cli: Optional[Path],
    batch: int,
    workers: int,
    device: str = "cuda",
    vis_limit: int = 40,
    tta: bool = False,
):
    # 1) Ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "convnext_tiny")
    img_size = int(ckpt.get("img_size", 380))
    print(f"CKPT: arch={arch}  classes={len(classes)}  img_size={img_size}")

    # 2) Norm (CLI meta > autodetect > ImageNet)
    meta_auto = try_autodetect_meta(images_dir) if not meta_path_cli else None
    meta_to_use = meta_path_cli or meta_auto
    mean, std, norm_src = load_norm(meta_to_use)
    if meta_path_cli:
        print(f"Norm: meta.json z CLI -> {meta_path_cli}")
    elif meta_auto:
        print(f"Norm: meta.json autodetected -> {meta_auto}")
    else:
        print("Norm: ImageNet fallback")

    # 3) Model
    model = create_model(arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    model = model.to(memory_format=torch.channels_last)
    # 4) Dane
    all_imgs = read_images_from_folder(images_dir)
    if not all_imgs:
        raise SystemExit(f"Brak obrazów w {images_dir}")

    has_labels = infer_has_labels(all_imgs, classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if tta:
        print(
            "TTA: włączone (oryginał + flip_horizontal + center_crop_80% + darken_80%), agregacja = średnia softmaxów"
        )
        ds = PathDataset(all_imgs, class_to_idx, has_labels)
    else:
        ds = FolderDataset(all_imgs, class_to_idx, img_size, mean, std, has_labels)

    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    # 5) Inference
    probs_all, preds_all, gts_all, paths_all = [], [], [], []
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    if tta:
        for yb, paths in tqdm(dl, desc="inference", ncols=120):
            bsz = len(paths)
            # [B,4,3,H,W] -> [B*4,3,H,W]
            tta_stack = torch.stack(
                [_tta_stack_from_path(p, img_size, mean, std) for p in paths], dim=0
            )
            xb = tta_stack.view(-1, 3, img_size, img_size)
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                probs_t = F.softmax(logits, dim=1).float()  # [B*4,C]

            probs_t = probs_t.view(bsz, 4, -1).mean(dim=1)  # [B,C]
            probs = probs_t.cpu().numpy()

            pred = np.argmax(probs, axis=1)
            probs_all.append(probs)
            preds_all.append(pred)
            gts_all.append(yb.numpy())
            paths_all.extend(list(paths))
    else:
        for xb, yb, paths in tqdm(dl, desc="inference", ncols=120):
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                probs = F.softmax(logits, dim=1).float().cpu().numpy()

            pred = np.argmax(probs, axis=1)
            probs_all.append(probs)
            preds_all.append(pred)
            gts_all.append(yb.numpy())
            paths_all.extend(list(paths))

    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)

    # 6) Save dir obok ckpt
    save_dir = ckpt_path.parent / "eval2"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Wyniki zapisuję do: {save_dir.resolve()}")
    preds_csv = save_dir / "preds.csv"
    preds_raw_jsonl = save_dir / "preds_raw.jsonl"
    classes_json = save_dir / "classes.json"
    cm_png = save_dir / "cm_counts.png"
    cmn_png = save_dir / "cm_norm.png"
    metrics_json = save_dir / "metrics.json"
    report_txt = save_dir / "report.txt"

    # 6.1) Zapisz mapę klas (przydatne do późniejszego scalania nazw)
    classes_payload = {
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "idx_to_class": {str(i): c for i, c in enumerate(classes)},
        "norm_source": norm_src,
        "img_size": img_size,
        "arch": arch,
        "ckpt": str(ckpt_path),
    }
    classes_json.write_text(json.dumps(classes_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {classes_json}")

    # 6.2) Zapisz surowe predykcje do JSONL (1 linia = 1 obraz)
    #      To pozwala później przeliczać macierz błędów po scaleniu klas bez ponownego inferencingu.
    with open(preds_raw_jsonl, "w", encoding="utf-8") as f:
        for path, gt, pred, prob_row in zip(paths_all, gts_all, preds_all, probs_all):
            rec = {
                "path": path,
                "y_true": int(gt) if int(gt) >= 0 else None,
                "y_true_name": classes[int(gt)] if int(gt) >= 0 else None,
                "y_pred": int(pred),
                "y_pred_name": classes[int(pred)],
                "probs": [float(x) for x in prob_row.tolist()],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Zapisano: {preds_raw_jsonl}")

    # 7) Metryki + CM + wizualizacje
    if has_labels:
        valid_idx = np.where(gts_all >= 0)[0]
        y_true = gts_all[valid_idx]
        y_pred = preds_all[valid_idx]
        probs_valid = probs_all[valid_idx]
        paths_valid = [paths_all[i] for i in valid_idx]

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"[ztest] ACC={acc:.4f}  F1(macro)={f1m:.4f}")

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        plot_confusion(
            cm,
            classes,
            cm_png,
            normalize=False,
            title="Confusion matrix (counts)",
            show=True,
        )
        plot_confusion(
            cm,
            classes,
            cmn_png,
            normalize=True,
            title="Confusion matrix (row-normalized)",
            show=True,
        )

        rep = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
        report_txt.write_text(rep, encoding="utf-8")
        print(f"Zapisano: {report_txt}")

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0
        )
        per_class = {}
        for i, cname in enumerate(classes):
            per_class[cname] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }

        metrics = {
            "acc": float(acc),
            "f1_macro": float(f1m),
            "classes": classes,
            "norm_source": norm_src,
            "img_size": img_size,
            "per_class": per_class,
        }
        metrics_json.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Zapisano: {metrics_json}")

        # przykładowe obrazki
        save_visual_examples(
            save_dir=save_dir,
            paths=paths_valid,
            preds=y_pred,
            probs=probs_valid,
            classes=classes,
            gts=y_true,
            vis_limit=vis_limit,
            show=True,
        )
    else:
        print(
            "[ztest] Brak etykiet w folderze. CM i metryki pomijam, zapisuję same predykcje + wizualizacje pred."
        )
        save_visual_examples(
            save_dir=save_dir,
            paths=paths_all,
            preds=preds_all,
            probs=probs_all,
            classes=classes,
            gts=None,
            vis_limit=vis_limit,
            show=True,
        )

    # 8) CSV z predykcjami
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred", "pred_name", "conf"] + [f"p_{c}" for c in classes])
        for path, pred, prob_row in zip(paths_all, preds_all, probs_all):
            pred_name = classes[int(pred)]
            conf = float(prob_row[int(pred)])
            w.writerow(
                [path, int(pred), pred_name, f"{conf:.6f}"]
                + [f"{float(p):.6f}" for p in prob_row]
            )
    print(f"Zapisano: {preds_csv}")


def parse_args():
    ap = argparse.ArgumentParser(
        "Auto-eval na folderze obrazów (metryki + confusion matrix + przykładowe obrazki)"
    )
    ap.add_argument("--ckpt", required=True, help="Ścieżka do best.pt")
    ap.add_argument(
        "--images",
        required=True,
        help="Folder z obrazami (images/<klasa>/*.jpg lub images/*.jpg)",
    )
    ap.add_argument(
        "--meta",
        default=None,
        help="Opcjonalny meta.json (nadpisuje autodetekcję; domyślnie bierze z train/s000/meta.json)",
    )
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--vis_limit",
        type=int,
        default=40,
        help="Maksymalna liczba obrazków w pojedynczej galerii (błędne / poprawne)",
    )

    ap.add_argument(
        "--tta",
        action="store_true",
        help="Włącza Test-Time Augmentation: oryginał, flip_horizontal, center crop 80% + resize, darken 80%. Wynik = średnia softmaxów.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        ckpt_path=Path(args.ckpt),
        images_dir=Path(args.images),
        meta_path_cli=Path(args.meta) if args.meta else None,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        vis_limit=args.vis_limit,
        tta=args.tta,
    )
