import argparse, os, json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ========= Model =========
def _get_mobilenet_v3_small(pretrained: bool = False):
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    except Exception:
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(pretrained=pretrained)
    features = m.features
    last_ch = 576
    return features, last_ch

class MobileNetV3Dual(nn.Module):
    """Dwie głowice: bin (healthy/diseased) i cls (wieloklasowa)."""
    def __init__(self, num_classes: int, pretrained: bool = False, healthy_idx: int = 0):
        super().__init__()
        self.features, last_ch = _get_mobilenet_v3_small(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn = nn.BatchNorm1d(last_ch)
        self.relu = nn.ReLU(inplace=True)
        # >>> ważne: dokładnie jak w treningu: Dropout(0.2) + Linear <<<
        self.head_bin = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_ch, 2))
        self.head_cls = nn.Linear(last_ch, num_classes)
        self.healthy_idx = healthy_idx

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        return self.head_bin(x), self.head_cls(x)


def _normalize_state_dict(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


# ========= Confusion & plots =========
def confusion_matrix_torch(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def save_confusion(cm: np.ndarray, class_names: List[str], title: str, out_png: Path, normalize: bool = False):
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            M = np.nan_to_num(cmn)
    else:
        M = cm
    import itertools
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = M.max() / 2.0 if M.size else 0
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        val = M[i, j]
        plt.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                 ha="center", va="center",
                 color="white" if val > thresh else "black", fontsize=8)
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Predykcja")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# ========= Test loaders z test_images =========
class _MapLabelsToFull(Dataset):
    """Owija ImageFolder i mapuje lokalne etykiety na indeksy wg classes_full."""
    def __init__(self, base: ImageFolder, classes_full: List[str]):
        self.base = base
        self.local_to_full = {i: classes_full.index(name) for i, name in enumerate(base.classes) if name in classes_full}
        self.keep_idx = [i for i, (_, y) in enumerate(base.samples) if y in self.local_to_full]
    def __len__(self): return len(self.keep_idx)
    def __getitem__(self, i):
        real_i = self.keep_idx[i]
        x, y_local = self.base[real_i]
        return x, int(self.local_to_full[y_local])

class _MapLabelsBIN(Dataset):
    """Wymusza healthy=0, diseased=1 (niezależnie od alfabetu)."""
    NAME_TO_BIN = {"healthy": 0, "diseased": 1}
    def __init__(self, base: ImageFolder):
        self.base = base
        self.local_to_bin = {i: self.NAME_TO_BIN.get(name, None) for i, name in enumerate(base.classes)}
        self.keep_idx = [i for i, (_, y) in enumerate(base.samples) if self.local_to_bin.get(y, None) is not None]
    def __len__(self): return len(self.keep_idx)
    def __getitem__(self, i):
        real_i = self.keep_idx[i]
        x, y_local = self.base[real_i]
        return x, int(self.local_to_bin[y_local])

def _find_test_root(data_root: Path) -> Path:
    if data_root.name == "test_images":
        return data_root
    cand = data_root / "test_images"
    if cand.exists():
        return cand
    parent = data_root.parent
    if (parent / "test_images").exists():
        return (parent / "test_images")
    raise SystemExit(f"Nie znaleziono test_images pod: {data_root.as_posix()}")

def _transform(size: int):
    # jak w builderze: resize z zachowaniem proporcji + centralny crop 256
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])

def _make_test_loader_bin(test_bin_dir: Path, size: int, batch: int, workers: int):
    if not test_bin_dir.exists(): return None
    base = ImageFolder(test_bin_dir.as_posix(), transform=_transform(size))
    ds = _MapLabelsBIN(base)
    if len(ds) == 0: return None
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

def _make_test_loader_cls(test_cls_dir: Path, classes_full: List[str], size: int, batch: int, workers: int):
    if not test_cls_dir.exists(): return None
    base = ImageFolder(test_cls_dir.as_posix(), transform=_transform(size))
    ds = _MapLabelsToFull(base, classes_full=classes_full)
    if len(ds) == 0: return None
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)


# ========= Pipeline eval (BIN→CLS) =========
def _final_space(classes_full: List[str], healthy_idx: Optional[int]) -> Tuple[List[str], int, List[int]]:
    """
    Zwraca:
      - final_names: ["healthy"] + choroby (bez healthy z CLS)
      - final_healthy = 0
      - disease_indices_in_cls: indeksy klas chorób w przestrzeni CLS
    """
    if healthy_idx is None or healthy_idx < 0 or healthy_idx >= len(classes_full):
        diseases = list(range(len(classes_full)))
        final_names = ["healthy"] + [classes_full[i] for i in diseases]
        disease_indices_in_cls = diseases
    else:
        diseases = [i for i in range(len(classes_full)) if i != healthy_idx]
        final_names = ["healthy"] + [classes_full[i] for i in diseases]
        disease_indices_in_cls = diseases
    return final_names, 0, disease_indices_in_cls

@torch.no_grad()
def eval_pipeline_BIN_then_CLS(
    model: nn.Module,
    device: torch.device,
    dl_bin_healthy: Optional[DataLoader],
    dl_cls: Optional[DataLoader],
    classes_full: List[str],
    healthy_idx: Optional[int],
    bin_threshold: float = 0.8,
) -> dict:
    """
    Predykcja:
      - jeśli BIN[p(healthy)] >= threshold → final = "healthy"
      - w p.p. final = argmax po chorobach z CLS
    """
    final_names, final_healthy, disease_idx_in_cls = _final_space(classes_full, healthy_idx)
    n_final_classes = len(final_names)

    all_preds, all_t = [], []
    n_samples = 0

    # 1) healthy z BIN
    if dl_bin_healthy is not None:
        for x, _ in dl_bin_healthy:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            lg_bin, lg_cls = model(x)
            prob_bin = torch.softmax(lg_bin, dim=1)[:, 0]  # p(healthy)
            is_healthy = prob_bin >= bin_threshold
            if not is_healthy.all():
                lg_sub = lg_cls[:, disease_idx_in_cls]
                arg = lg_sub.argmax(1) + 1  # +1 bo final space ma healthy=0
                pred = torch.where(is_healthy, torch.zeros_like(arg), arg)
            else:
                pred = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            all_preds.append(pred.cpu())
            all_t.append(torch.zeros(x.size(0), dtype=torch.long))
            n_samples += x.size(0)

    # 2) choroby z CLS (healthy w CLS ignorujemy)
    if dl_cls is not None:
        for x, y_cls in dl_cls:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y_cls = y_cls.to(device, non_blocking=True)

            if healthy_idx is not None and 0 <= healthy_idx < len(classes_full):
                mask_d = (y_cls != healthy_idx)
            else:
                mask_d = torch.ones_like(y_cls, dtype=torch.bool)

            if mask_d.any():
                x_d = x[mask_d]
                y_d = y_cls[mask_d]

                lg_bin, lg_cls = model(x_d)
                prob_bin = torch.softmax(lg_bin, dim=1)[:, 0]
                lg_sub = lg_cls[:, disease_idx_in_cls]
                arg = lg_sub.argmax(1) + 1  # 1..K
                is_healthy = prob_bin >= bin_threshold
                pred = torch.where(is_healthy, torch.zeros_like(arg), arg)

                cls_to_final = {cls_i: fi+1 for fi, cls_i in enumerate(disease_idx_in_cls)}
                y_final = torch.tensor([cls_to_final[int(t.item())] for t in y_d], dtype=torch.long)

                all_preds.append(pred.cpu())
                all_t.append(y_final.cpu())
                n_samples += x_d.size(0)

    if not all_preds:
        return {
            "acc": 0.0,
            "cm": np.zeros((n_final_classes, n_final_classes), dtype=np.int64),
            "n": 0,
            "final_names": final_names,
            "healthy_idx_final": final_healthy,
            "metrics": {}
        }

    preds = torch.cat(all_preds)
    t = torch.cat(all_t)

    cm = confusion_matrix_torch(preds, t, num_classes=n_final_classes).numpy()
    acc = float((preds == t).float().mean().item())

    # sub-metryki
    metrics = {}
    tp_h = cm[final_healthy, final_healthy]
    fp_h = cm[:, final_healthy].sum() - tp_h
    fn_h = cm[final_healthy, :].sum() - tp_h
    prec_h = float(tp_h / (tp_h + fp_h)) if (tp_h + fp_h) > 0 else 0.0
    rec_h  = float(tp_h / (tp_h + fn_h)) if (tp_h + fn_h) > 0 else 0.0
    metrics["healthy_precision"] = prec_h
    metrics["healthy_recall"] = rec_h

    disease_rows = list(range(1, n_final_classes))
    if disease_rows:
        mask = torch.isin(t, torch.tensor(disease_rows))
        acc_diseases = float((preds[mask] == t[mask]).float().mean().item()) if mask.any() else None
        metrics["diseases_acc"] = acc_diseases
        f1s = []
        for k in disease_rows:
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        metrics["diseases_macro_f1"] = float(np.mean(f1s)) if f1s else None

    return {
        "acc": acc,
        "cm": cm,
        "n": int(t.numel()),
        "final_names": final_names,
        "healthy_idx_final": final_healthy,
        "metrics": metrics
    }


# ========= Main =========
def main():
    ap = argparse.ArgumentParser("Ewaluacja dual-head (BIN→CLS) na test_images/ w trybie jak w apce")
    ap.add_argument("--ckpt", required=True, help="Ścieżka do final.pt (lub best_bin.pt)")
    ap.add_argument("--data_root", required=True, help="Root datasetu (zawiera test_images/) lub sam test_images/")
    ap.add_argument("--out", default=None, help="Folder wyjściowy na wyniki (domyślnie: <folder_ckpt>/eval)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--bin_threshold", type=float, default=0.8, help="Próg p(healthy) z BIN dla decyzji finalnej")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    if "model" not in ck or "classes" not in ck:
        raise SystemExit("Checkpoint nie zawiera kluczy 'model'/'classes'.")

    classes_full: List[str] = ck["classes"]
    healthy_idx: Optional[int] = int(ck.get("healthy_idx", 0)) if "healthy_idx" in ck else None
    img_size_ckpt: int = int(ck.get("img_size", args.size))
    size = img_size_ckpt if img_size_ckpt else args.size

    # model
    model = MobileNetV3Dual(num_classes=len(classes_full), pretrained=False, healthy_idx=(healthy_idx or 0))
    sd = _normalize_state_dict(ck["model"])
    # dzięki identycznej architekturze BIN (Dropout+Linear) klucze pasują 1:1
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    # test roots
    test_root = _find_test_root(Path(args.data_root))
    test_bin_dir = test_root / "bin"
    test_cls_dir = test_root / "cls"

    # out
    out_dir = Path(args.out) if args.out else (Path(args.ckpt).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loaders do pipeline’u
    dl_bin_healthy = None
    if test_bin_dir.exists():
        base_bin = ImageFolder(test_bin_dir.as_posix(), transform=_transform(size))
        if "healthy" in base_bin.classes:
            idx_healthy_local = base_bin.classes.index("healthy")
            keep_idx = [i for i, (_, y) in enumerate(base_bin.samples) if y == idx_healthy_local]
            class _OnlyIdx(Dataset):
                def __init__(self, base, idxs): self.base, self.idxs = base, idxs
                def __len__(self): return len(self.idxs)
                def __getitem__(self, i): return self.base[self.idxs[i]]
            ds_h = _OnlyIdx(base_bin, keep_idx)
            if len(ds_h) > 0:
                dl_bin_healthy = DataLoader(ds_h, batch_size=args.batch, shuffle=False,
                                            num_workers=args.workers, pin_memory=True)

    dl_cls = _make_test_loader_cls(test_cls_dir, classes_full=classes_full, size=size,
                                   batch=args.batch, workers=args.workers)

    # 1) Pipeline eval jak w apce
    pipe = eval_pipeline_BIN_then_CLS(
        model=model,
        device=device,
        dl_bin_healthy=dl_bin_healthy,
        dl_cls=dl_cls,
        classes_full=classes_full,
        healthy_idx=healthy_idx,
        bin_threshold=args.bin_threshold,
    )

    # zapisz wyniki pipeline
    final_names = pipe["final_names"]
    np.savetxt(out_dir / "confusion_pipeline.csv", pipe["cm"], fmt="%d", delimiter=",")
    save_confusion(pipe["cm"], final_names, f"PIPE acc={pipe['acc']:.3f}", out_dir / "confusion_pipeline.png", normalize=False)
    save_confusion(pipe["cm"], final_names, f"PIPE (norm) acc={pipe['acc']:.3f}", out_dir / "confusion_pipeline_norm.png", normalize=True)

    summary = {
        "pipeline": {
            "acc": pipe["acc"],
            "n": pipe["n"],
            "classes": final_names,
            "metrics": pipe["metrics"],
            "bin_threshold": args.bin_threshold
        }
    }

    # 2) Surowe osobne metryki (opcjonalnie)
    if test_bin_dir.exists():
        dl_bin_all = _make_test_loader_bin(test_bin_dir, size=size, batch=args.batch, workers=args.workers)
        if dl_bin_all is not None:
            with torch.no_grad():
                all_preds, all_t = [], []
                for x, y in dl_bin_all:
                    x = x.to(device).to(memory_format=torch.channels_last)
                    lg_bin, _ = model(x)
                    p = lg_bin.argmax(1)
                    all_preds.append(p.cpu()); all_t.append(y.cpu())
                if all_preds:
                    pr = torch.cat(all_preds); gt = torch.cat(all_t)
                    cm = confusion_matrix_torch(pr, gt, 2).numpy()
                    acc = float((pr == gt).float().mean().item())
                    summary["bin_raw"] = {"acc": acc, "n": int(gt.numel())}
                    np.savetxt(out_dir / "confusion_bin_test_raw.csv", cm, fmt="%d", delimiter=",")
                    save_confusion(cm, ["healthy","diseased"], f"BIN raw acc={acc:.3f}", out_dir / "confusion_bin_test_raw.png", normalize=False)

    if dl_cls is not None:
        with torch.no_grad():
            all_preds, all_t = [], []
            for x, y in dl_cls:
                x = x.to(device).to(memory_format=torch.channels_last)
                _, lg_cls = model(x)
                p = lg_cls.argmax(1)
                all_preds.append(p.cpu()); all_t.append(y.cpu())
            if all_preds:
                pr = torch.cat(all_preds); gt = torch.cat(all_t)
                cm = confusion_matrix_torch(pr, gt, num_classes=len(classes_full)).numpy()
                acc = float((pr == gt).float().mean().item())
                summary["cls_raw"] = {"acc": acc, "n": int(gt.numel())}
                np.savetxt(out_dir / "confusion_cls_test_raw.csv", cm, fmt="%d", delimiter=",")
                save_confusion(cm, classes_full, f"CLS raw acc={acc:.3f}", out_dir / "confusion_cls_test_raw.png", normalize=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PIPELINE] n={pipe['n']} acc={pipe['acc']:.4f}  -> {out_dir.as_posix()}")
    if 'metrics' in pipe:
        m = pipe['metrics']
        print(f"   healthy_precision={m.get('healthy_precision'):.3f}  healthy_recall={m.get('healthy_recall'):.3f}  "
              f"diseases_acc={m.get('diseases_acc')}  diseases_macro_f1={m.get('diseases_macro_f1')}")

if __name__ == "__main__":
    main()
