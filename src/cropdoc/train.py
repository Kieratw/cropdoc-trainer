import os
import json
import time
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
import matplotlib.pyplot as plt

from .models import MobileNetV3Dual

# =================== Stałe ===================
IMG_SIZE = 256
HEALTHY_TOKENS = {"healthy", "healthyleaf"}

# =================== Utils ===================
def set_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_state_dict_clean(model: torch.nn.Module):
    sd = model.state_dict()
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

class ShardPack(Dataset):
    """
    Loader paczek z build_dataset.py
    Struktura: root/<split>/s000/{images.dat, index.npy, meta.json}
    """
    def __init__(self, root: str, split: str):
        root = Path(root)
        part = root / split / "s000"
        if not part.exists():
            raise FileNotFoundError(f"Missing shard: {part.as_posix()}")
        meta = json.loads((part / "meta.json").read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        C, H, W = [int(x) for x in meta["shape"]]
        data_file = part / meta.get("data_file", "images.dat")
        labels_file = part / meta.get("labels_file", "index.npy")
        bytes_per = C * H * W  # uint8
        n = os.path.getsize(data_file) // bytes_per
        self.mm = np.memmap(data_file, dtype=np.uint8, mode="r", shape=(n, C, H, W))
        self.y = np.load(labels_file, mmap_mode="r")
        assert len(self.mm) == len(self.y), "images.dat and index.npy size mismatch"

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        arr = np.array(self.mm[i], copy=True)  # CHW uint8
        x = torch.from_numpy(arr).float() / 255.0  # CHW float32
        y = int(self.y[i])
        return x, y

class RemapTargets(Dataset):
    """
    Umożliwia przesunięcie etykiet (np. dodanie healthy=0 przed listą klas CLS).
    """
    def __init__(self, base: Dataset, add: int = 0):
        self.base = base
        self.add = int(add)
        self.classes = getattr(base, "classes", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y) + self.add

def try_compile(model: torch.nn.Module, force_disable: bool = False):
    if force_disable:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        import triton  # noqa: F401
        return torch.compile(model, mode="max-autotune")
    except Exception:
        try:
            return torch.compile(model, backend="eager")
        except Exception:
            return model

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item())

def train_epoch(model, loader, loss_fn, device, opt=None):
    model.train() if opt is not None else model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        lg_bin, lg_cls = model(x)
        loss, logits = loss_fn((lg_bin, lg_cls), y)
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        bs = y.size(0)
        tot_loss += float(loss.item()) * bs
        tot_acc  += accuracy(logits, y) * bs
        n += bs
    return (tot_loss / max(1, n)), (tot_acc / max(1, n))

def plot_training_history(history: dict, out_dir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["bin_loss"], label="BIN loss")
    ax[0].plot(history["cls_loss"], label="CLS loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(history["bin_acc"], label="BIN train acc")
    ax[1].plot(history["bin_val_acc"], label="BIN val acc")
    ax[1].plot(history["cls_acc"], label="CLS train acc")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(); ax[1].grid(True)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_summary.png", dpi=150)
    plt.close(fig)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# =================== Confusion matrix & eval ===================
import itertools

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

def evaluate_loader(model, loader, device, head: str, num_classes: int, class_names: List[str]) -> dict:
    """
    head ∈ {"bin","cls"}
    """
    model.eval()
    all_preds, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            lg_bin, lg_cls = model(x)
            lg = lg_bin if head == "bin" else lg_cls
            p = lg.argmax(1)
            all_preds.append(p.cpu())
            all_t.append(y.cpu())
    if not all_preds:
        return {"acc": 0.0, "cm": np.zeros((num_classes, num_classes), dtype=np.int64)}
    preds = torch.cat(all_preds)
    t = torch.cat(all_t)
    cm = confusion_matrix_torch(preds, t, num_classes=num_classes).numpy()
    acc = float((preds == t).float().mean().item())
    return {"acc": acc, "cm": cm}

# =================== Balans BIN (opcjonalne undersampling) ===================
def make_bin_balanced_subset(ds: Dataset, ratio: float) -> Dataset:
    """
    ratio: #diseased ≈ ratio * #healthy (1.0 = tyle samo)
    Jeżeli ratio <= 0 lub >= duże → zwracamy oryginał.
    """
    if ratio is None or ratio <= 0 or ratio >= 9e9:
        return ds
    ys = np.array([int(ds[i][1]) for i in range(len(ds))])
    idx_h = np.where(ys == 0)[0]  # healthy zakładamy index 0 w BIN packs
    idx_d = np.where(ys == 1)[0]
    if len(idx_h) == 0 or len(idx_d) == 0:
        return ds
    target_d = int(min(len(idx_d), round(len(idx_h) * ratio)))
    if target_d <= 0 or target_d >= len(idx_d):
        return ds
    rng = np.random.default_rng(1337)
    keep_d = rng.choice(idx_d, size=target_d, replace=False)
    keep = np.concatenate([idx_h, keep_d])
    rng.shuffle(keep)
    return Subset(ds, keep.tolist())

# =================== Dodatki: healthy z BIN dla CLS ===================
def subset_by_label(ds: Dataset, label: int, max_items: Optional[int] = None, seed: int = 1337) -> Dataset:
    """
    Zwraca Subset zawierający tylko próbki z etykietą == label.
    Jeśli max_items podane, losowo ogranicza do tej liczby.
    """
    idx = [i for i in range(len(ds)) if int(ds[i][1]) == int(label)]
    if max_items is not None and len(idx) > max_items:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=int(max_items), replace=False).tolist()
    return Subset(ds, idx)

# =================== Opcja B: CLS z kotwicą BIN ===================
def infinite_loader(loader):
    """Nieskończony generator po DataLoaderze."""
    while True:
        for batch in loader:
            yield batch

def train_epoch_cls_with_bin_anchor(
    model: torch.nn.Module,
    dl_cls: DataLoader,
    dl_bin_anchor: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    loss_ce: nn.Module,
    bin_anchor_w: float = 0.3,
):
    """
    Trenuje CLS, ale w każdej iteracji dokładamy mały składnik straty BIN (kotwica),
    by nie 'zapominał' po fazie BIN. Zwraca: (średni loss CLS, acc CLS).
    """
    model.train()
    tot_loss_cls = 0.0
    tot_acc_cls  = 0.0
    n = 0

    bin_iter = infinite_loader(dl_bin_anchor) if bin_anchor_w and bin_anchor_w > 0 else None

    for x_cls, y_cls in dl_cls:
        x_cls = x_cls.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y_cls = y_cls.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        _, lg_cls = model(x_cls)
        loss_cls = loss_ce(lg_cls, y_cls)

        if bin_iter is not None:
            x_bin, y_bin = next(bin_iter)
            x_bin = x_bin.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y_bin = y_bin.to(device, non_blocking=True)
            lg_bin_b, _ = model(x_bin)
            loss_bin_anchor = loss_ce(lg_bin_b, y_bin)
            loss = loss_cls + float(bin_anchor_w) * loss_bin_anchor
        else:
            loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        bs = y_cls.size(0)
        tot_loss_cls += float(loss_cls.item()) * bs            # raportujemy czysty loss CLS
        preds_cls = lg_cls.argmax(1)
        tot_acc_cls  += float((preds_cls == y_cls).float().mean().item()) * bs
        n += bs

    return (tot_loss_cls / max(1, n)), (tot_acc_cls / max(1, n))

# =================== Balans CLS (równa liczebność chorób) ===================
def labels_of_dataset(ds: Dataset) -> np.ndarray:
    """
    Zwraca tablicę etykiet dla dowolnego Datasetu, który w __getitem__ zwraca (x,y).
    Działa też dla Subset i ConcatDataset.
    """
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = ds.indices
        ys_base = labels_of_dataset(base)
        return ys_base[idxs]
    if isinstance(ds, ConcatDataset):
        parts = [labels_of_dataset(p) for p in ds.datasets]
        return np.concatenate(parts) if len(parts) else np.array([], dtype=np.int64)
    ys = np.empty(len(ds), dtype=np.int64)
    for i in range(len(ds)):
        _, y = ds[i]
        ys[i] = int(y)
    return ys

def make_cls_balanced_subset(
    ds: Dataset,
    num_classes: int,
    healthy_idx: int,
    seed: int = 1337,
    cap_healthy: bool = True,
) -> Dataset:
    """
    Wyrównuje liczebność klas CHORÓB (wszystko poza healthy_idx) do wspólnego minimum,
    a healthy przycina opcjonalnie do sumy próbek chorób (żeby nie zalały zbioru).
    Zwraca Subset z równym rozkładem chorób.
    """
    ys = labels_of_dataset(ds)
    rng = np.random.default_rng(seed)

    per_class = {c: np.where(ys == c)[0] for c in range(num_classes)}
    disease_classes = [c for c in range(num_classes) if c != healthy_idx]
    if not disease_classes:
        return ds

    min_d = min(len(per_class[c]) for c in disease_classes)
    if min_d == 0:
        return ds

    keep = []
    for c in disease_classes:
        idxs = per_class[c]
        if len(idxs) > min_d:
            idxs = rng.choice(idxs, size=min_d, replace=False)
        keep.append(idxs)

    if healthy_idx in per_class:
        idx_h = per_class[healthy_idx]
        if cap_healthy:
            target_h = sum(len(x) for x in keep)
            if len(idx_h) > target_h:
                idx_h = rng.choice(idx_h, size=target_h, replace=False)
        keep.append(idx_h)

    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return Subset(ds, keep.tolist())

# =================== Główna procedura ===================
def run(args):
    """
    Oczekuje argumentów przekazywanych przez __main__.py:
      --bin, --cls, --out, --epochs, --batch, --lr, --workers, --seed, --no_compile, --device, --bin_undersample_ratio
      (opcjonalnie: --bin_anchor_w w __main__.py)
    """
    set_seeds(getattr(args, "seed", 1337))
    device = torch.device(getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_total = int(getattr(args, "epochs", 40))
    batch = int(getattr(args, "batch", 64))
    lr = float(getattr(args, "lr", 2e-4))
    num_workers = int(getattr(args, "workers", 0))
    pin_memory = True
    unders_ratio = float(getattr(args, "bin_undersample_ratio", 1.0))
    bin_anchor_w = float(getattr(args, "bin_anchor_w", 0.3))  # 0.0 wyłącza kotwicę

    # ---- Załaduj paczki ----
    ds_bin_tr = ShardPack(args.bin, "train")
    try:
        ds_bin_va = ShardPack(args.bin, "val")
    except FileNotFoundError:
        val_len = max(1, int(len(ds_bin_tr) * 0.10))
        tr_len = len(ds_bin_tr) - val_len
        ds_bin_tr, ds_bin_va = random_split(ds_bin_tr, [tr_len, val_len])

    ds_cls_tr = ShardPack(args.cls, "train")
    try:
        ds_cls_va = ShardPack(args.cls, "val")
    except FileNotFoundError:
        ds_cls_va = None

    # ---- CLS: zdrowa klasa z metki; jeśli brak, dokładamy z BIN ----
    classes_cls = ds_cls_tr.classes
    healthy_idx_in_cls = next(
        (i for i, c in enumerate(classes_cls) if c.strip().lower() in HEALTHY_TOKENS),
        None
    )

    added_healthy = False  # <-- czy dołożyliśmy „healthy” z BIN
    if healthy_idx_in_cls is not None:
        classes_full = classes_cls
        healthy_idx = healthy_idx_in_cls
        ds_cls_wrapped_tr = ds_cls_tr
        ds_cls_wrapped_va = ds_cls_va
        cls_offset = 0
    else:
        HEALTHY_TOKEN = "healthy"
        classes_full = [HEALTHY_TOKEN] + classes_cls
        healthy_idx = 0
        cls_offset = 1
        added_healthy = True

        ds_cls_shifted_tr = RemapTargets(ds_cls_tr, add=cls_offset)
        ds_cls_shifted_va = RemapTargets(ds_cls_va, add=cls_offset) if ds_cls_va is not None else None

        # healthy z BIN (label 0), balansuj do skali CLS
        max_h_tr = len(ds_cls_tr)
        healthy_from_bin_tr = subset_by_label(ds_bin_tr, label=0, max_items=max_h_tr)
        if ds_cls_va is not None:
            max_h_va = len(ds_cls_va)
            healthy_from_bin_va = subset_by_label(ds_bin_va, label=0, max_items=max_h_va)
        else:
            healthy_from_bin_va = None

        ds_cls_wrapped_tr = ConcatDataset([healthy_from_bin_tr, ds_cls_shifted_tr])
        ds_cls_wrapped_va = (ConcatDataset([healthy_from_bin_va, ds_cls_shifted_va])
                             if (healthy_from_bin_va is not None and ds_cls_shifted_va is not None)
                             else (ds_cls_shifted_va if ds_cls_shifted_va is not None else healthy_from_bin_va))

    # sanity: brak duplikatów nazw klas
    if len(set(classes_full)) != len(classes_full):
        raise SystemExit(f"Duplet klas w CLS: {classes_full}")

    # >>> Balansuj klasy chorób w CLS, healthy przytnij do sumy chorób <<<
    ds_cls_wrapped_tr = make_cls_balanced_subset(
        ds=ds_cls_wrapped_tr,
        num_classes=len(classes_full),
        healthy_idx=healthy_idx,
        seed=1337,
        cap_healthy=True
    )

    # ---- BIN: opcjonalny undersampling diseased ----
    ds_bin_tr = make_bin_balanced_subset(ds_bin_tr, unders_ratio)

    # ---- DataLoaders ----
    dl_bin_tr = DataLoader(ds_bin_tr, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    dl_bin_va = DataLoader(ds_bin_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    dl_cls_tr = DataLoader(ds_cls_wrapped_tr, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    dl_cls_va = DataLoader(ds_cls_wrapped_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if ds_cls_wrapped_va is not None else None

    # ---- Model ----
    model = MobileNetV3Dual(num_classes=len(classes_full), pretrained=True, healthy_idx=healthy_idx)
    # odrobina regularizacji na głowie BIN
    model.head_bin = nn.Sequential(nn.Dropout(0.2), model.head_bin)
    model = model.to(device).to(memory_format=torch.channels_last)
    model = try_compile(model, force_disable=getattr(args, "no_compile", False))

    # ---- Lossy ----
    loss_ce = nn.CrossEntropyLoss()
    def loss_bin(out_pair, y):
        lg, _ = out_pair
        return loss_ce(lg, y), lg
    def loss_cls(out_pair, y):
        _, lg = out_pair
        return loss_ce(lg, y), lg

    # ---- Historia ----
    history = {"bin_loss": [], "bin_acc": [], "bin_val_acc": [], "cls_loss": [], "cls_acc": []}

    # ---- Trening: najpierw BIN (≈50%), potem CLS (reszta + kotwica BIN) ----
    ep_bin = max(1, int(round(0.5 * epochs_total)))
    ep_cls = max(1, epochs_total - ep_bin)

    # BIN + early stopping
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_bin)
    best_val = -1.0
    patience = 7
    no_imp = 0

    for ep in range(ep_bin):
        tl, ta = train_epoch(model, dl_bin_tr, loss_bin, device, opt)
        history["bin_loss"].append(tl)
        history["bin_acc"].append(ta)

        # walidacja BIN
        model.eval()
        with torch.no_grad():
            v_acc = 0.0
            n = 0
            for x, y in dl_bin_va:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                lg, _ = model(x)
                bs = y.size(0)
                v_acc += accuracy(lg, y) * bs
                n += bs
            v_acc = v_acc / max(1, n)
        history["bin_val_acc"].append(v_acc)

        scheduler.step()
        if v_acc > best_val:
            best_val = v_acc
            no_imp = 0
            torch.save({
                "model": get_state_dict_clean(model),
                "classes": classes_full,
                "healthy_idx": healthy_idx,
                "img_size": IMG_SIZE
            }, out_dir / "best_bin.pt")
        else:
            no_imp += 1

        print(f"[BIN {ep+1}/{ep_bin}] loss={tl:.4f} acc={ta:.4f} val_acc={v_acc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        if no_imp >= patience:
            print(f"Early stopping BIN at epoch {ep+1}")
            break

    # CLS z kotwicą BIN (opcja B)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(ep_cls):
        tl, ta = train_epoch_cls_with_bin_anchor(
            model=model,
            dl_cls=dl_cls_tr,
            dl_bin_anchor=dl_bin_tr,
            device=device,
            opt=opt,
            loss_ce=loss_ce,
            bin_anchor_w=bin_anchor_w,
        )
        history["cls_loss"].append(tl)
        history["cls_acc"].append(ta)
        print(f"[CLS {ep+1}/{ep_cls}] loss={tl:.4f} acc={ta:.4f}")

    # Kontrola driftu BIN po CLS
    post_bin = evaluate_loader(model, dl_bin_va, device, head="bin", num_classes=2, class_names=["healthy","diseased"])
    print(f"[POST-CLS BIN] val_acc={post_bin['acc']:.4f}")

    # ---- Zapis finalny ----
    ckpt = {
        "model": get_state_dict_clean(model),
        "classes": classes_full,
        "healthy_idx": healthy_idx,
        "img_size": IMG_SIZE,
        "time_s": round(time.time(), 2),
        "cls_offset": 1 if added_healthy else 0,  # <-- poprawne źródło prawdy
    }
    torch.save(ckpt, out_dir / "final.pt")
    print(f"Saved to {out_dir/'final.pt'}")

    # ---- Ewaluacja: macierze na walidacji ----
    bin_eval = evaluate_loader(model, dl_bin_va, device, head="bin", num_classes=2, class_names=["healthy", "diseased"])
    np.savetxt(out_dir / "confusion_bin_val.csv", bin_eval["cm"], fmt="%d", delimiter=",")
    save_confusion(bin_eval["cm"], ["healthy", "diseased"], f"BIN val acc={bin_eval['acc']:.3f}", out_dir / "confusion_bin_val.png", normalize=False)
    save_confusion(bin_eval["cm"], ["healthy", "diseased"], f"BIN val (norm) acc={bin_eval['acc']:.3f}", out_dir / "confusion_bin_val_norm.png", normalize=True)
    print(f"[EVAL BIN] val_acc={bin_eval['acc']:.4f}")

    if dl_cls_va is not None:
        cls_eval = evaluate_loader(model, dl_cls_va, device, head="cls", num_classes=len(classes_full), class_names=classes_full)
        np.savetxt(out_dir / "confusion_cls_val.csv", cls_eval["cm"], fmt="%d", delimiter=",")
        save_confusion(cls_eval["cm"], classes_full, f"CLS val acc={cls_eval['acc']:.3f}", out_dir / "confusion_cls_val.png", normalize=False)
        save_confusion(cls_eval["cm"], classes_full, f"CLS val (norm) acc={cls_eval['acc']:.3f}", out_dir / "confusion_cls_val_norm.png", normalize=True)
        print(f"[EVAL CLS] val_acc={cls_eval['acc']:.4f}")

    # ---- Wykresy ----
    plot_training_history(history, out_dir)
    return 0
