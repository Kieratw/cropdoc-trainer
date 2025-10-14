import os
import json
import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .models import MobileNetV3Dual

IMG_SIZE = 256
LR = 2e-4
EPOCHS = 20
HEALTHY_TOKEN = "healthy"


def set_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ShardPack(Dataset):
    """
    Czyta format z build_dataset.py:
      <root>/<split>/s000/{ images.dat, index.npy, meta.json }
    images.dat: surowy memmap (N, C, H, W) w uint8. N wyliczamy z rozmiaru pliku.
    index.npy: etykiety int64
    meta.json: {"classes": [...], "shape": [C,H,W], ...}
    """
    def __init__(self, root: str, split: str):
        root = Path(root)
        part = root / split / "s000"
        if not part.exists():
            raise FileNotFoundError(f"Brak shardu {part.as_posix()}")
        meta_p = part / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"Brak meta.json w {part.as_posix()}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        C, H, W = [int(x) for x in meta["shape"]]
        data_file = part / meta.get("data_file", "images.dat")
        labels_file = part / meta.get("labels_file", "index.npy")
        if not data_file.exists():
            raise FileNotFoundError(f"Nie znaleziono {data_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Nie znaleziono {labels_file}")

        # Wylicz N po rozmiarze pliku (uint8 => 1 bajt na element)
        item_bytes = C * H * W
        fsize = os.path.getsize(data_file)
        if fsize % item_bytes != 0:
            raise ValueError(
                f"images.dat rozmiar={fsize} nie jest wielokrotnością C*H*W={item_bytes}. "
                f"Sprawdź meta.json albo uszkodzony plik."
            )
        N = fsize // item_bytes
        if N <= 0:
            raise ValueError("Wyliczone N<=0; sprawdź dane.")

        self.mm = np.memmap(data_file, dtype=np.uint8, mode="r", shape=(N, C, H, W))
        self.y = np.load(labels_file, mmap_mode="r")
        if len(self.mm) != len(self.y):
            raise ValueError(f"Długości X={len(self.mm)} i y={len(self.y)} nie zgadzają się")

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        arr = np.array(self.mm[i], copy=True)  # (C,H,W) uint8; kopia żeby uniknąć non-writable
        x = torch.from_numpy(arr).float() / 255.0
        y = int(self.y[i])
        return x, y


class RemapTargets(Dataset):
    """
    Owijką datasetu do przesunięcia etykiet o stałą (np. gdy dodajemy 'healthy' na indeksie 0).
    """
    def __init__(self, base: Dataset, add: int = 0):
        self.base = base
        self.add = int(add)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, int(y) + self.add


def try_compile(model: torch.nn.Module, force_disable: bool = False):
    if force_disable:
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
    return (pred == targets).float().mean().item()


def train_epoch(model, loader, loss_fn, device, opt=None):
    model.train() if opt is not None else model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        lg_bin, lg_cls = model(x)
        loss, logits = loss_fn((lg_bin, lg_cls), y)
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        bs = y.shape[0]
        tot_loss += loss.item() * bs
        tot_acc += accuracy(logits, y) * bs
        n += bs
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


def get_state_dict_clean(model: torch.nn.Module) -> dict:
    """
    Zwraca czysty state_dict bez prefiksów po torch.compile() i DataParallel.
    """
    base = getattr(model, "_orig_mod", model)
    base = getattr(base, "module", base)
    return base.state_dict()


def run(args):
    set_seeds(getattr(args, "seed", 1337))
    device = torch.device(getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Wczytanie shardów
    ds_bin_tr = ShardPack(args.bin, split="train")
    ds_bin_va = ShardPack(args.bin, split="val")
    ds_cls_tr = ShardPack(args.cls, split="train")

    # Klasy i ewentualne dodanie 'healthy' na indeks 0
    classes_cls = ds_cls_tr.classes
    if HEALTHY_TOKEN in classes_cls:
        classes_full = classes_cls
        healthy_idx = classes_full.index(HEALTHY_TOKEN)
        cls_offset = 0
        ds_cls_tr_wrapped = ds_cls_tr
    else:
        classes_full = [HEALTHY_TOKEN] + classes_cls
        healthy_idx = 0
        cls_offset = 1
        ds_cls_tr_wrapped = RemapTargets(ds_cls_tr, add=1)  # kluczowa poprawka

    # Loaders
    num_workers = getattr(args, "workers", 0)
    batch = getattr(args, "batch", 64)
    pin = device.type == "cuda"

    dl_bin_tr = DataLoader(ds_bin_tr, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_bin_va = DataLoader(ds_bin_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_cls_tr = DataLoader(ds_cls_tr_wrapped, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin)

    # Model
    model = MobileNetV3Dual(num_classes=len(classes_full), pretrained=True, healthy_idx=healthy_idx).to(device).to(
        memory_format=torch.channels_last
    )
    model = try_compile(model, force_disable=getattr(args, "no_compile", False))
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args, "lr", LR))
    loss_ce = nn.CrossEntropyLoss()

    def loss_bin(out_pair, y_bin):
        lg_bin, _ = out_pair
        return loss_ce(lg_bin, y_bin), lg_bin

    def loss_cls(out_pair, y_cls):
        _, lg_cls = out_pair
        return loss_ce(lg_cls, y_cls), lg_cls

    # Faza 1: BIN (zamrażamy feature extractor, BN w trybie train dla stabilności)
    for p in model.features.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()

    best_val = -1.0
    epochs = getattr(args, "epochs", EPOCHS)
    t0 = time.time()
    for ep in range(epochs):
        tl, _ = train_epoch(model, dl_bin_tr, loss_bin, device, opt)
        vl, va = train_epoch(model, dl_bin_va, loss_bin, device, opt=None)
        if va > best_val:
            best_val = va
            torch.save(
                {
                    "model": get_state_dict_clean(model),
                    "classes": classes_full,
                    "healthy_idx": healthy_idx,
                    "img_size": IMG_SIZE,
                },
                os.path.join(out_dir, "best_bin.pt"),
            )
        print(f"[BIN {ep+1:03d}/{epochs}] loss={tl:.4f} val_acc={va:.4f}")

    # Faza 2: CLS (odmrażamy wszystko)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args, "lr", LR))

    for ep in range(epochs):
        tl, ta = train_epoch(model, dl_cls_tr, loss_cls, device, opt)
        print(f"[CLS {ep+1:03d}/{epochs}] loss={tl:.4f} acc={ta:.4f}")

    # Zapis końcowy
    ckpt = {
        "model": get_state_dict_clean(model),
        "classes": classes_full,
        "healthy_idx": healthy_idx,
        "img_size": IMG_SIZE,
        "train_time_s": round(time.time() - t0, 2),
        "cls_offset": cls_offset,
    }
    torch.save(ckpt, os.path.join(out_dir, "final.pt"))
    print(f"Saved to {os.path.join(out_dir, 'final.pt')}")
    return 0