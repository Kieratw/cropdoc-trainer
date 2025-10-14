import os
import json
import time
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import matplotlib.pyplot as plt

from .models import MobileNetV3Dual

IMG_SIZE = 256
LR = 1e-3
EPOCHS_BIN = 100
EPOCHS_CLS = 20
HEALTHY_TOKEN = "healthy"


def set_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ShardPack(Dataset):
    """Dataset do shardów z build_dataset.py"""
    def __init__(self, root: str, split: str):
        root = Path(root)
        part = root / split / "s000"
        if not part.exists(): raise FileNotFoundError(f"Brak shardu {part.as_posix()}")
        meta = json.loads((part / "meta.json").read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        C,H,W = [int(x) for x in meta["shape"]]
        data_file = part / meta.get("data_file","images.dat")
        labels_file = part / meta.get("labels_file","index.npy")
        self.mm = np.memmap(data_file, dtype=np.uint8, mode="r", shape=(os.path.getsize(data_file)//(C*H*W),C,H,W))
        self.y = np.load(labels_file, mmap_mode="r")

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor,int]:
        arr = np.array(self.mm[i], copy=True)
        x = torch.from_numpy(arr).float() / 255.0
        y = int(self.y[i])
        return x, y


class RemapTargets(Dataset):
    """Dataset wrapper do przesunięcia etykiet o add"""
    def __init__(self, base: Dataset, add:int=0):
        self.base = base
        self.add = int(add)

    def __len__(self):
        return len(self.base)

    def __getitem__(self,i):
        x,y = self.base[i]
        return x,int(y)+self.add


def try_compile(model: torch.nn.Module, force_disable: bool=False):
    if force_disable: return model
    try:
        import triton  # noqa
        return torch.compile(model, mode="max-autotune")
    except Exception:
        try: return torch.compile(model, backend="eager")
        except Exception: return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred==targets).float().mean().item()


def train_epoch(model, loader, loss_fn, device, opt=None):
    model.train() if opt is not None else model.eval()
    tot_loss, tot_acc, n = 0.0,0.0,0
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if opt is not None: opt.zero_grad(set_to_none=True)
        lg_bin, lg_cls = model(x)
        loss, logits = loss_fn((lg_bin, lg_cls), y)
        if opt is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        bs = y.shape[0]
        tot_loss += loss.item()*bs
        tot_acc += accuracy(logits, y)*bs
        n += bs
    return tot_loss/max(n,1), tot_acc/max(n,1)


def get_state_dict_clean(model: torch.nn.Module) -> dict:
    base = getattr(model,"_orig_mod",model)
    base = getattr(base,"module",base)
    return base.state_dict()


def gather_labels_from_dataset(dataset: Dataset) -> Tuple[List[int], List[int]]:
    if hasattr(dataset,"dataset") and hasattr(dataset,"indices"):
        base = dataset.dataset
        idxs = list(dataset.indices)
    else:
        base = dataset
        idxs = list(range(len(base)))
    labels=[]
    for i in idxs:
        try: _x,y = base[i]
        except Exception:
            if hasattr(base,"y"): y=base.y[i]
            else: raise
        labels.append(int(y))
    return idxs, labels


def plot_training_history(history: dict, out_dir: Path):
    fig, ax = plt.subplots(1,2, figsize=(14,5))

    # Loss
    for phase in ["bin","cls","ft"]:
        if f"{phase}_loss" in history:
            ax[0].plot(history[f"{phase}_loss"], label=f"{phase.upper()} loss")
    ax[0].set_title("Loss over epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Accuracy
    for phase in ["bin","cls","ft"]:
        if f"{phase}_acc" in history:
            ax[1].plot(history[f"{phase}_acc"], label=f"{phase.upper()} acc")
    ax[1].set_title("Accuracy over epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir/"training_summary.png", dpi=150)
    plt.close(fig)
    print(f"Training summary saved → {out_dir/'training_summary.png'}")
    with open(out_dir/"training_history.json","w") as f:
        json.dump(history,f,indent=2)


def run(args):
    set_seeds(getattr(args,"seed",1337))
    device = torch.device(getattr(args,"device","cuda") if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load shards ----
    ds_bin = ShardPack(args.bin,"train")
    val_len = max(1,int(len(ds_bin)*0.10))
    tr_len = len(ds_bin)-val_len
    ds_bin_tr, ds_bin_va = random_split(ds_bin,[tr_len,val_len])

    ds_cls = ShardPack(args.cls,"train")
    classes_cls = ds_cls.classes

    if HEALTHY_TOKEN in classes_cls:
        classes_full = classes_cls
        healthy_idx = classes_full.index(HEALTHY_TOKEN)
        ds_cls_wrapped = ds_cls
        cls_offset = 0
    else:
        classes_full = [HEALTHY_TOKEN]+classes_cls
        healthy_idx = 0
        cls_offset = 1
        ds_cls_wrapped = RemapTargets(ds_cls, add=cls_offset)

    batch = getattr(args,"batch",64)
    num_workers = getattr(args,"workers",0)
    pin = device.type=="cuda"

    # ---- BIN undersample ----
    undersample_ratio = float(getattr(args,"bin_undersample_ratio",0.7))  # zmniejszony ratio dla lepszej równowagi
    idxs_all, labels_all = gather_labels_from_dataset(ds_bin_tr)
    labels_arr = np.array(labels_all,dtype=int)
    healthy_mask = labels_arr==0
    diseased_mask = ~healthy_mask
    healthy_idxs = [idxs_all[i] for i in np.where(healthy_mask)[0].tolist()]
    diseased_idxs = [idxs_all[i] for i in np.where(diseased_mask)[0].tolist()]
    print(f"BIN dataset counts (before undersample): healthy={len(healthy_idxs)}, diseased={len(diseased_idxs)}")

    target_diseased = min(len(diseased_idxs),int(max(0,len(healthy_idxs)*undersample_ratio)))
    if target_diseased < len(diseased_idxs):
        rng = np.random.default_rng(getattr(args,"seed",1337))
        chosen_diseased = rng.choice(diseased_idxs,size=target_diseased,replace=False).tolist()
    else: chosen_diseased = diseased_idxs

    final_bin_indices = healthy_idxs+chosen_diseased
    rng = np.random.default_rng(getattr(args,"seed",1337))
    rng.shuffle(final_bin_indices)

    # ===== POPRAWKA indeksów =====
    if isinstance(ds_bin_tr, Subset):
        base_indices = ds_bin_tr.indices
        idx_map = {orig_idx:i for i,orig_idx in enumerate(base_indices)}
        final_bin_indices_in_tr = [idx_map[i] for i in final_bin_indices if i in idx_map]
    else:
        final_bin_indices_in_tr = final_bin_indices

    ds_bin_tr_subset = Subset(ds_bin_tr, final_bin_indices_in_tr)
    dl_bin_tr = DataLoader(ds_bin_tr_subset,batch_size=batch,shuffle=True,num_workers=num_workers,pin_memory=pin)
    dl_bin_va = DataLoader(ds_bin_va,batch_size=batch,shuffle=False,num_workers=num_workers,pin_memory=pin)
    dl_cls_tr = DataLoader(ds_cls_wrapped,batch_size=batch,shuffle=True,num_workers=num_workers,pin_memory=pin)

    # ---- Model ----
    model = MobileNetV3Dual(num_classes=len(classes_full), pretrained=True, healthy_idx=healthy_idx).to(device).to(memory_format=torch.channels_last)
    model.head_bin = nn.Sequential(nn.Dropout(0.2), model.head_bin)  # Dropout w head BIN
    model = try_compile(model, force_disable=getattr(args,"no_compile",False))
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args,"lr",LR))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_BIN)  # dynamiczny LR

    loss_ce = nn.CrossEntropyLoss()
    def loss_bin(out_pair,y): lg,_=out_pair; return loss_ce(lg,y), lg
    def loss_cls(out_pair,y): _,lg=out_pair; return loss_ce(lg,y), lg

    # ---- Freeze features for BIN ----
    for p in model.features.parameters(): p.requires_grad=False
    for m in model.modules():
        if isinstance(m,(nn.BatchNorm1d, nn.BatchNorm2d)): m.train()

    history = {"bin_loss":[], "bin_acc":[], "cls_loss":[], "cls_acc":[]}

    # ---- BIN phase z early stopping ----
    best_val = -1.0
    patience = 100
    no_improve = 0
    for ep in range(EPOCHS_BIN):
        tl,_ = train_epoch(model, dl_bin_tr, loss_bin, device,opt)
        vl, va = train_epoch(model, dl_bin_va, loss_bin, device,opt=None)
        scheduler.step()
        history["bin_loss"].append(tl)
        history["bin_acc"].append(va)
        if va>best_val:
            best_val = va
            no_improve = 0
            torch.save({"model":get_state_dict_clean(model),"classes":classes_full,"healthy_idx":healthy_idx,"img_size":IMG_SIZE},out_dir/"best_bin.pt")
        else:
            no_improve += 1
        print(f"[BIN {ep+1}/{EPOCHS_BIN}] loss={tl:.4f} val_acc={va:.4f} LR={scheduler.get_last_lr()[0]:.6f}")
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    # ---- CLS phase ----
    for p in model.parameters(): p.requires_grad=True
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args,"lr",LR))
    for ep in range(EPOCHS_CLS):
        tl,ta = train_epoch(model, dl_cls_tr, loss_cls, device,opt)
        history["cls_loss"].append(tl)
        history["cls_acc"].append(ta)
        print(f"[CLS {ep+1}/{EPOCHS_CLS}] loss={tl:.4f} acc={ta:.4f}")

    # ---- Final save ----
    ckpt = {"model":get_state_dict_clean(model),"classes":classes_full,"healthy_idx":healthy_idx,"img_size":IMG_SIZE,"train_time_s":round(time.time()-time.time(),2),"cls_offset":cls_offset}
    torch.save(ckpt,out_dir/"final.pt")
    print(f"Saved to {out_dir/'final.pt'}")

    # ---- Plot history ----
    plot_training_history(history, out_dir)

    return 0
