# train_kd.py
# Knowledge Distillation na memmap packach (cls/train/s000 i cls/val/s000).
# Funkcje:
# - pretrained backbone, freeze→unfreeze
# - param groups: mniejszy LR dla backbone
# - KD z ramp-upem alpha, temperatura T
# - WeightedRandomSampler do balansu klas
# - prosty zestaw augmentacji (opcjonalny)
# - AMP, EMA, warmup+cosine LR
# - zapis best.pt wg F1(macro) na walidacji

import argparse
import json
import math
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _tv_weights(arch: str, pretrained: bool):
    if not pretrained:
        return None
    arch = arch.lower()
    try:
        if arch in {"mobilenetv3", "mobilenetv3_large", "mobilenetv3-large"}:
            return tvm.MobileNet_V3_Large_Weights.DEFAULT
        if arch in {"convnext_tiny", "convnext-tiny", "convnext"}:
            return tvm.ConvNeXt_Tiny_Weights.DEFAULT
        if arch in {"efficientnet_b0", "efficientnet"}:
            return tvm.EfficientNet_B0_Weights.DEFAULT
    except Exception:
        pass
    return None


def create_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch in {"mobilenetv3", "mobilenetv3_large", "mobilenetv3-large"}:
        m = tvm.mobilenet_v3_large(weights=_tv_weights(arch, pretrained))
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch in {"convnext_tiny", "convnext-tiny", "convnext"}:
        m = tvm.convnext_tiny(weights=_tv_weights(arch, pretrained))
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch in {"efficientnet_b0", "efficientnet"}:
        m = tvm.efficientnet_b0(weights=_tv_weights(arch, pretrained))
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise SystemExit(f"Nieznana architektura: {arch}")


# -------------------- dataset --------------------
class MemmapPack(Dataset):
    def __init__(self, pack_dir: Path, normalize: bool = True):
        self.pack_dir = Path(pack_dir)
        meta_path = self.pack_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"Brak meta.json w {self.pack_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.meta = meta
        self.labels = np.load(self.pack_dir / "index.npy")
        C, H, W = meta["shape"]
        self.shape = (len(self.labels), C, H, W)
        self.mm = np.memmap(self.pack_dir / "images.dat", dtype=np.uint8, mode="r", shape=self.shape)
        self.normalize = normalize
        self.mean = np.array(meta["mean"], dtype=np.float32)
        self.std = np.array(meta["std"], dtype=np.float32)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.mm[idx].astype(np.float32) / 255.0
        if self.normalize:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        y = int(self.labels[idx])
        return torch.from_numpy(x), y


def _weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    uniq, cnt = np.unique(labels, return_counts=True)
    inv = {int(u): 1.0 / float(c) for u, c in zip(uniq, cnt)}
    weights = np.array([inv[int(y)] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(labels), replacement=True)


def load_dataloaders(dataset_root: Path, batch_size: int, workers: int, balance: bool):
    train_pack = Path(dataset_root) / "cls" / "train" / "s000"
    val_pack = Path(dataset_root) / "cls" / "val" / "s000"

    if not (train_pack / "meta.json").exists():
        raise SystemExit(f"Brak packa train: {train_pack}")
    if not (val_pack / "meta.json").exists():
        raise SystemExit(f"Brak packa val:   {val_pack}")

    meta = json.loads((train_pack / "meta.json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    img_size = int(meta["shape"][1])

    ds_train = MemmapPack(train_pack, normalize=True)
    ds_val = MemmapPack(val_pack, normalize=True)

    if balance:
        sampler = _weighted_sampler(ds_train.labels)
        tr = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        tr = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
    va = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return tr, va, classes, img_size


# -------------------- KD / EMA / Eval --------------------
def kd_loss(student_logits, teacher_logits, T: float):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_tensor(v) and v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and torch.is_tensor(v) and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v.clone()
        model.load_state_dict(sd, strict=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)
            correct += int((pred == yb).sum().item())
            total += yb.size(0)
            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    acc = correct / max(1, total)
    return f1, acc


def one_cycle_cosine(it, total, base_lr, min_lr=1e-6):
    cos = (1 + math.cos(math.pi * it / max(1, total - 1))) / 2
    return min_lr + (base_lr - min_lr) * cos


def split_head_backbone(model):
    head, back = [], []
    for name, p in model.named_parameters():
        if name.startswith("classifier"):
            head.append(p)
        else:
            back.append(p)
    return head, back


# -------------------- augmentacje --------------------
def cheap_augs(xb: torch.Tensor) -> torch.Tensor:
    # xb: [B,C,H,W] float po normalizacji
    b, c, h, w = xb.shape
    out = []
    for i in range(b):
        x = xb[i]
        if torch.rand(1).item() < 0.5:
            x = TF.hflip(x)
        angle = float(torch.empty(1).uniform_(-8, 8))
        x = TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR)
        if torch.rand(1).item() < 0.8:
            bright = float(torch.empty(1).uniform_(0.9, 1.1))
            x = x * bright
            x = x + float(torch.empty(1).uniform_(-0.03, 0.03))
        out.append(x.clamp_(-3.0, 3.0))
    return torch.stack(out, dim=0)


# -------------------- train loop --------------------
def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tr_dl, va_dl, classes, img_size = load_dataloaders(Path(args.data), args.batch_size, args.workers, args.balance)
    num_classes = len(classes)

    arch_now = args.arch_student if args.mode == "student" else args.arch_teacher
    model = create_model(arch_now, num_classes, pretrained=args.pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)

    print(f"Model: {args.mode} | arch={arch_now} | params={count_params(model)/1e6:.2f}M | classes={num_classes} | img={img_size}")

    # Teacher dla trybu student
    teacher = None
    if args.mode == "student":
        if not args.teacher_ckpt or not Path(args.teacher_ckpt).exists():
            raise SystemExit("Podaj --teacher_ckpt do trenowania ucznia (ścieżka do best.pt nauczyciela).")
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
        t_arch = ckpt.get("arch", args.arch_teacher)
        if t_arch.lower() != args.arch_teacher.lower():
            print(f"[info] Ignoruję --arch_teacher={args.arch_teacher} i biorę arch z ckpt: {t_arch}")
        teacher = create_model(t_arch, num_classes, pretrained=False).to(device)
        teacher.load_state_dict(ckpt["model"], strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Załadowano nauczyciela z: {args.teacher_ckpt}")

    # loss dynamiczny będzie przez F.cross_entropy z label_smoothing
    params_head, params_back = split_head_backbone(model)
    for p in params_back:
        p.requires_grad = False  # freeze backbone na start

    def make_opt(backbone_lr_scale: float):
        back_trainable = [p for p in params_back if p.requires_grad]
        head_trainable = [p for p in params_head if p.requires_grad]
        groups = []
        if back_trainable:
            groups.append({"params": back_trainable, "lr": args.lr * backbone_lr_scale, "base_lr": args.lr * backbone_lr_scale})
        if head_trainable:
            groups.append({"params": head_trainable, "lr": args.lr, "base_lr": args.lr})
        if not groups:
            raise SystemExit("Brak trenowalnych parametrów. Sprawdź freeze/backbone.")
        return AdamW(groups, weight_decay=args.wd)

    opt = make_opt(backbone_lr_scale=args.backbone_lr_scale)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model, decay=0.999) if args.ema else None

    total_steps = args.epochs * len(tr_dl)
    warmup_steps = min(500, total_steps // 10)
    step = 0
    best_f1 = -1.0
    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone po kilku epokach
        if epoch == (args.freeze_epochs + 1):
            for p in params_back:
                p.requires_grad = True
            opt = make_opt(backbone_lr_scale=args.backbone_lr_scale)

        model.train()
        pbar = tqdm(tr_dl, desc=f"epoch {epoch}/{args.epochs}", ncols=120)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            # LR schedule: warmup -> cosine
            if step < warmup_steps:
                lr_now = args.lr * (step + 1) / max(1, warmup_steps)
            else:
                lr_now = one_cycle_cosine(step - warmup_steps, total_steps - warmup_steps, args.lr, args.min_lr)
            scale = lr_now / args.lr
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

            # proste augmenty na train
            if args.aug:
                xb = cheap_augs(xb)

            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(xb)

                # KD ramp-up: alpha rośnie liniowo przez kd_warmup_epochs
                alpha_now = 0.0
                if teacher is not None:
                    alpha_now = min(args.kd_alpha, args.kd_alpha * max(0, epoch - 1) / max(1, args.kd_warmup_epochs))

                # zmniejsz smoothing, gdy KD rośnie (żeby nie podwójnie "zmiękczać")
                ls_now = args.label_smoothing * (1.0 - alpha_now)
                loss_ce = F.cross_entropy(logits, yb, label_smoothing=ls_now)

                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(xb)
                    loss_kd = kd_loss(logits, t_logits, args.kd_temp)
                    loss = (1 - alpha_now) * loss_ce + alpha_now * loss_kd
                else:
                    loss = loss_ce

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            step += 1

            pbar.set_postfix(lr=f"{lr_now:.1e}", loss=f"{loss.item():.4f}", kd_a=f"{alpha_now:.2f}", ls=f"{ls_now:.3f}")

        # Walidacja
        eval_model = model
        if ema:
            ema_model = create_model(arch_now, num_classes, pretrained=False).to(device)
            ema.apply_to(ema_model)
            eval_model = ema_model

        f1, acc = evaluate(eval_model, va_dl, device)
        print(f"[val] F1(macro)={f1:.4f}  acc={acc:.4f}")

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model": eval_model.state_dict(),
                    "classes": classes,
                    "img_size": img_size,
                    "arch": arch_now,
                    "mode": args.mode,
                },
                save_dir / "best.pt",
            )
            print(f"✓ Zapisano best.pt (F1={best_f1:.4f})")

    print(f"Koniec. Best F1={best_f1:.4f}. Pliki w: {save_dir}")


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser("KD training na memmap packach (pretrained + freeze/unfreeze + class-balance)")
    ap.add_argument("--data", required=True, help="Ścieżka do datasetu, np. D:/.../wheat")
    ap.add_argument("--mode", choices=["teacher", "student"], required=True)
    ap.add_argument("--arch_teacher", default="convnext_tiny")
    ap.add_argument("--arch_student", default="mobilenetv3_large")
    ap.add_argument("--teacher_ckpt", default=None, help="best.pt nauczyciela (dla mode=student)")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=5)

    ap.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.05, help="Skalowanie LR dla backbone vs head")

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_warmup_epochs", type=int, default=5)
    ap.add_argument("--kd_temp", type=float, default=3.0)

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--balance", action="store_true", default=True, help="WeightedRandomSampler na train")
    ap.add_argument("--aug", action="store_true", help="Włącz proste augmentacje na train")
    ap.add_argument("--out", required=True, help="Folder na wyniki, np. runs/wheat_kd")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
