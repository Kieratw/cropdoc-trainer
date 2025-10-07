import argparse, json, time, inspect
from pathlib import Path
from contextlib import nullcontext

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# --- AMP compatibility shim (działa z różnymi wersjami PyTorch) ---
try:
    # nowsze API (torch>=2.4)
    from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
    _AMP_NEW = True
except Exception:
    # starsze API
    from torch.cuda.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
    _AMP_NEW = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def loaders(data_dir: Path, img=256, bs=64, workers=4):
    tr_tf = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    te_tf = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tr = datasets.ImageFolder(data_dir / "train", tr_tf)
    va = datasets.ImageFolder(data_dir / "val",   te_tf)
    te = datasets.ImageFolder(data_dir / "test",  te_tf)
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    dl_te = DataLoader(te, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return dl_tr, dl_va, dl_te, tr.classes, tr


def class_weights_from_dataset(ds, nclasses):
    counts = np.zeros(nclasses, dtype=np.int64)
    for _, y in ds.samples:
        counts[y] += 1
    weights = counts.max() / np.clip(counts, 1, None)
    return torch.tensor(weights, dtype=torch.float32)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Odwraca normalizację do [0,1] dla wizualizacji."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _find_last_conv(m: nn.Module):
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    return last


@torch.no_grad()
def eval_epoch(model, dl, crit, device):
    model.eval()
    tot, corr, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        o = model(x)
        loss = crit(o, y)
        loss_sum += loss.item() * x.size(0)
        corr += (o.argmax(1) == y).sum().item()
        tot  += x.size(0)
    return loss_sum / tot, corr / tot


def _save_pred_grid(model, dl, classes, device, out_dir: Path, epoch: int, n=12):
    """Siatka obrazków z predykcjami i prawdą."""
    model.eval()
    x, y = next(iter(dl))
    x, y = x.to(device)[:n], y[:n]
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1)
        conf, pred = probs.max(1)

    imgs = _denorm(x).cpu()
    n = imgs.size(0)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        t = classes[y[i].item()]
        p = classes[pred[i].item()]
        c = conf[i].item()
        ax.set_title(f"T:{t}\nP:{p} ({c:.2f})", fontsize=9)
        ax.axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{epoch:02d}_preds.png", dpi=160)
    plt.close(fig)


def _save_gradcam_grid(model, dl, classes, device, out_dir: Path, epoch: int, n=8):
    """Heatmapy Grad-CAM – kompatybilne z różnymi wersjami biblioteki."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception:
        print("[WARN] Grad-CAM niedostępny (zainstaluj: pip install opencv-python && "
              "pip install 'git+https://github.com/jacobgil/pytorch-grad-cam.git'). Pomijam.")
        return

    target_layer = _find_last_conv(model)
    if target_layer is None:
        print("[WARN] Nie znaleziono warstwy Conv2d do Grad-CAM. Pomijam.")
        return

    # Wykryj dostępne parametry konstruktora (różne API między wersjami)
    kwargs = {}
    try:
        sig = inspect.signature(GradCAM.__init__)
        params = sig.parameters
        if "device" in params:
            kwargs["device"] = device.type  # "cuda" albo "cpu"
        elif "use_cuda" in params:
            kwargs["use_cuda"] = (device.type == "cuda")
    except Exception:
        pass

    try:
        cam = GradCAM(model=model, target_layers=[target_layer], **kwargs)
    except TypeError:
        cam = GradCAM(model=model, target_layers=[target_layer])

    model.eval()
    x, y = next(iter(dl))
    x, y = x.to(device)[:n], y[:n]
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1)
        conf, pred = probs.max(1)

    imgs = _denorm(x).permute(0, 2, 3, 1).cpu().numpy()
    n = imgs.shape[0]
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))

    # Nie wszystkie wersje mają context manager; fallback na nullcontext
    ctx = cam if hasattr(cam, "__enter__") else nullcontext()
    try:
        with ctx:
            for i in range(n):
                targets = [ClassifierOutputTarget(int(pred[i].item()))]
                grayscale_cam = cam(input_tensor=x[i:i + 1], targets=targets)[0]
                vis = show_cam_on_image(imgs[i], grayscale_cam, use_rgb=True, image_weight=0.5)

                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.imshow(vis)
                t = classes[y[i].item()]
                p = classes[pred[i].item()]
                c = conf[i].item()
                ax.set_title(f"T:{t}\nP:{p} ({c:.2f})", fontsize=9)
                ax.axis("off")
    finally:
        try:
            del cam
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{epoch:02d}_cam.png", dpi=160)
    plt.close(fig)


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    viz_dir = out / "viz"

    dl_tr, dl_va, dl_te, classes, tr_ds = loaders(Path(args.data), args.img, args.batch, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    weights = class_weights_from_dataset(tr_ds, len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # --- AMP scaler + autocast kompatybilne z różnymi wersjami ---
    use_cuda = torch.cuda.is_available() and not args.no_amp
    try:
        if _AMP_NEW:
            # nowe API: pierwszy parametr to "cuda" / "cpu" / None
            scaler = AmpGradScaler("cuda" if use_cuda else None, enabled=use_cuda)
            autocast_cm = lambda enabled: amp_autocast("cuda", enabled=enabled)
        else:
            # stare API
            scaler = AmpGradScaler(enabled=use_cuda)
            autocast_cm = lambda enabled: amp_autocast(enabled=enabled)
    except TypeError:
        scaler = AmpGradScaler(enabled=use_cuda)
        try:
            autocast_cm = lambda enabled: amp_autocast("cuda", enabled=enabled)
        except TypeError:
            autocast_cm = lambda enabled: amp_autocast(enabled=enabled)

    best_acc, patience = 0.0, args.patience
    best_path = out / "best.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = corr = tot = 0

        for x, y in tqdm(dl_tr, leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_cm(scaler.is_enabled()):
                o = model(x)
                loss = crit(o, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            corr += (o.argmax(1) == y).sum().item()
            tot  += x.size(0)

        tr_loss, tr_acc = loss_sum / tot, corr / tot
        va_loss, va_acc = eval_epoch(model, dl_va, crit, device)
        sch.step()
        print(f"[{ep:02d}/{args.epochs}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | {time.time()-t0:.1f}s")

        # Wizualizacje
        if args.viz and (ep % args.viz_every == 0):
            _save_pred_grid(model, dl_va, classes, device, viz_dir, ep, n=args.viz_n)
            if args.cam:
                _save_gradcam_grid(model, dl_va, classes, device, viz_dir, ep, n=min(args.viz_n, 8))

        if va_acc > best_acc:
            best_acc, patience = va_acc, args.patience
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.model}, best_path)
            print(f"  ↳ saved best.pt (val_acc={best_acc:.4f})")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    # Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_pred = [], []
    for x, y in dl_te:
        x = x.to(device)
        with torch.no_grad():
            o = model(x).argmax(1).cpu().tolist()
        y_pred += o
        y_true += y.tolist()

    rep = classification_report(y_true, y_pred, target_names=classes, digits=4)
    (out / "report.txt").write_text(rep, encoding="utf-8")
    print("\n[TEST] Classification report:\n", rep)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=90)
    plt.yticks(tick, classes)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    print(f"Saved: {out/'report.txt'}, {out/'confusion_matrix.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    type=str, default="data_merged")
    ap.add_argument("--out",     type=str, default="runs/poc_mnv3")
    ap.add_argument("--model",   type=str, default="mobilenetv3_small_100")
    ap.add_argument("--epochs",  type=int, default=8)
    ap.add_argument("--img",     type=int, default=256)
    ap.add_argument("--batch",   type=int, default=64)
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--wd",      type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience",type=int, default=5)
    ap.add_argument("--no-amp",  action="store_true")

    # Wizualizacje
    ap.add_argument("--viz",        action="store_true", help="zapisuj podgląd predykcji z walidacji")
    ap.add_argument("--viz-every",  type=int, default=1, help="co ile epok zapisywać wizualizacje", dest="viz_every")
    ap.add_argument("--viz-n",      type=int, default=12, help="ile obrazów w siatce", dest="viz_n")
    ap.add_argument("--cam",        action="store_true", help="zapisuj też Grad-CAM (wymaga pytorch-grad-cam)")

    args = ap.parse_args()
    train(args)
