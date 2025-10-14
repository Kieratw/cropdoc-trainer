
import argparse
import json
import os
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# =========================
#  Model: MobileNetV3-Dual
# =========================

def _get_mobilenet_v3_small(pretrained: bool = False):
    """
    Pobiera backbone MobileNetV3-Small, kompatybilnie ze starszym/nowym API.
    """
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        if pretrained:
            m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            m = mobilenet_v3_small(weights=None)
    except Exception:
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(pretrained=pretrained)
    features = m.features
    last_ch = 576  # kanały po features dla v3-small
    return features, last_ch


class MobileNetV3Dual(nn.Module):
    """
    Dwa heady:
      - head_bin: healthy vs diseased (2)
      - head_cls: wieloklasowy (healthy + choroby)
    """
    def __init__(self, num_classes: int, pretrained: bool = False, healthy_idx: int = 0):
        super().__init__()
        self.features, last_ch = _get_mobilenet_v3_small(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(last_ch)
        self.relu = nn.ReLU(inplace=True)
        self.head_bin = nn.Linear(last_ch, 2)
        self.head_cls = nn.Linear(last_ch, num_classes)
        self.healthy_idx = healthy_idx

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        return self.head_bin(x), self.head_cls(x)


# =========================
#  Checkpoint helpers
# =========================

def _normalize_state_dict(sd: dict) -> dict:
    """
    Usuwa prefiksy po DataParallel ('module.') i torch.compile ('_orig_mod.').
    """
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_ckpt(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ck or "classes" not in ck:
        raise RuntimeError("Nieprawidłowy checkpoint: brakuje kluczy 'model' lub 'classes'.")
    return ck


def build_model_from_ckpt(ck):
    m = MobileNetV3Dual(
        num_classes=len(ck["classes"]),
        pretrained=False,
        healthy_idx=ck.get("healthy_idx", 0),
    )
    sd = _normalize_state_dict(ck["model"])
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


# =========================
#  Preprocess / Inference
# =========================

def preprocess_pil(img: Image.Image, size: int = 256) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),  # 0..1
    ])
    return tf(img).unsqueeze(0)  # [1,3,H,W]


def infer_image(model, x: torch.Tensor):
    with torch.no_grad():
        lg_bin, lg_cls = model(x)
        pb = torch.softmax(lg_bin, dim=1)[0]  # [2]
        pc = torch.softmax(lg_cls, dim=1)[0]  # [C]
    return pb, pc


def decide(pb: torch.Tensor, pc: torch.Tensor, classes, healthy_idx: int,
           bin_threshold: float, cls_threshold: float):
    """
    Prosta logika decyzji:
      - jeśli p(healthy) z head_bin >= bin_threshold => healthy
      - w przeciwnym razie bierzemy argmax head_cls, jeśli != healthy i p >= cls_threshold => choroba
      - inaczej uncertain
    """
    if pb[0].item() >= bin_threshold:
        return "healthy", "healthy"
    pred_cls = int(torch.argmax(pc).item())
    if pred_cls != healthy_idx and pc[pred_cls].item() >= cls_threshold:
        return "diseased", classes[pred_cls]
    return "uncertain", None


def show_image_with_title(img: Image.Image, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
#  CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Demo: pojedynczy obraz -> predykcja i podgląd")
    ap.add_argument("--ckpt", required=True, help="Ścieżka do final.pt")
    ap.add_argument("--image", help="Ścieżka do obrazu (RGB). Jeśli brak, otworzy okno wyboru pliku.")
    ap.add_argument("--bin_threshold", type=float, default=0.60, help="Próg binarny healthy (default 0.60)")
    ap.add_argument("--cls_threshold", type=float, default=0.55, help="Próg pewności klasy choroby (default 0.55)")
    args = ap.parse_args()

    bin_thr = args.bin_threshold
    cls_thr = args.cls_threshold

    img_path = args.image
    if not img_path:
        # opcjonalne okienko wyboru pliku
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            img_path = filedialog.askopenfilename(
                title="Wybierz obraz",
                filetypes=[("Obrazy", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("Wszystkie pliki", "*.*")]
            )
        except Exception:
            raise SystemExit("Podaj --image albo zainstaluj tkinter, żeby użyć okienka wyboru pliku.")

    if not img_path or not os.path.isfile(img_path):
        raise SystemExit(f"Nie znaleziono pliku obrazu: {img_path}")

    # 1) wczytaj ckpt i model
    ck = load_ckpt(args.ckpt)
    classes = ck["classes"]
    healthy_idx = ck.get("healthy_idx", 0)
    img_size = int(ck.get("img_size", 256))
    model = build_model_from_ckpt(ck)

    # 2) wczytaj obraz i zrób preprocess
    img = Image.open(img_path).convert("RGB")
    x = preprocess_pil(img, img_size)

    # 3) inferencja i decyzja
    pb, pc = infer_image(model, x)
    decision, label = decide(pb, pc, classes, healthy_idx, bin_thr, cls_thr)

    # 4) top-3 klasy
    pc_list = pc.tolist()
    topk = min(3, len(classes))
    idxs = sorted(range(len(classes)), key=lambda i: pc_list[i], reverse=True)[:topk]
    top3 = [{"label": classes[i], "p": float(pc_list[i])} for i in idxs]

    result = {
        "decision": decision,
        "label": label,
        "p_bin": {"healthy": float(pb[0].item()), "diseased": float(pb[1].item())},
        "p_cls_top3": top3,
        "all_labels": classes
    }

    # 5) wypisz JSON na konsolę
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 6) pokaż obraz z tytułem
    if decision == "healthy":
        title = f"HEALTHY  (p={result['p_bin']['healthy']:.2f})"
    elif decision == "diseased" and label:
        title = f"{label}  (p={top3[0]['p']:.2f})"
    else:
        title = f"UNCERTAIN  (healthy={result['p_bin']['healthy']:.2f}, best={top3[0]['label']}:{top3[0]['p']:.2f})"

    show_image_with_title(img, title)


if __name__ == "__main__":
    main()