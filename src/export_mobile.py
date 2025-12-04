import argparse, json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.mobile_optimizer import optimize_for_mobile

# ---- (Skopiuj te funkcje ze starego skryptu BEZ ZMIAN) ----
# 1. create_mobilenetv3_large
# 2. try_autodetect_meta
# 3. load_norm_from_meta
# 4. WrappedClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_mobilenetv3_large(num_classes: int) -> nn.Module:
    m = tvm.mobilenet_v3_large(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def try_autodetect_meta(images_dir: Optional[Path]) -> Optional[Path]:
    try:
        if images_dir and images_dir.exists() and images_dir.name == "cls" and images_dir.parent.name == "test_images":
            root = images_dir.parent.parent
            cand = root / "cls" / "train" / "s000" / "meta.json"
            if cand.exists():
                return cand
    except Exception:
        pass
    return None


def load_norm_from_meta(meta_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray, str]:
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mean = np.array(meta.get("mean", IMAGENET_MEAN), dtype=np.float32)
        std = np.array(meta.get("std", IMAGENET_STD), dtype=np.float32)
        return mean, std, "meta.json"
    return np.array(IMAGENET_MEAN, dtype=np.float32), np.array(IMAGENET_STD, dtype=np.float32), "imagenet"


class WrappedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, mean: np.ndarray, std: np.ndarray, add_softmax: bool):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean.reshape(1, 3, 1, 1)))
        self.register_buffer("std", torch.tensor(std.reshape(1, 3, 1, 1)))
        self.add_softmax = add_softmax

    def forward(self, x_nchw: torch.Tensor):
        x = (x_nchw - self.mean) / self.std
        logits = self.backbone(x)
        if self.add_softmax:
            return F.softmax(logits, dim=1)
        return logits


# ---- (Główny skrypt) ----

def export_pytorch_mobile(ckpt_path: Path, images_dir: Optional[Path], meta_path: Optional[Path],
                          out_dir: Path, add_softmax: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Wczytanie ckpt (bez zmian)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", None)
    arch = ckpt.get("arch", "")
    img_size = int(ckpt.get("img_size", 380))
    if classes is None: raise SystemExit("Brak 'classes' w ckpt.")
    if "mobilenetv3" not in arch.lower(): raise SystemExit(f"Wspierany tylko MobileNetV3, znaleziono: {arch}.")

    # 3) Normy mean/std (bez zmian)
    meta_auto = try_autodetect_meta(images_dir) if not meta_path else None
    meta_to_use = meta_path or meta_auto
    mean, std, norm_src = load_norm_from_meta(meta_to_use)
    print(f"[norm] źródło={norm_src}  mean={mean.tolist()}  std={std.tolist()}")

    # 4) Złożenie modelu (bez zmian)
    backbone = create_mobilenetv3_large(len(classes))
    backbone.load_state_dict(ckpt["model"], strict=True)
    wrapped = WrappedClassifier(backbone, mean, std, add_softmax=add_softmax).eval()

    # 5) === NOWA LOGIKA EKSPORTU (PyTorch Script) ===
    ptl_path = out_dir / "oilseed_rape_v2.ptl"
    dummy = torch.rand(1, 3, img_size, img_size, dtype=torch.float32)

    try:
        print("[INFO] Uruchamiam torch.jit.script...")
        scripted_model = torch.jit.script(wrapped)
        print("[INFO] Uruchamiam optimize_for_mobile...")
        optimized_model = optimize_for_mobile(scripted_model)

        # Zapisz jako delegowany plik .ptl
        optimized_model._save_for_lite_interpreter(str(ptl_path))

        print(f"[OK] Zapisano model PyTorch Mobile -> {ptl_path}")

    except Exception as e:
        raise SystemExit(f"Eksport do PyTorch Mobile (.ptl) nie wyszedł. Błąd: {e}")

    # 6) Metadane (bez zmian)
    (out_dir / "export_info.json").write_text(json.dumps({
        "classes": classes, "arch": arch, "img_size": img_size,
        "normalization_source": norm_src, "mean": mean.tolist(), "std": std.tolist(),
        "outputs": "probs" if add_softmax else "logits",
        "mobile_model": "oilseed_rape_v2.ptl"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] export_info.json zapisany.")


# ---- (Parser CLI bez zmian) ----
def parse_args():
    ap = argparse.ArgumentParser("Export MobileNetV3 student -> PyTorch Mobile (.ptl)")
    ap.add_argument("--ckpt", required=True, help="Ścieżka do best.pt (uczeń MobileNetV3)")
    ap.add_argument("--images", default=None, help="Opcjonalnie test_images/cls do autodetekcji meta.json")
    ap.add_argument("--meta", default=None, help="meta.json z mean/std; jeśli podasz, nadpisze autodetekcję")
    ap.add_argument("--out", required=True, help="Folder wyjściowy")
    ap.add_argument("--softmax", action="store_true", help="Dołącz softmax do wyjścia (łatwiejsza apka)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = Path(args.ckpt)
    out = Path(args.out)
    images = Path(args.images) if args.images else None
    meta = Path(args.meta) if args.meta else None
    export_pytorch_mobile(ckpt, images, meta, out, add_softmax=args.softmax)