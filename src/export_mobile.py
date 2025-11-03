import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


# ===== Architektura zgodna z treningiem: MobileNetV3Dual (BIN + CLS) =====
def _get_mobilenet_v3_small(pretrained: bool = False):
    """
    Zwraca features i liczbę kanałów wyjściowych ostatniego bloku MobileNetV3 Small.
    """
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
    """
    Dwie głowice:
      - head_bin: Dropout(0.2) + Linear(last_ch, 2)  -> [healthy, diseased]
      - head_cls: Linear(last_ch, num_classes)       -> wieloklasowa (z healthy w classes)
    """
    def __init__(self, num_classes: int, pretrained: bool = False, healthy_idx: int = 0):
        super().__init__()
        self.features, last_ch = _get_mobilenet_v3_small(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(last_ch)
        self.relu = nn.ReLU(inplace=True)
        self.head_bin = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_ch, 2))
        self.head_cls = nn.Linear(last_ch, num_classes)
        self.healthy_idx = int(healthy_idx)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        return self.head_bin(x), self.head_cls(x)


# ===== Utils =====
def _normalize_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Usuwa prefiksy 'module.' i '_orig_mod.' z kluczy state_dict, jeśli występują.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[10:]
        out[k] = v
    return out


def export_one(plant: str, ckpt_path: Path, out_assets_root: Path, img_size: int, make_dict: bool):
    """
    Eksport jednego checkpointu (final.pt) do assets:
      android/app/src/main/assets/models/<plant>/model_mobile.pt
      android/app/src/main/assets/models/<plant>/model_mobile.pt.json
    Opcjonalnie tworzy assets/i18n/pl_<plant>.json (szablon).
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    classes = ck["classes"]
    healthy_idx = int(ck.get("healthy_idx", 0))
    ck_img_size = int(ck.get("img_size", img_size))

    model = MobileNetV3Dual(num_classes=len(classes), pretrained=False, healthy_idx=healthy_idx).eval()
    model.load_state_dict(_normalize_sd(ck["model"]), strict=True)

    # Torch 2.x: zapis standardowym TorchScript .pt
    scripted = torch.jit.script(model)

    model_dir = out_assets_root / "models" / plant
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = model_dir / "model_mobile.pt"
    torch.jit.save(scripted, str(out_model_path))

    meta = {"classes": classes, "healthy_idx": healthy_idx, "img_size": ck_img_size}
    meta_path = model_dir / "model_mobile.pt.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    dict_path = None
    if make_dict:
        i18n_dir = out_assets_root / "i18n"
        i18n_dir.mkdir(parents=True, exist_ok=True)
        dict_path = i18n_dir / f"pl_{plant}.json"
        if not dict_path.exists():
            base = {"healthy": "zdrowa"}
            for c in classes:
                if c not in base:
                    base[c] = c  # start: etykieta EN skopiowana, dopisz PL ręcznie
            dict_path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")

    return out_model_path, meta_path, dict_path


def parse_models_arg(items):
    """
    Parsuje wiele --model plant=PATH do dict {plant: Path}.
    """
    pairs = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Zły format --model: {it}. Użyj: plant=ścieżka_do_final.pt")
        plant, path = it.split("=", 1)
        plant = plant.strip()
        if not plant:
            raise SystemExit(f"Pusta nazwa rośliny w: {it}")
        pairs[plant] = Path(path).expanduser().resolve()
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Export do TorchScript .pt (Torch 2.x) dla wielu roślin")
    ap.add_argument("--model", action="append", required=True,
                    help="Para plant=ścieżka_do_final.pt (możesz podać wiele razy)")
    ap.add_argument("--out-assets", default="android/app/src/main/assets",
                    help="Katalog assets w projekcie (domyślnie android/app/src/main/assets)")
    ap.add_argument("--img-size", type=int, default=256, help="Rozmiar wejścia SxS (fallback, gdy ckpt nie ma img_size)")
    ap.add_argument("--make-dict", action="store_true", help="Utwórz szablon słownika PL dla każdej rośliny")
    args = ap.parse_args()

    out_root = Path(args.out_assets).resolve()
    pairs = parse_models_arg(args.model)

    print(f"Wyjście assets: {out_root}")
    for plant, ckpt in pairs.items():
        if not ckpt.exists():
            raise SystemExit(f"Nie ma pliku checkpointu: {ckpt}")
        print(f"[{plant}] eksport z {ckpt} ...")
        out_model, meta_json, dict_path = export_one(plant, ckpt, out_root, args.img_size, args.make_dict)
        print(f"  -> model: {out_model}")
        print(f"  -> meta : {meta_json}")
        if args.make_dict:
            print(f"  -> dict : {dict_path}")

    print("Gotowe.")


if __name__ == "__main__":
    main()