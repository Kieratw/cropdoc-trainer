# app.py — Streamlit tester: BIN→CLS + Grad-CAM (bez tłumaczeń)
import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import streamlit as st

# kolormap do heatmapy (bez cv2)
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


# ===== Model =====
def _get_mobilenet_v3_small(pretrained: bool = False):
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    except Exception:
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(pretrained=pretrained)
    return m.features, 576  # last_ch

class MobileNetV3Dual(nn.Module):
    def __init__(self, num_classes: int, healthy_idx: int = 0):
        super().__init__()
        self.features, last_ch = _get_mobilenet_v3_small(False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(last_ch)
        self.relu = nn.ReLU(inplace=True)
        # identycznie jak w treningu
        self.head_bin = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_ch, 2))
        self.head_cls = nn.Linear(last_ch, num_classes)
        self.healthy_idx = healthy_idx

    def forward(self, x):
        feats = self.features(x)          # [B,576,h,w]
        x = self.pool(feats).flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        return self.head_bin(x), self.head_cls(x), feats


# ===== Grad-CAM =====
class GradCAM:
    """Grad-CAM na ostatniej warstwie cech."""
    def __init__(self, model: MobileNetV3Dual):
        self.model = model
        self.target_layer = self.model.features[-1]
        self.acts = None
        self.grads = None
        self._reg_hooks()

    def _reg_hooks(self):
        def fwd(_, __, out):  self.acts = out.detach()
        def bwd(_, grad_in, grad_out): self.grads = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, x: torch.Tensor, target_logit: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=True)
        w = self.grads.mean(dim=(2, 3))[0]     # [C]
        A = self.acts[0]                        # [C,h,w]
        cam = torch.relu((w[:, None, None] * A).sum(0))
        cam = cam / (cam.max() + 1e-12)
        return cam.detach().cpu().numpy()

def overlay_cam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    W, H = pil_img.size
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray((cam * 255).astype(np.uint8)).resize((W, H), PILImage.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    heat = np.uint8(cm.get_cmap("jet")(cam_arr)[..., :3] * 255)
    return PILImage.blend(pil_img.convert("RGB"), PILImage.fromarray(heat), alpha=alpha)


# ===== I/O =====
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    classes_full: List[str] = ck["classes"]
    healthy_idx = int(ck.get("healthy_idx", 0))
    img_size = int(ck.get("img_size", 256))

    model = MobileNetV3Dual(num_classes=len(classes_full), healthy_idx=healthy_idx)
    sd = ck["model"]
    # zdejmij ewentualne prefiksy z DDP/compile
    fixed = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k.replace("module.", "", 1)
        if k.startswith("_orig_mod."): k = k.replace("_orig_mod.", "", 1)
        fixed[k] = v
    model.load_state_dict(fixed, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    tfm = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])
    return model, device, classes_full, healthy_idx, tfm, img_size

def pil_to_tensor(img: Image.Image, tfm: T.Compose, device: torch.device):
    return tfm(img.convert("RGB")).unsqueeze(0).to(device).to(memory_format=torch.channels_last)


# ===== UI =====
st.set_page_config(page_title="CropDoc – tester modeli", layout="wide")
st.title("🌿 CropDoc – tester modeli (BIN → CLS)")

with st.sidebar:
    st.header("⚙️ Ustawienia")
    ckpt_path = st.text_input("Ścieżka do checkpointu (.pt)", value="")
    bin_threshold = st.slider("Próg BIN – p(healthy) ≥", 0.0, 1.0, 0.50, 0.01)
    topk = st.number_input("Top-K chorób (CLS)", 1, 10, 3)
    show_cam = st.checkbox("Pokaż Grad-CAM", value=True)
    cam_head = st.selectbox("Wyjaśniaj logit", ["Automatycznie (final)", "BIN (healthy/diseased)", "CLS (top-1)"])
    btn_load = st.button("Załaduj model")

state = st.session_state
if (("model_loaded" not in state) and btn_load) or (btn_load and ckpt_path):
    try:
        model, device, classes_full, healthy_idx, tfm, img_size = load_model(ckpt_path)
        state.model, state.device = model, device
        state.classes_full, state.healthy_idx = classes_full, healthy_idx
        state.tfm, state.img_size = tfm, img_size
        state.model_loaded = True
        st.success(f"Załadowano ✅ • klasy: {classes_full} • healthy_idx={healthy_idx} • img_size={img_size}")
    except Exception as e:
        st.error(f"Nie udało się załadować modelu: {e}")
        state.model_loaded = False

if not state.get("model_loaded", False):
    st.info("Podaj ścieżkę do **final.pt** i kliknij **Załaduj model**.")
    st.stop()

st.markdown("---")
st.subheader("🔎 Przetestuj obrazy")
files = st.file_uploader("Przeciągnij JPG/PNG/BMP/TIFF/WEBP (wiele plików).",
                         type=["jpg","jpeg","png","bmp","tif","tiff","webp"],
                         accept_multiple_files=True)

def color_badge(p: float) -> str:
    if p >= 0.8: return "✅"
    if p >= 0.6: return "🟢"
    if p >= 0.4: return "🟡"
    return "🟠"

def infer_one(img_pil: Image.Image):
    x = pil_to_tensor(img_pil, state.tfm, state.device).requires_grad_(True)
    gradcam = GradCAM(state.model)

    lg_bin, lg_cls, feats = state.model(x)
    prob_bin = torch.softmax(lg_bin, dim=1)[0]  # [2]
    prob_cls = torch.softmax(lg_cls, dim=1)[0]  # [C]

    p_healthy = float(prob_bin[0].item())
    is_healthy = p_healthy >= bin_threshold

    # top-K po chorobach (bez healthy w CLS)
    if 0 <= state.healthy_idx < len(state.classes_full):
        mask = [j for j in range(len(state.classes_full)) if j != state.healthy_idx]
    else:
        mask = list(range(len(state.classes_full)))

    sub = prob_cls[mask]
    k = min(int(topk), len(mask))
    vals, idxs = torch.topk(sub, k=k)
    topk_names = [state.classes_full[mask[int(i.item())]] for i in idxs]
    topk_vals  = [float(v.item()) for v in vals]

    top1_global = mask[int(idxs[0].item())] if k > 0 else 0
    final_label = "healthy" if is_healthy else state.classes_full[top1_global]

    # wybór logitu do Grad-CAM
    if cam_head.startswith("BIN"):
        target = lg_bin[0, 0 if is_healthy else 1]
    elif cam_head.startswith("CLS"):
        target = lg_cls[0, top1_global]
    else:
        target = lg_bin[0, 0] if is_healthy else lg_cls[0, top1_global]

    cam = None
    try:
        cam = gradcam.generate(x, target)
    except Exception:
        cam = None

    return {
        "final_label": final_label,
        "p_healthy": p_healthy,
        "topk": list(zip(topk_names, topk_vals)),
        "cam": cam,
        "cam_head": ("BIN" if cam_head.startswith("BIN") else ("CLS" if cam_head.startswith("CLS") else ("BIN" if is_healthy else "CLS")))
    }

if files:
    cols = st.columns(2)
    with cols[0]:
        for up in files[:16]:
            st.image(up, caption=up.name, use_container_width=True)
    with cols[1]:
        for up in files:
            img = Image.open(io.BytesIO(up.read()))
            out = infer_one(img)
            st.markdown(f"**{up.name}** → **{out['final_label']}** {color_badge(out['p_healthy'])} "
                        f"*(p(healthy)={out['p_healthy']:.2f}, CAM: {out['cam_head']})*")
            if out["topk"]:
                st.caption("Top-{} (CLS): {}".format(
                    len(out["topk"]),
                    ", ".join([f"{n}: {p:.2f}" for n, p in out["topk"]])
                ))
            if show_cam and out["cam"] is not None:
                st.image(overlay_cam(img, out["cam"], alpha=0.35),
                         caption="Grad-CAM", use_container_width=True)

st.markdown("---")
st.caption("Pipeline: jeśli p(healthy) ≥ próg → wynik końcowy = 'healthy'; w p.p. choroba = argmax z CLS.")
