from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

PIXRESTORE_ROOT = Path(os.getenv("PIXRESTORE_ROOT", "/opt/PixRestore"))
if str(PIXRESTORE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIXRESTORE_ROOT))

from inference import apply_checkpoint_architecture_flags, build_flow, build_model, load_weights, read_config, read_state_dict
from pixrestore.vision import extract_layers, load_dinov2

MODEL_REPO = os.getenv("PIXRESTORE_MODEL_REPO", "VCLab-PolyU/PixRestore")
MODEL_REVISION = os.getenv("PIXRESTORE_MODEL_REVISION", "main")
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", "/models/pixrestore"))
DINO_REPO = os.getenv("DINOV2_REPOSITORY", "") or None
DINO_CHECKPOINT = os.getenv("DINOV2_CHECKPOINT", "") or None
NUDENET_PATH = Path(os.getenv("NUDENET_PATH", "/models/nudenet.onnx"))
NUDENET_URL = os.getenv("NUDENET_URL", "https://raw.githubusercontent.com/vladmandic/sd-extension-nudenet/main/nudenet.onnx")
MAX_OUTPUT_LONG = int(os.getenv("MAX_OUTPUT_LONG", "3072"))

NUDE_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
SENSITIVE = {
    "FEMALE_GENITALIA_COVERED", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
}

class EnhanceRequest(BaseModel):
    image: str
    profile: str = "wow-v1"

@dataclass
class DegradationProfile:
    brightness: float
    contrast: float
    noise: float
    blur: float
    severity: float

@dataclass
class Region:
    label: str
    score: float
    box: tuple[float, float, float, float]

class SilentBodyContext:
    """Optional local detector used only as a structural quality gate."""
    def __init__(self) -> None:
        self.session = None
        self.input_name = None
        self.output_name = None
        try:
            import onnxruntime as ort
            NUDENET_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not NUDENET_PATH.exists():
                urllib.request.urlretrieve(NUDENET_URL, NUDENET_PATH)
            self.session = ort.InferenceSession(str(NUDENET_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as exc:
            print(f"[body-context] optional detector unavailable: {exc}", flush=True)
            self.session = None

    @staticmethod
    def _iou(a: Region, b: Region) -> float:
        ax, ay, aw, ah = a.box; bx, by, bw, bh = b.box
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / (aw * ah + bw * bh - inter + 1e-6)

    def detect(self, image: Image.Image) -> list[Region]:
        if self.session is None:
            return []
        src = image.convert("RGB")
        w, h = src.size
        size = 320
        scale = min(size / w, size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        px, py = (size - nw) // 2, (size - nh) // 2
        canvas = Image.new("RGB", (size, size), "black")
        canvas.paste(src.resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
        tensor = arr.transpose(2, 0, 1)[None]
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        data = raw[0]
        if data.shape[0] <= 64:
            data = data.T
        candidates: list[Region] = []
        for row in data:
            if len(row) < 5:
                continue
            scores = row[4:4 + len(NUDE_LABELS)]
            cls = int(np.argmax(scores)); score = float(scores[cls])
            if score < 0.28:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x = max(0.0, (cx - bw / 2 - px) / scale); y = max(0.0, (cy - bh / 2 - py) / scale)
            rw = min(float(w) - x, bw / scale); rh = min(float(h) - y, bh / scale)
            if rw > 1 and rh > 1:
                candidates.append(Region(NUDE_LABELS[cls], score, (x / w, y / h, rw / w, rh / h)))
        out: list[Region] = []
        for item in sorted(candidates, key=lambda r: r.score, reverse=True):
            if all(item.label != kept.label or self._iou(item, kept) < 0.45 for kept in out):
                out.append(item)
            if len(out) >= 24:
                break
        return out

    def structural_drift(self, before: list[Region], after: list[Region]) -> float:
        a = [r for r in before if r.label in SENSITIVE]; b = [r for r in after if r.label in SENSITIVE]
        if not a:
            return 0.0
        penalty = 0.0
        for r in a:
            same = [q for q in b if q.label == r.label]
            if not same:
                penalty += 1.0; continue
            penalty += max(0.0, 1.0 - max(self._iou(r, q) for q in same))
        count_delta = abs(len(a) - len(b)) / max(1, len(a))
        return min(1.0, 0.8 * penalty / max(1, len(a)) + 0.2 * count_delta)

def analyse_degradation(image: Image.Image) -> DegradationProfile:
    small = image.convert("RGB"); small.thumbnail((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    gray = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
    brightness = float(gray.mean()); contrast = float(gray.std())
    blurred = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=1.0))).astype(np.float32) / 255.0
    bgray = blurred[..., 0] * 0.2126 + blurred[..., 1] * 0.7152 + blurred[..., 2] * 0.0722
    noise = float(np.mean(np.abs(gray - bgray)))
    gx = np.diff(gray, axis=1, prepend=gray[:, :1]); gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    sharp_energy = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    blur = float(np.clip((0.055 - sharp_energy) / 0.055, 0, 1))
    dark = float(np.clip((0.38 - brightness) / 0.38, 0, 1)); flat = float(np.clip((0.16 - contrast) / 0.16, 0, 1))
    noisy = float(np.clip((noise - 0.012) / 0.05, 0, 1))
    severity = float(np.clip(0.32 * blur + 0.28 * noisy + 0.24 * dark + 0.16 * flat, 0, 1))
    return DegradationProfile(brightness, contrast, noise, blur, severity)

def gamma_polish(image: Image.Image, profile: DegradationProfile) -> Image.Image:
    out = image.convert("RGB")
    if profile.brightness < 0.34:
        gamma = max(0.78, 0.96 - (0.34 - profile.brightness) * 0.55)
        lut = [min(255, round(((i / 255.0) ** gamma) * 255)) for i in range(256)]
        out = out.point(lut * 3)
    if profile.contrast < 0.17:
        out = ImageEnhance.Contrast(out).enhance(1.04)
    return ImageEnhance.Color(out).enhance(1.025)

def global_fidelity_mix(original: Image.Image, restored: Image.Image, keep_original: float) -> Image.Image:
    base = original.convert("RGB").resize(restored.size, Image.Resampling.LANCZOS)
    low = base.filter(ImageFilter.GaussianBlur(radius=max(1.2, min(restored.size) / 700)))
    return Image.blend(restored, low, alpha=min(0.16, max(0.0, keep_original)))

class PixRestoreEngine:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        self.lock = threading.Lock(); self.accelerator = Accelerator(mixed_precision="fp16"); self.device = self.accelerator.device
        self.body = SilentBodyContext(); model_root = self._download_model(); config_path, weight_path = self._locate_model_files(model_root)
        config_data = read_config(config_path); state = read_state_dict(weight_path); config_data = apply_checkpoint_architecture_flags(config_data, state)
        config_data["mixed_precision"] = "fp16"
        if DINO_REPO: config_data["dinov2_repository"] = DINO_REPO
        if DINO_CHECKPOINT: config_data["dinov2_checkpoint"] = DINO_CHECKPOINT
        self.config = SimpleNamespace(**config_data)
        self.model = build_model(self.config); load_weights(self.model, weight_path, state=state)
        self.model.to(self.device).eval().requires_grad_(False); self.flow = build_flow(self.config, self.accelerator)
        self.encoder = None
        if self.config.use_venc:
            self.encoder = load_dinov2(self.config.encoder_type, self.device, repository=getattr(self.config, "dinov2_repository", None), checkpoint=getattr(self.config, "dinov2_checkpoint", None))
        print(f"[pixrestore] ready: {weight_path}", flush=True)

    def _download_model(self) -> Path:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        return Path(snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=str(MODEL_CACHE), token=os.getenv("HF_TOKEN") or None))

    @staticmethod
    def _locate_model_files(root: Path) -> tuple[Path, Path]:
        configs = list(root.rglob("config.json")); weights = list(root.rglob("*.safetensors")) + list(root.rglob("*.pt")) + list(root.rglob("*.pth"))
        if not configs or not weights: raise FileNotFoundError(f"PixRestore checkpoint files not found under {root}")
        def rank(path: Path) -> tuple[int, int, str]:
            s = str(path).lower(); preferred = any(k in s for k in ("gan", "one-step", "onestep", "1step")); ema = "ema_model" in path.name.lower()
            return (0 if preferred else 1, 0 if ema else 1, s)
        weight = sorted(weights, key=rank)[0]; config = None
        for parent in weight.parents:
            candidate = parent / "config.json"
            if candidate in configs: config = candidate; break
        return config or configs[0], weight

    @staticmethod
    def _prepare(image: Image.Image, patch_size: int, max_long: int = 2048) -> Image.Image:
        img = ImageOps.exif_transpose(image).convert("RGB"); long = max(img.size)
        if long > max_long:
            scale = max_long / long; img = img.resize((max(8, round(img.width * scale)), max(8, round(img.height * scale))), Image.Resampling.LANCZOS)
        w = img.width - img.width % patch_size; h = img.height - img.height % patch_size
        if w < patch_size or h < patch_size: raise ValueError("Image is too small")
        if (w, h) != img.size:
            left = (img.width - w) // 2; top = (img.height - h) // 2; img = img.crop((left, top, left + w, top + h))
        return img

    def restore(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        profile = analyse_degradation(image); prepared = self._prepare(image, self.config.patch_size); before_regions = self.body.detect(prepared)
        tensor = pil_to_tensor(prepared).unsqueeze(0).to(self.device).float().div_(127.5).sub_(1)
        infer_steps = int(os.getenv("PIXRESTORE_INFER_STEPS", "1")); schedule = os.getenv("PIXRESTORE_SCHEDULE", "linear"); seed = int(os.getenv("PIXRESTORE_SEED", "0"))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); started = time.time()
        with self.lock, torch.inference_mode(), self.accelerator.autocast():
            features = extract_layers(self.encoder, tensor, self.config.encoder_layers, self.config.encoder_input_size) if self.encoder is not None else None
            restored = self.flow.sample_multistep_fm(self.model, tensor, venc_fea=features, n_steps=infer_steps, schedule=schedule)
        out = to_pil_image(restored[0].float().cpu().add(1).mul(0.5).clamp(0, 1)); out = gamma_polish(out, profile)
        drift = self.body.structural_drift(before_regions, self.body.detect(out))
        if drift > 0.52: out = global_fidelity_mix(prepared, out, 0.12)
        if max(out.size) > MAX_OUTPUT_LONG:
            scale = MAX_OUTPUT_LONG / max(out.size); out = out.resize((round(out.width * scale), round(out.height * scale)), Image.Resampling.LANCZOS)
        return out, {"engine":"PixRestore","profile":"auto-wow-v1","input_size":list(prepared.size),"output_size":list(out.size),"severity":round(profile.severity,3),"structural_qa":"pass" if drift <= 0.52 else "global-fidelity-guard","seconds":round(time.time()-started,3)}

app = FastAPI(title="Dirty Studio Restoration Worker", docs_url=None, redoc_url=None)
ENGINE: PixRestoreEngine | None = None; STARTUP_ERROR: str | None = None

@app.on_event("startup")
def startup() -> None:
    global ENGINE, STARTUP_ERROR
    try: ENGINE = PixRestoreEngine()
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"; print(f"[startup] {STARTUP_ERROR}", flush=True)

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": ENGINE is not None, "engine": "PixRestore", "error": STARTUP_ERROR}

def decode_data_url(value: str) -> Image.Image:
    if not value.startswith("data:image/") or "," not in value: raise ValueError("Expected an image data URL")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    if len(raw) > 18 * 1024 * 1024: raise ValueError("Image payload is too large")
    image = Image.open(io.BytesIO(raw)); image.load(); return ImageOps.exif_transpose(image).convert("RGB")

def encode_result(image: Image.Image, target_bytes: int = 3_100_000) -> str:
    img = image.convert("RGB"); quality = 95
    for _ in range(8):
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=0); raw = buf.getvalue()
        if len(raw) <= target_bytes: return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        if quality > 82: quality -= 4
        else: img = img.resize((max(768, round(img.width * 0.88)), max(768, round(img.height * 0.88))), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def perform(req: EnhanceRequest) -> dict[str, Any]:
    if ENGINE is None: raise HTTPException(status_code=503, detail=STARTUP_ERROR or "Model is loading")
    try:
        output, meta = ENGINE.restore(decode_data_url(req.image)); return {"image": encode_result(output), "meta": meta}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500, detail=f"Restoration failed: {exc}") from exc

@app.post("/")
def root(req: EnhanceRequest) -> dict[str, Any]: return perform(req)

@app.post("/enhance")
def enhance(req: EnhanceRequest) -> dict[str, Any]: return perform(req)
