"""
Media Sorter — General-purpose AI photo & video organiser
Run: python app.py  →  browser opens at http://localhost:5000

Modes:
  Collect  — gather all files of selected type, no AI, one flat folder
  Sort     — AI detection → category subfolders (multi-copy optional)
  Filter   — AND logic: must match ALL selected categories
  By Date  — EXIF date → Year/Month subfolders
  By Location — GPS → City/Country subfolders

Detection stack:
  IR channel diff       → Night / Infrared
  Laplacian variance    → Damaged / Blurry
  EXIF tags             → Screenshots, Selfies, Panoramas
  MegaDetector v5       → Animals, People, Vehicles
  OpenCLIP ViT-B/32     → Pets, Nature, Sunset, Water, Buildings,
                           Flowers, Food, Documents, Snow, Indoor, Events
  Perceptual hash       → Duplicates
"""

import cv2, shutil, json, re, os, base64, time, threading, subprocess, webbrowser, urllib.request
import warnings, socket
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import psutil, torch
from PIL import Image, ExifTags

app = Flask(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
APP_DIR            = Path(__file__).parent
DOCS               = Path.home() / "Documents"
MD_MODEL_PATH      = str(APP_DIR / "md_v5a.0.0.pt")
MD_MODEL_URL       = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
CUSTOM_CATS_FILE   = APP_DIR / "custom_categories.json"
FACE_PROFILES_DIR  = APP_DIR / "face_profiles"
FACE_THRESHOLD     = 0.85   # L2 distance; lower = stricter

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_SECS    = [2, 4, 6, 8, 10]
IR_THRESHOLD   = 8
BLUR_THRESHOLD = 80
MD_CONFIDENCE  = 0.15
CLIP_THRESHOLD = 0.21
HASH_THRESHOLD = 8        # perceptual hash difference for duplicates

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# EXIF numeric tag IDs
ETAG_DATE     = 36867   # DateTimeOriginal
ETAG_GPS      = 34853   # GPSInfo
ETAG_SOFTWARE = 305
ETAG_LENS     = 42036   # LensModel
ETAG_SCENE    = 41990   # SceneCaptureType  3=panorama
ETAG_FLASH    = 37385

# ── category registry ─────────────────────────────────────────────────────────
CATEGORIES = [
    # ── fast / no-AI ──
    {"key": "night",      "label": "Night / IR",    "emoji": "🌙", "color": "#7c5cbf",
     "detector": "ir"},
    {"key": "damaged",    "label": "Damaged",        "emoji": "⚠️",  "color": "#bf5c5c",
     "detector": "blur"},
    {"key": "screenshot", "label": "Screenshots",   "emoji": "📱", "color": "#5a7a9a",
     "detector": "exif", "exif_type": "screenshot"},
    {"key": "selfie",     "label": "Selfies",        "emoji": "🤳", "color": "#9a5a7a",
     "detector": "exif", "exif_type": "selfie"},
    {"key": "panorama",   "label": "Panoramas",      "emoji": "🌄", "color": "#5a8a7a",
     "detector": "exif", "exif_type": "panorama"},
    {"key": "duplicate",  "label": "Duplicates",     "emoji": "👥", "color": "#7a6a5a",
     "detector": "hash"},
    # ── MegaDetector ──
    {"key": "animal",     "label": "Animals",        "emoji": "🐾", "color": "#3a8a3a",
     "detector": "megadetector", "md_class": 0},
    {"key": "person",     "label": "People",         "emoji": "🚶", "color": "#3a5abf",
     "detector": "megadetector", "md_class": 1},
    {"key": "vehicle",    "label": "Vehicles",       "emoji": "🚗", "color": "#bf8c3a",
     "detector": "megadetector", "md_class": 2},
    # ── CLIP ──
    {"key": "pets",       "label": "Pets",           "emoji": "🐕", "color": "#c8782a",
     "detector": "clip",
     "clip_prompt": "a photo of a pet dog or cat"},
    {"key": "nature",     "label": "Nature",         "emoji": "🌿", "color": "#4a9a4a",
     "detector": "clip",
     "clip_prompt": "a photo of nature, forest, wilderness, mountains, lake, river, or natural landscape outdoors"},
    {"key": "sunset",     "label": "Sunsets",        "emoji": "🌅", "color": "#bf6030",
     "detector": "clip",
     "clip_prompt": "a beautiful photo of a sunset or sunrise with colorful orange pink or red sky"},
    {"key": "water",      "label": "Water / Beach",  "emoji": "🌊", "color": "#3a7abf",
     "detector": "clip",
     "clip_prompt": "a photo of ocean, beach, sea, waves, lake, or large body of water"},
    {"key": "buildings",  "label": "Buildings",      "emoji": "🏛️", "color": "#8a7a5a",
     "detector": "clip",
     "clip_prompt": "a photo of buildings, architecture, houses, city, streets, or urban environment"},
    {"key": "flowers",    "label": "Flowers",        "emoji": "🌸", "color": "#d060a0",
     "detector": "clip",
     "clip_prompt": "a close-up photo of flowers, plants, or botanical subjects"},
    {"key": "food",       "label": "Food",           "emoji": "🍽️", "color": "#bf6a3a",
     "detector": "clip",
     "clip_prompt": "a close-up photo of food, a meal, drink, snack, or cuisine on a plate or table"},
    {"key": "documents",  "label": "Documents",      "emoji": "📄", "color": "#7a8a9a",
     "detector": "clip",
     "clip_prompt": "a photo of a document, receipt, menu, whiteboard, book page, or printed text"},
    {"key": "snow",       "label": "Snow / Winter",  "emoji": "❄️", "color": "#8aaabb",
     "detector": "clip",
     "clip_prompt": "a photo in snow, winter landscape, or cold weather scene with snow or ice"},
    {"key": "indoor",     "label": "Indoor",         "emoji": "🏠", "color": "#5a7abf",
     "detector": "clip",
     "clip_prompt": "a photo taken indoors, inside a room, house interior, or building interior"},
    {"key": "events",     "label": "Events",         "emoji": "🎉", "color": "#bf4a8a",
     "detector": "clip",
     "clip_prompt": "a photo at a party, celebration, birthday, wedding, or social gathering event"},
]

# ── load persisted custom categories ─────────────────────────────────────────
def _load_custom_categories():
    if CUSTOM_CATS_FILE.exists():
        try:
            customs = json.loads(CUSTOM_CATS_FILE.read_text(encoding="utf-8"))
            existing_keys = {c["key"] for c in CATEGORIES}
            for c in customs:
                if c.get("key") and c["key"] not in existing_keys:
                    CATEGORIES.append(c)
        except Exception:
            pass

_load_custom_categories()


# ── face recognition helpers ──────────────────────────────────────────────────
FACE_PROFILES_DIR.mkdir(exist_ok=True)

_face_detector = None
_face_embedder = None
_face_lock     = threading.Lock()


def load_face_models():
    global _face_detector, _face_embedder
    with _face_lock:
        if _face_detector is None:
            push({"type": "log", "text": "Loading face recognition models (~110 MB one-time download)...", "cat": "info"})
            from facenet_pytorch import MTCNN, InceptionResnetV1
            _face_detector = MTCNN(keep_all=True, device="cpu", post_process=True)
            _face_embedder = InceptionResnetV1(pretrained="vggface2").eval()
            push({"type": "log", "text": "Face recognition ready.", "cat": "info"})
    return _face_detector, _face_embedder


def get_face_embeddings(frame):
    """Returns list of numpy embedding arrays, one per detected face."""
    try:
        detector, embedder = load_face_models()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces   = detector(pil_img)
        if faces is None or len(faces) == 0:
            return []
        with torch.no_grad():
            return [e.numpy() for e in embedder(faces)]
    except Exception:
        return []


def load_face_profiles():
    """Returns {safe_name: [np.array, ...]} loaded from .npy files."""
    profiles = {}
    for f in sorted(FACE_PROFILES_DIR.glob("*.npy")):
        try:
            profiles[f.stem] = list(np.load(str(f), allow_pickle=False))
        except Exception:
            pass
    return profiles


def check_faces(fp, name, frames, out, i, total, face_profiles, counts, folders, thumb_name):
    """Check frames against all face profiles and copy matches."""
    if not face_profiles or not frames:
        return
    # Only scan first 2 frames for speed
    all_embs = []
    for frame in frames[:2]:
        all_embs.extend(get_face_embeddings(frame))
    if not all_embs:
        return
    for profile_name, ref_embs in face_profiles.items():
        best_dist = float("inf")
        for emb in all_embs:
            for ref in ref_embs:
                d = float(np.linalg.norm(emb - ref))
                best_dist = min(best_dist, d)
        if best_dist < FACE_THRESHOLD:
            face_dir = out / f"People_{profile_name}"
            face_dir.mkdir(parents=True, exist_ok=True)
            face_key = f"face_{profile_name.lower()}"
            shutil.copy2(str(fp), face_dir / name)
            folders[face_key] = str(face_dir)
            counts[face_key]  = counts.get(face_key, 0) + 1
            conf = round(max(0.0, 1.0 - best_dist / FACE_THRESHOLD), 2)
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "face_match",
                  "label": profile_name, "conf": conf, "thumb": thumb_name})

# ── global state ──────────────────────────────────────────────────────────────
_md_model     = None
_clip_model   = None
_clip_preproc = None
_clip_feats   = {}
_model_lock   = threading.Lock()
_sort_thread  = None
_stop_event   = threading.Event()
_log          = []
_log_lock     = threading.Lock()
_results      = {}
_status       = "idle"
_seen_hashes  = {}   # filename → hash string, for duplicate detection


def push(msg):
    with _log_lock:
        _log.append(msg)


# ── model loading ─────────────────────────────────────────────────────────────
def _download_megadetector():
    push({"type": "log", "text": "Downloading MegaDetector (~290 MB) — one time only…", "cat": "info"})
    def prog(c, b, t):
        pct = min(c * b / t * 100, 100)
        if int(pct) % 10 == 0:
            push({"type": "log", "text": f"  {pct:.0f}% downloaded…", "cat": "info"})
    urllib.request.urlretrieve(MD_MODEL_URL, MD_MODEL_PATH, reporthook=prog)


def load_megadetector():
    global _md_model
    with _model_lock:
        if _md_model is None:
            mp = Path(MD_MODEL_PATH)
            if not mp.exists() or mp.stat().st_size < 50_000_000:
                if mp.exists(): mp.unlink()
                _download_megadetector()
            push({"type": "log", "text": "Loading MegaDetector…", "cat": "info"})
            m = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=MD_MODEL_PATH, force_reload=False, verbose=False, trust_repo=True,
            )
            m.conf = MD_CONFIDENCE
            m.eval()
            _md_model = m
            push({"type": "log", "text": "MegaDetector ready.", "cat": "info"})
    return _md_model


def load_clip(enabled_keys):
    global _clip_model, _clip_preproc, _clip_feats
    clip_cats = [c for c in CATEGORIES if c["detector"] == "clip" and c["key"] in enabled_keys]
    if not clip_cats:
        return
    with _model_lock:
        if _clip_model is None:
            push({"type": "log", "text": "Loading CLIP model (first run downloads ~350 MB)…", "cat": "info"})
            import open_clip
            _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_model.eval()
            push({"type": "log", "text": "CLIP ready.", "cat": "info"})
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for cat in clip_cats:
            if cat["key"] not in _clip_feats:
                tokens = tokenizer([cat["clip_prompt"]])
                with torch.no_grad():
                    feat = _clip_model.encode_text(tokens)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                _clip_feats[cat["key"]] = feat


# ── EXIF helpers ──────────────────────────────────────────────────────────────
def get_exif(image_path):
    try:
        img = Image.open(image_path)
        raw = img._getexif()
        if raw:
            return {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        pass
    return {}


def exif_is_screenshot(exif, filepath):
    sw = str(exif.get("Software", "")).lower()
    if "screenshot" in sw:
        return True
    try:
        img = Image.open(filepath)
        w, h = img.size
        # Common phone screen resolutions
        screen_res = {(1080,2340),(1080,2400),(1170,2532),(1284,2778),
                      (1080,1920),(720,1280),(1440,3088),(1440,3200)}
        if (w, h) in screen_res or (h, w) in screen_res:
            return True
    except Exception:
        pass
    return False


def exif_is_selfie(exif):
    lens = str(exif.get("LensModel", "")).lower()
    return "front" in lens


def exif_is_panorama(exif):
    return exif.get("SceneCaptureType") == 3


def get_photo_date(filepath):
    """Returns datetime from EXIF, or file mtime as fallback."""
    exif = get_exif(filepath)
    date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
    if date_str:
        try:
            return datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    return datetime.fromtimestamp(Path(filepath).stat().st_mtime)


def get_photo_gps(filepath):
    """Returns (lat, lon) or (None, None)."""
    exif = get_exif(filepath)
    gps  = exif.get("GPSInfo")
    if not gps or not isinstance(gps, dict):
        return None, None
    try:
        def to_deg(vals):
            d, m, s = float(vals[0]), float(vals[1]), float(vals[2])
            return d + m / 60 + s / 3600
        lat = to_deg(gps.get(2, [0, 0, 0]))
        lon = to_deg(gps.get(4, [0, 0, 0]))
        if gps.get(1) == "S": lat = -lat
        if gps.get(3) == "W": lon = -lon
        return lat, lon
    except Exception:
        return None, None


def reverse_geocode(lat, lon):
    """Returns 'CityName_CC' string or None."""
    try:
        import reverse_geocoder as rg
        res = rg.search((lat, lon), mode=1)
        if res:
            r = res[0]
            city = r.get("name", "Unknown")
            cc   = r.get("cc", "")
            return f"{city}_{cc}".replace(" ", "_")
    except Exception:
        pass
    return None


def compute_phash(frame):
    """Perceptual hash of a frame (BGR numpy array)."""
    try:
        import imagehash
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return imagehash.phash(img)
    except Exception:
        return None


# ── video/image frame extraction ──────────────────────────────────────────────
def extract_frames(video_path):
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for sec in SAMPLE_SECS:
        idx = min(int(sec * fps), total - 1)
        if idx < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def load_frames(filepath):
    """Returns list of frames regardless of file type."""
    ext = Path(filepath).suffix.upper()
    if ext in {e.upper() for e in IMAGE_EXTS}:
        img = cv2.imread(str(filepath))
        return [img] if img is not None else []
    return extract_frames(str(filepath))


# ── per-frame detectors ───────────────────────────────────────────────────────
def is_infrared(frames):
    hits = sum(
        1 for f in frames
        if (lambda b, g, r:
            (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3 < IR_THRESHOLD
        )(*cv2.split(f.astype(np.float32)))
    )
    return hits >= max(1, len(frames) // 2 + 1)


def is_blurry(frames):
    hits = sum(
        1 for f in frames
        if cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < BLUR_THRESHOLD
    )
    return hits >= max(1, len(frames) // 2 + 1)


def run_megadetector(model, frames, md_class, confidence):
    best = 0.0
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for d in model(rgb, size=640).xyxy[0]:
            if int(d[5]) == md_class:
                best = max(best, float(d[4]))
    return best >= confidence, best


def run_clip_cat(frames, category_key, threshold=CLIP_THRESHOLD):
    if _clip_model is None or category_key not in _clip_feats:
        return False, 0.0
    text_feat = _clip_feats[category_key]
    best = 0.0
    for frame in frames:
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = _clip_preproc(img).unsqueeze(0)
        with torch.no_grad():
            img_feat = _clip_model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim  = (img_feat @ text_feat.T).item()
        best = max(best, sim)
    return best >= threshold, best


# ── classify → list of all matches ───────────────────────────────────────────
def classify_all(filepath, frames, enabled_cats, md_model, confidence, seen_hashes,
                 clip_threshold=CLIP_THRESHOLD):
    """
    Returns list of (category_key, confidence) for every category that matches.
    Ordered by priority.  Caller decides whether to use first-only or all.
    """
    matches = []
    enabled = {c["key"] for c in enabled_cats}

    for cat in enabled_cats:
        key = cat["key"]
        det = cat["detector"]

        if det == "ir":
            if is_infrared(frames):
                matches.append((key, 1.0))

        elif det == "blur":
            ext = Path(filepath).suffix.upper()
            if ext in {e.upper() for e in IMAGE_EXTS} and is_blurry(frames):
                matches.append((key, 1.0))

        elif det == "exif":
            ext = Path(filepath).suffix.upper()
            if ext in {e.upper() for e in IMAGE_EXTS}:
                exif = get_exif(filepath)
                etype = cat.get("exif_type")
                if etype == "screenshot" and exif_is_screenshot(exif, filepath):
                    matches.append((key, 1.0))
                elif etype == "selfie" and exif_is_selfie(exif):
                    matches.append((key, 1.0))
                elif etype == "panorama" and exif_is_panorama(exif):
                    matches.append((key, 1.0))

        elif det == "hash":
            if frames:
                h = compute_phash(frames[0])
                if h is not None:
                    for existing_name, existing_h in seen_hashes.items():
                        if abs(h - existing_h) <= HASH_THRESHOLD:
                            matches.append((key, 1.0))
                            break

        elif det == "megadetector" and md_model:
            found, conf = run_megadetector(md_model, frames, cat["md_class"], confidence)
            if found:
                matches.append((key, conf))

        elif det == "clip":
            found, conf = run_clip_cat(frames, key, threshold=clip_threshold)
            if found:
                matches.append((key, conf))

    # Store hash for future duplicate detection
    if "duplicate" in enabled and frames:
        h = compute_phash(frames[0])
        if h is not None:
            seen_hashes[Path(filepath).name] = h

    return matches


# ── sort worker ───────────────────────────────────────────────────────────────
def run_sort(source_dir, output_dir, confidence, enabled_keys, file_types,
             mode, multi_copy, filter_all):
    global _status, _results, _seen_hashes

    out          = Path(output_dir)
    enabled_cats = [c for c in CATEGORIES if c["key"] in enabled_keys]

    # Load models only if needed
    need_md   = mode in ("sort","filter") and any(c["detector"]=="megadetector" for c in enabled_cats)
    need_clip = mode in ("sort","filter") and any(c["detector"]=="clip"          for c in enabled_cats)

    _status       = "loading"
    _seen_hashes  = {}
    md_model      = load_megadetector() if need_md   else None
    if md_model: md_model.conf = confidence
    if need_clip: load_clip(enabled_keys)
    face_profiles = load_face_profiles() if mode in ("sort", "filter") else {}
    need_faces    = bool(face_profiles) and mode in ("sort", "filter")
    if need_faces and face_profiles:
        load_face_models()   # warm up before the loop
    _status = "running"

    # Gather files
    exts = set()
    if "video" in file_types: exts |= {e.upper() for e in VIDEO_EXTS}
    if "image" in file_types: exts |= {e.upper() for e in IMAGE_EXTS}

    # Recursive scan — skip anything already inside the output folder
    src_path = Path(source_dir)
    out_upper = str(out).upper()
    files = []
    for p in src_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.upper() not in exts:
            continue
        if str(p).upper().startswith(out_upper + os.sep.upper()):
            continue   # don't re-process our own output
        files.append(p)
    files.sort(key=lambda p: str(p).upper())
    total  = len(files)

    counts    = {}
    thumb_map = {}
    folders   = {}

    push({"type": "start", "total": total, "mode": mode, "categories": [
        {"key": c["key"], "label": c["label"], "emoji": c["emoji"], "color": c["color"]}
        for c in enabled_cats
    ]})

    thumbs_dir = out / "thumbs"

    for i, fp in enumerate(files, 1):
        if _stop_event.is_set():
            push({"type": "log", "text": "Stopped by user.", "cat": "info"})
            break

        name = fp.name

        # ── COLLECT mode: just copy everything ───────────────────────────────
        if mode == "collect":
            dest = out / "Collected"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fp), dest / name)
            counts["collected"] = counts.get("collected", 0) + 1
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "collected", "conf": 0, "thumb": None})
            continue

        # ── DATE mode: EXIF date → Year/Month folders ─────────────────────
        if mode == "date":
            dt   = get_photo_date(str(fp))
            folder_name = dt.strftime("%Y/%B")
            dest = out / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fp), dest / name)
            key = dt.strftime("%Y-%m")
            counts[key] = counts.get(key, 0) + 1
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "date", "conf": 0,
                  "label": dt.strftime("%Y / %B"), "thumb": None})
            continue

        # ── LOCATION mode: GPS → City folders ─────────────────────────────
        if mode == "location":
            lat, lon = get_photo_gps(str(fp))
            if lat is not None:
                loc = reverse_geocode(lat, lon) or "Unknown"
            else:
                loc = "No_GPS"
            dest = out / loc
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fp), dest / name)
            counts[loc] = counts.get(loc, 0) + 1
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "location", "conf": 0,
                  "label": loc.replace("_", " "), "thumb": None})
            continue

        # ── SORT / FILTER mode: AI detection ─────────────────────────────
        frames = load_frames(str(fp))
        if not frames:
            counts["skip"] = counts.get("skip", 0) + 1
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "skip", "conf": 0, "thumb": None})
            continue

        # CLIP threshold scales with slider: default 15%→0.21, higher = stricter
        clip_thresh = max(0.15, CLIP_THRESHOLD + (confidence - MD_CONFIDENCE))
        all_matches = classify_all(str(fp), frames, enabled_cats, md_model,
                                   confidence, _seen_hashes,
                                   clip_threshold=clip_thresh)

        if mode == "filter":
            # AND logic: only copy if ALL enabled detector categories matched
            det_keys = {c["key"] for c in enabled_cats if c["detector"] != "hash"}
            matched_keys = {m[0] for m in all_matches}
            if not det_keys.issubset(matched_keys):
                counts["other"] = counts.get("other", 0) + 1
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "other", "conf": 0, "thumb": None})
                continue
            # Passes the AND filter — copy to a combined folder
            label = "_AND_".join(sorted(det_keys))
            dest  = out / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fp), dest / name)
            counts[label] = counts.get(label, 0) + 1
            conf = all_matches[0][1] if all_matches else 0
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "filter_match", "conf": round(conf,2),
                  "label": label, "thumb": None})
            continue

        # ── SORT mode ─────────────────────────────────────────────────────
        if not all_matches:
            unsorted_dir = out / "Unsorted"
            unsorted_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(fp), unsorted_dir / name)
            folders["unsorted"] = str(unsorted_dir)
            counts["unsorted"] = counts.get("unsorted", 0) + 1
            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": "unsorted", "conf": 0, "thumb": None})
            check_faces(fp, name, frames, out, i, total, face_profiles,
                        counts, folders, None)
            continue

        # Multi-copy: copy to every matched category, or just the first
        to_copy = all_matches if multi_copy else [all_matches[0]]
        thumb_saved = False

        for cat_key, conf in to_copy:
            cat_dir = out / cat_key.capitalize()
            cat_dir.mkdir(parents=True, exist_ok=True)
            folders[cat_key] = str(cat_dir)
            shutil.copy2(str(fp), cat_dir / name)
            counts[cat_key] = counts.get(cat_key, 0) + 1
            if cat_key not in thumb_map:
                thumb_map[cat_key] = []

            thumb_name = None
            if not thumb_saved:
                thumbs_dir.mkdir(parents=True, exist_ok=True)
                thumb_name = fp.stem + ".jpg"
                small = cv2.resize(frames[0],
                    (320, int(frames[0].shape[0] * 320 / max(frames[0].shape[1], 1))))
                cv2.imwrite(str(thumbs_dir / thumb_name), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 75])
                thumb_map[cat_key].append(thumb_name)
                thumb_saved = True

            push({"type": "progress", "current": i, "total": total,
                  "file": name, "cat": cat_key, "conf": round(conf, 2),
                  "thumb": thumb_name if not thumb_saved or cat_key == to_copy[0][0] else None})

        check_faces(fp, name, frames, out, i, total, face_profiles,
                    counts, folders, thumb_name)

    _results = {"counts": counts, "thumbs": thumb_map, "folders": folders, "mode": mode}
    push({"type": "done", "results": _results})
    _status = "done"


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    default_output = r"C:\MediaSorted"
    return render_template("index.html", categories=CATEGORIES, default_output=default_output)


@app.route("/api/categories")
def get_categories():
    return jsonify(CATEGORIES)


@app.route("/api/drives")
def get_drives():
    drives = []
    try:
        for p in psutil.disk_partitions(all=True):
            opts = p.opts.lower()
            fs   = p.fstype.upper()
            if "removable" in opts or fs in ("FAT32", "EXFAT", "FAT", "UDF"):
                path = p.mountpoint.rstrip("\\")
                if path not in drives:
                    drives.append(path)
    except Exception:
        pass
    return jsonify(drives)


@app.route("/api/start", methods=["POST"])
def start():
    global _sort_thread, _stop_event, _log, _results, _status
    if _sort_thread and _sort_thread.is_alive():
        return jsonify({"error": "Already running"})
    data         = request.json or {}
    source       = data.get("source", "")
    output       = data.get("output", str(DOCS / "MediaSorted"))
    confidence   = float(data.get("confidence", 0.15))
    enabled_keys = set(data.get("enabled", [c["key"] for c in CATEGORIES]))
    file_types   = set(data.get("fileTypes", ["video", "image"]))
    mode         = data.get("mode", "sort")          # collect|sort|filter|date|location
    multi_copy   = bool(data.get("multiCopy", False))
    filter_all   = bool(data.get("filterAll", True))
    _stop_event  = threading.Event()
    _log         = []
    _results     = {}
    _status      = "running"
    _sort_thread = threading.Thread(
        target=run_sort,
        args=(source, output, confidence, enabled_keys, file_types,
              mode, multi_copy, filter_all),
        daemon=True,
    )
    _sort_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop():
    _stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/status")
def status():
    return jsonify({"status": _status})


@app.route("/api/stream")
def stream():
    def generate():
        sent = 0
        while True:
            with _log_lock:
                chunk = _log[sent:]
            for msg in chunk:
                yield f"data: {json.dumps(msg)}\n\n"
                sent += 1
            if _status in ("done", "idle") and sent >= len(_log):
                break
            time.sleep(0.15)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    folder = (request.json or {}).get("folder", "")
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    # shell=True routes through Windows start, which grants foreground focus
    subprocess.Popen(f'start "" "{str(p)}"', shell=True)
    return jsonify({"ok": True})


@app.route("/api/categories/add", methods=["POST"])
def add_category():
    data   = request.json or {}
    name   = data.get("name", "").strip()
    prompt = data.get("prompt", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if not prompt:
        prompt = f"a photo of {name.lower()}"
    # Build a safe unique key
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:24]
    existing_keys = {c["key"] for c in CATEGORIES}
    if key in existing_keys:
        key = key + "_custom"
    cat = {
        "key":         key,
        "label":       name,
        "emoji":       "🏷️",
        "color":       "#7a6a9a",
        "detector":    "clip",
        "clip_prompt": prompt,
        "custom":      True,
    }
    CATEGORIES.append(cat)
    # Persist all custom categories to disk
    customs = [c for c in CATEGORIES if c.get("custom")]
    CUSTOM_CATS_FILE.write_text(json.dumps(customs, indent=2), encoding="utf-8")
    return jsonify(cat)


@app.route("/thumbs/<path:fname>")
def thumb(fname):
    base = list(_results.get("folders", {}).values())
    thumbs = (Path(base[0]).parent / "thumbs") if base else (DOCS / "MediaSorted" / "thumbs")
    return send_from_directory(str(thumbs), fname)



@app.route("/api/faces")
def list_faces():
    profiles = []
    for f in sorted(FACE_PROFILES_DIR.glob("*.npy")):
        try:
            data = np.load(str(f), allow_pickle=False)
            profiles.append({"safe_name": f.stem, "face_count": len(data)})
        except Exception:
            pass
    return jsonify(profiles)


@app.route("/api/faces/add", methods=["POST"])
def add_face():
    data       = request.json or {}
    name       = data.get("name", "").strip()
    images_b64 = data.get("images", [])
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not images_b64:
        return jsonify({"error": "At least one photo is required."}), 400

    all_embeddings = []
    for img_b64 in images_b64:
        try:
            raw   = base64.b64decode(img_b64.split(",")[-1])
            arr   = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                all_embeddings.extend(get_face_embeddings(frame))
        except Exception:
            pass

    if not all_embeddings:
        return jsonify({"error": "No faces detected. Use a clear, well-lit, front-facing photo."}), 400

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")[:40]
    np.save(str(FACE_PROFILES_DIR / f"{safe_name}.npy"), np.array(all_embeddings))
    return jsonify({"safe_name": safe_name, "face_count": len(all_embeddings)})


@app.route("/api/faces/delete", methods=["POST"])
def delete_face():
    safe_name = (request.json or {}).get("safe_name", "")
    f = FACE_PROFILES_DIR / f"{safe_name}.npy"
    if f.exists():
        f.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "your-pc-ip"
    threading.Timer(1.3, lambda: webbrowser.open("http://localhost:5000")).start()
    print("=" * 54)
    print("  Media Sorter")
    print(f"  This PC       →  http://localhost:5000")
    print(f"  Phone / tablet →  http://{local_ip}:5000")
    print("  (phone must be on the same WiFi — no internet needed)")
    print("=" * 54)
    app.run(host="0.0.0.0", port=5000, threaded=True)
