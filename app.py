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

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass   # HEIC support optional — install pillow-heif to enable

app = Flask(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
APP_DIR            = Path(__file__).parent
DOCS               = Path.home() / "Documents"
MD_MODEL_PATH      = str(APP_DIR / "md_v5a.0.0.pt")
MD_MODEL_URL       = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
CUSTOM_CATS_FILE      = APP_DIR / "custom_categories.json"
CAT_THRESHOLDS_FILE   = APP_DIR / "category_thresholds.json"
FACE_PROFILES_DIR     = APP_DIR / "face_profiles"
FACE_THRESHOLD     = 0.85   # L2 distance; lower = stricter

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_SECS    = [2, 4, 6, 8, 10]
IR_THRESHOLD   = 12      # mean channel diff threshold; raised to catch trail-cam IR with slight LED cast
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

# ── per-category CLIP thresholds (user-adjustable, persisted) ─────────────
_cat_thresholds = {}   # key → absolute float threshold (overrides global CLIP_THRESHOLD)

def _load_cat_thresholds():
    global _cat_thresholds
    if CAT_THRESHOLDS_FILE.exists():
        try:
            _cat_thresholds = json.loads(CAT_THRESHOLDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

_load_cat_thresholds()


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
_scan_id      = 0    # incremented on every new scan start; SSE generators exit when their id is stale
_seen_hashes  = {}   # filename → hash string, for duplicate detection
_ref_embedding = None   # CLIP image feature for visual similarity search
_ref_name      = ""     # sanitised filename stem of the reference image
_uncertain     = []     # files sorted with conf < UNCERTAIN_THRESHOLD, for review screen
_dup_files     = []     # files placed in Duplicates/ folder, for review screen

UNCERTAIN_THRESHOLD = 0.40


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


def load_clip(enabled_keys=None):
    """Load CLIP model. If enabled_keys provided, also encode text prompts for those categories."""
    global _clip_model, _clip_preproc, _clip_feats
    with _model_lock:
        if _clip_model is None:
            push({"type": "log", "text": "Loading CLIP model (first run downloads ~350 MB)…", "cat": "info"})
            import open_clip
            _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _clip_model.eval()
            push({"type": "log", "text": "CLIP ready.", "cat": "info"})
    if not enabled_keys:
        return
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_cats = [c for c in CATEGORIES if c["detector"] == "clip" and c["key"] in enabled_keys]
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


def compute_quick_hash(filepath):
    """
    Fast exact-copy fingerprint for videos: file size + MD5 of first and last 1 MB.

    Why not pHash for videos?  Trail cameras are stationary — the background never
    changes.  Two completely different recordings of the same empty field will produce
    nearly identical perceptual hashes regardless of how many frames are sampled,
    causing massive false positives.

    Exact-copy detection is the right goal for video deduplication: the same file
    backed up to two locations will be byte-for-byte identical (same size, same bytes).
    Different recordings are different sizes because video bitrate varies with motion.
    """
    import hashlib
    sample = 1024 * 1024   # read first + last 1 MB
    try:
        size = Path(filepath).stat().st_size
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            h.update(f.read(min(sample, size)))
            if size > sample * 2:
                f.seek(-sample, 2)   # seek 1 MB from end
                h.update(f.read(sample))
        return (size, h.hexdigest())
    except Exception:
        return None


def compute_phash_list(frames):
    """Return list of pHash objects for all frames (used for video dedupe)."""
    return [h for h in (compute_phash(f) for f in frames) if h is not None]


def is_video_dup(new_hashes, seen_hash_lists):
    """
    True only if:
      1. Both videos produced the same number of sampled frames (same duration),  AND
      2. Every frame pair matches within HASH_THRESHOLD.

    Requiring equal frame counts eliminates the 'short-video contamination' bug:
    a 2-second clip only extracts 1 frame.  Using min(1, 5)=1 would reduce
    comparison to a single frame for ALL longer videos tested against it, recreating
    the original false-positive problem.  Genuine duplicates (same file copied/renamed)
    will always extract the same number of frames.  Different-length clips are not
    duplicates by definition.
    """
    if not new_hashes:
        return False
    for existing in seen_hash_lists:
        if not existing:
            continue
        # Different frame counts → different durations → cannot be a duplicate
        if len(new_hashes) != len(existing):
            continue
        if all(abs(new_hashes[j] - existing[j]) <= HASH_THRESHOLD
               for j in range(len(new_hashes))):
            return True
    return False


# ── video/image frame extraction ──────────────────────────────────────────────
def extract_frames(video_path):
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    seen_idx = set()   # prevent duplicate frames from short videos getting clamped to same index
    for sec in SAMPLE_SECS:
        idx = int(sec * fps)
        if idx >= total or idx < 0:
            continue           # skip sample points beyond the video duration
        if idx in seen_idx:
            continue           # skip if already sampled this exact frame
        seen_idx.add(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def load_frames(filepath, timeout_sec=30):
    """
    Returns list of frames regardless of file type.
    Runs in a thread with a timeout so a single corrupt or enormous file
    cannot stall the entire scan indefinitely.
    """
    import concurrent.futures
    def _load():
        ext = Path(filepath).suffix.upper()
        if ext in {e.upper() for e in IMAGE_EXTS}:
            img = cv2.imread(str(filepath))
            return [img] if img is not None else []
        return extract_frames(str(filepath))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_load)
            return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        log(f"[TIMEOUT] Skipped (took >{timeout_sec}s): {Path(filepath).name}")
        return []
    except Exception:
        return []


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


def write_xmp_sidecar(dest_path, labels, confidence):
    """Write an XMP sidecar file alongside dest_path with AI category tags."""
    xmp_path = Path(dest_path).with_suffix(".xmp")
    tags_xml = "".join(f"        <rdf:li>{lbl}</rdf:li>\n" for lbl in labels)
    xmp = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Media Sorter">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '      xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '      xmlns:xmp="http://ns.adobe.com/xap/1.0/">\n'
        '   <dc:subject>\n'
        '    <rdf:Bag>\n'
        f'      <rdf:li>MediaSorter</rdf:li>\n'
        f'{tags_xml}'
        '    </rdf:Bag>\n'
        '   </dc:subject>\n'
        f'   <xmp:Rating>{max(1, min(5, round(confidence * 5)))}</xmp:Rating>\n'
        '   <xmp:CreatorTool>Media Sorter</xmp:CreatorTool>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )
    try:
        xmp_path.write_text(xmp, encoding="utf-8")
    except Exception:
        pass


def encode_image_clip(frame):
    """Encode a BGR frame with CLIP image encoder. Returns normalised tensor or None."""
    if _clip_model is None or _clip_preproc is None:
        return None
    try:
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = _clip_preproc(img).unsqueeze(0)
        with torch.no_grad():
            feat = _clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat
    except Exception:
        return None


def run_clip_similarity(frames, ref_feat, threshold=0.70):
    """Image-to-image CLIP similarity. Returns (match, best_score)."""
    best = 0.0
    for frame in frames:
        feat = encode_image_clip(frame)
        if feat is None:
            continue
        sim  = float((feat @ ref_feat.T).item())
        best = max(best, sim)
    return best >= threshold, round(best, 3)


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
                hl = compute_phash_list(frames)
                if hl and is_video_dup(hl, list(seen_hashes.values())):
                    matches.append((key, 1.0))

        elif det == "megadetector" and md_model:
            found, conf = run_megadetector(md_model, frames, cat["md_class"], confidence)
            if found:
                matches.append((key, conf))

        elif det == "clip":
            # Per-category override takes precedence over the global CLIP threshold
            thresh = _cat_thresholds.get(key, clip_threshold)
            found, conf = run_clip_cat(frames, key, threshold=thresh)
            if found:
                matches.append((key, conf))

    # Store hash-list for future duplicate detection (multi-frame fingerprint)
    if "duplicate" in enabled and frames:
        hl = compute_phash_list(frames)
        if hl:
            seen_hashes[Path(filepath).name] = hl

    return matches


# ── sort worker ───────────────────────────────────────────────────────────────
def run_sort(source_dir, output_dir, confidence, enabled_keys, file_types,
             mode, multi_copy, filter_all, write_xmp=False, date_in_cat=False,
             copy_unsorted=False):
    global _status, _results, _seen_hashes, _uncertain, _dup_files
    _uncertain      = []
    _dup_files      = []
    seen_hash_objs   = []    # pHash lists for IMAGE dedupe
    seen_video_hashes = set() # (size, md5) tuples for VIDEO dedupe — exact-copy only

    out          = Path(output_dir)
    enabled_cats = [c for c in CATEGORIES if c["key"] in enabled_keys]

    # Load models only if needed
    need_md   = mode in ("sort","filter") and any(c["detector"]=="megadetector" for c in enabled_cats)
    need_clip = (mode in ("sort","filter") and any(c["detector"]=="clip" for c in enabled_cats)) \
                or mode == "similar"

    _status       = "loading"
    _seen_hashes  = {}
    md_model      = load_megadetector() if need_md   else None
    if md_model: md_model.conf = confidence
    if need_clip:
        load_clip(enabled_keys if mode != "similar" else None)
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
    src_upper = str(src_path).upper()
    # If source is a subfolder of output (e.g. output=D:\sorted, source=D:\sorted\Unique)
    # the prefix check would exclude every file.  In that case skip the check — rglob is
    # already scoped to source so we can't accidentally re-process other output subfolders.
    source_inside_output = src_upper.startswith(out_upper + os.sep.upper())
    files = []
    for p in src_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.upper() not in exts:
            continue
        if not source_inside_output and str(p).upper().startswith(out_upper + os.sep.upper()):
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

    # ── EVENT mode: sort by date, group into events, copy ─────────────────────
    if mode == "event":
        EVENT_GAP_HOURS = 8
        push({"type": "log", "text": "Reading photo dates for event grouping…", "cat": "info"})
        dated = []
        for fp in files:
            dt = get_photo_date(str(fp))
            dated.append((dt, fp))
        dated.sort(key=lambda x: x[0])

        # Group consecutive files with gap < 8 hours into the same event
        events = []
        if dated:
            current_event = [dated[0]]
            for dt, fp in dated[1:]:
                gap_hours = (dt - current_event[-1][0]).total_seconds() / 3600
                if gap_hours > EVENT_GAP_HOURS:
                    events.append(current_event)
                    current_event = [(dt, fp)]
                else:
                    current_event.append((dt, fp))
            events.append(current_event)

        ec = len(events)
        push({"type": "log", "text": f"Found {ec} event{'s' if ec != 1 else ''} across {total} files.", "cat": "info"})

        processed = 0
        stopped   = False
        for event_files in events:
            start_dt = event_files[0][0]
            end_dt   = event_files[-1][0]
            if start_dt.date() == end_dt.date():
                folder_name = f"Event_{start_dt.strftime('%Y-%m-%d')}"
            else:
                folder_name = (f"Event_{start_dt.strftime('%Y-%m-%d')}"
                               f"_to_{end_dt.strftime('%Y-%m-%d')}")
            event_dir = out / folder_name
            event_dir.mkdir(parents=True, exist_ok=True)
            folders[folder_name] = str(event_dir)

            for dt, fp in event_files:
                if _stop_event.is_set():
                    push({"type": "log", "text": "Stopped by user.", "cat": "info"})
                    stopped = True
                    break
                processed += 1
                name = fp.name
                dest = event_dir / name
                if dest.exists():
                    stem, suffix = fp.stem, fp.suffix
                    counter = 1
                    while dest.exists():
                        dest = event_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(str(fp), dest)
                counts[folder_name] = counts.get(folder_name, 0) + 1
                push({"type": "progress", "current": processed, "total": total,
                      "file": name, "cat": "date", "conf": 0,
                      "label": folder_name, "thumb": None})
            if stopped:
                break

        _results = {"counts": counts, "thumbs": {}, "folders": folders, "mode": mode,
                    "uncertain_count": 0, "dup_count": 0}
        push({"type": "done", "results": _results})
        _status = "done"
        return

    for i, fp in enumerate(files, 1):
        if _stop_event.is_set():
            push({"type": "log", "text": "Stopped by user.", "cat": "info"})
            break

        name = fp.name

        # ── COLLECT mode: just copy everything ───────────────────────────────
        if mode == "collect":
            dest = out / "Collected"
            dest.mkdir(parents=True, exist_ok=True)
            # Avoid overwriting files with the same name from different subfolders
            dest_file = dest / name
            if dest_file.exists():
                stem, suffix = fp.stem, fp.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(str(fp), dest_file)
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

        # ── SIMILAR mode: CLIP image-to-image similarity ─────────────────
        if mode == "similar":
            if _ref_embedding is None:
                push({"type": "log", "text": "No reference image loaded — stopping.", "cat": "info"})
                break
            frames = load_frames(str(fp))
            if not frames:
                counts["skip"] = counts.get("skip", 0) + 1
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "skip", "conf": 0, "thumb": None})
                continue
            sim_threshold = max(0.60, min(0.90, 0.70 + (confidence - 0.25) * 0.5))
            found, sim = run_clip_similarity(frames, _ref_embedding, sim_threshold)
            if found:
                ref_dir = out / f"Similar_to_{_ref_name}"
                ref_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(fp), ref_dir / name)
                folders["similar"] = str(ref_dir)
                counts["similar"]  = counts.get("similar", 0) + 1
                thumbs_dir.mkdir(parents=True, exist_ok=True)
                thumb_name = fp.stem + ".jpg"
                small = cv2.resize(frames[0],
                    (320, int(frames[0].shape[0] * 320 / max(frames[0].shape[1], 1))))
                cv2.imwrite(str(thumbs_dir / thumb_name), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "similar", "conf": sim, "thumb": thumb_name})
            else:
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "other", "conf": 0, "thumb": None})
            continue

        # ── DEDUPE mode ───────────────────────────────────────────────────────
        if mode == "dedupe":
            file_ext = fp.suffix.upper()
            is_video_file = file_ext in {e.upper() for e in VIDEO_EXTS}

            if is_video_file:
                # Videos: exact-copy detection via file size + partial MD5.
                # pHash is wrong for trail cameras — fixed background means any two
                # clips of the same duration look perceptually identical even when
                # they're completely different recordings.
                key = compute_quick_hash(str(fp))
                if key is None:
                    counts["skip"] = counts.get("skip", 0) + 1
                    push({"type": "progress", "current": i, "total": total,
                          "file": name, "cat": "skip", "conf": 0, "thumb": None})
                    continue
                is_dup = key in seen_video_hashes
                if not is_dup:
                    seen_video_hashes.add(key)
            else:
                # Images: perceptual hash — catches same photo re-saved at
                # different JPEG quality, slightly cropped, renamed, etc.
                frames = load_frames(str(fp))
                if not frames:
                    counts["skip"] = counts.get("skip", 0) + 1
                    push({"type": "progress", "current": i, "total": total,
                          "file": name, "cat": "skip", "conf": 0, "thumb": None})
                    continue
                hash_list = compute_phash_list(frames)
                if not hash_list:
                    counts["skip"] = counts.get("skip", 0) + 1
                    push({"type": "progress", "current": i, "total": total,
                          "file": name, "cat": "skip", "conf": 0, "thumb": None})
                    continue
                is_dup = is_video_dup(hash_list, seen_hash_objs)
                if not is_dup:
                    seen_hash_objs.append(hash_list)
            if is_dup:
                dup_dir = out / "Duplicates"
                dup_dir.mkdir(parents=True, exist_ok=True)
                dest = dup_dir / name
                if dest.exists():
                    stem, suffix = fp.stem, fp.suffix
                    counter = 1
                    while dest.exists():
                        dest = dup_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(str(fp), dest)
                counts["duplicate"] = counts.get("duplicate", 0) + 1
                folders["duplicate"] = str(dup_dir)
                _dup_files.append({"file": name, "folder": str(dup_dir),
                                   "thumb": fp.stem + ".jpg"})
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "collected", "conf": 0,
                      "thumb": None, "label": "Duplicates"})
            else:
                unique_dir = out / "Unique"
                unique_dir.mkdir(parents=True, exist_ok=True)
                dest = unique_dir / name
                if dest.exists():
                    stem, suffix = fp.stem, fp.suffix
                    counter = 1
                    while dest.exists():
                        dest = unique_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(str(fp), dest)
                counts["unique"] = counts.get("unique", 0) + 1
                folders["unique"] = str(unique_dir)
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "collected", "conf": 0,
                      "thumb": None, "label": "Unique"})
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
            if copy_unsorted:
                unsorted_dir = out / "Unsorted"
                unsorted_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(fp), unsorted_dir / name)
                folders["unsorted"] = str(unsorted_dir)
                counts["unsorted"] = counts.get("unsorted", 0) + 1
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "unsorted", "conf": 0, "thumb": None})
            else:
                counts["skipped"] = counts.get("skipped", 0) + 1
                push({"type": "progress", "current": i, "total": total,
                      "file": name, "cat": "other", "conf": 0, "thumb": None})
            check_faces(fp, name, frames, out, i, total, face_profiles,
                        counts, folders, None)
            continue

        # Multi-copy: copy to every matched category, or just the first
        to_copy = all_matches if multi_copy else [all_matches[0]]
        thumb_saved = False
        # Collect all matched labels for XMP (written once per destination copy)
        all_labels = [next((c["label"] for c in CATEGORIES if c["key"] == k), k.capitalize())
                      for k, _ in to_copy]

        # Date sub-folder path when date_in_cat is enabled
        date_subpath = ""
        if date_in_cat:
            try:
                _dt = get_photo_date(str(fp))
                date_subpath = _dt.strftime("%Y/%B")
            except Exception:
                date_subpath = ""

        for cat_key, conf in to_copy:
            cat_dir = (out / cat_key.capitalize() / date_subpath
                       if date_subpath else out / cat_key.capitalize())
            cat_dir.mkdir(parents=True, exist_ok=True)
            # Always store the top-level category folder so "Open folder" works
            folders[cat_key] = str(out / cat_key.capitalize())
            dest_file = cat_dir / name
            if dest_file.exists():
                stem, suffix = fp.stem, fp.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = cat_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(str(fp), dest_file)
            if write_xmp:
                write_xmp_sidecar(dest_file, all_labels, conf)
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

            # Track duplicate files for the duplicate review screen
            if cat_key == "duplicate" and mode == "sort":
                _dup_files.append({
                    "file":   name,
                    "folder": str(cat_dir),
                    "thumb":  fp.stem + ".jpg",
                })

            # Track uncertain placements for the review screen
            if 0 < conf < UNCERTAIN_THRESHOLD and mode == "sort":
                cat_info = next((c for c in CATEGORIES if c["key"] == cat_key), {})
                _uncertain.append({
                    "file":      name,
                    "folder":    str(cat_dir),
                    "cat":       cat_key,
                    "cat_label": cat_info.get("label", cat_key.capitalize()),
                    "cat_emoji": cat_info.get("emoji", "📁"),
                    "conf":      round(conf, 2),
                    "thumb":     thumb_name,
                })

        check_faces(fp, name, frames, out, i, total, face_profiles,
                    counts, folders, thumb_name)

    _results = {"counts": counts, "thumbs": thumb_map, "folders": folders, "mode": mode,
                "uncertain_count": len(_uncertain), "dup_count": len(_dup_files)}
    push({"type": "done", "results": _results})
    _status = "done"


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    default_output = r"C:\MediaSorted"
    return render_template("index.html", categories=CATEGORIES, default_output=default_output)


@app.route("/help")
def help_page():
    return render_template("help.html")


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


def _run_sort_safe(*args, **kwargs):
    """
    Crash-guard wrapper for run_sort.
    If run_sort throws an unhandled exception the SSE generator would loop
    forever because _status never reaches 'done'.  This wrapper catches any
    exception, logs it to the stream, and forces _status → 'done' so the
    SSE generator always terminates.
    """
    global _status
    try:
        run_sort(*args, **kwargs)
    except Exception as exc:
        push({"type": "log",
              "text": f"[ERROR] Scan crashed unexpectedly: {exc}",
              "cat": "error"})
        push({"type": "done", "results": _results})
        _status = "done"


@app.route("/api/start", methods=["POST"])
def start():
    global _sort_thread, _stop_event, _log, _results, _status, _scan_id
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
    write_xmp    = bool(data.get("writeXmp", False))
    date_in_cat   = bool(data.get("dateInCat", False))
    copy_unsorted = bool(data.get("copyUnsorted", False))
    _stop_event  = threading.Event()
    _log         = []
    _results     = {}
    _status      = "running"
    _scan_id    += 1          # invalidate any SSE generators from previous scans
    _sort_thread = threading.Thread(
        target=_run_sort_safe,
        args=(source, output, confidence, enabled_keys, file_types,
              mode, multi_copy, filter_all, write_xmp, date_in_cat, copy_unsorted),
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
    # Capture the scan generation at connection time.
    # If a new scan starts (_scan_id changes) this generator exits immediately,
    # freeing the Flask thread and preventing thread-pool exhaustion on re-runs.
    my_id = _scan_id

    def generate():
        sent = 0
        try:
            while True:
                # Stale stream — a new scan has started; stop immediately.
                if _scan_id != my_id:
                    break
                with _log_lock:
                    chunk = _log[sent:]
                for msg in chunk:
                    yield f"data: {json.dumps(msg)}\n\n"
                    sent += 1
                if _status in ("done", "idle") and sent >= len(_log):
                    break
                time.sleep(0.15)
        except GeneratorExit:
            pass   # browser closed the connection — exit cleanly

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    folder = (request.json or {}).get("folder", "")
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    # ShellExecuteW opens the folder directly in Explorer and brings it to the foreground.
    # More reliable than subprocess start "" on Windows 11 because it doesn't go through
    # a cmd.exe chain that can lose the foreground-window grant before Explorer spawns.
    try:
        import ctypes
        SW_SHOW = 5
        ctypes.windll.shell32.ShellExecuteW(None, "explore", str(p), None, None, SW_SHOW)
    except Exception:
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


@app.route("/api/duplicates")
def get_duplicates():
    return jsonify({"items": _dup_files,
                    "trash_folder": str(Path((_results.get("folders") or {}).get(
                        "duplicate", str(DOCS / "MediaSorted" / "Duplicate")
                    )).parent / "Trash")})


@app.route("/api/trash-file", methods=["POST"])
def trash_file():
    data       = request.json or {}
    src_folder = data.get("from_folder", "")
    filename   = data.get("filename",    "")
    trash_dir  = data.get("trash_folder", "")
    if not all([src_folder, filename, trash_dir]):
        return jsonify({"error": "Missing parameters"}), 400
    src = Path(src_folder) / filename
    if not src.exists():
        return jsonify({"error": "File not found"}), 404
    try:
        Path(trash_dir).mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(Path(trash_dir) / filename))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/thresholds", methods=["GET", "POST"])
def cat_thresholds():
    global _cat_thresholds
    if request.method == "POST":
        data = request.json or {}
        key  = data.get("key", "")
        val  = data.get("value")        # float 0-1, or None to reset to global
        if not key:
            return jsonify({"error": "key required"}), 400
        if val is None:
            _cat_thresholds.pop(key, None)
        else:
            _cat_thresholds[key] = round(float(val), 3)
        CAT_THRESHOLDS_FILE.write_text(
            json.dumps(_cat_thresholds, indent=2), encoding="utf-8"
        )
        return jsonify({"ok": True, "thresholds": _cat_thresholds})
    return jsonify(_cat_thresholds)


@app.route("/api/uncertain")
def get_uncertain():
    cat_map = {c["key"]: {"label": c["label"], "emoji": c["emoji"]} for c in CATEGORIES}
    available = []
    for k, v in (_results.get("folders") or {}).items():
        if k.startswith("face_") or k in ("unsorted", "skip", "other"):
            continue
        info = cat_map.get(k, {"label": k.capitalize(), "emoji": "📁"})
        available.append({"key": k, "folder": v,
                          "label": info["label"], "emoji": info["emoji"]})
    return jsonify({"items": _uncertain, "available": available})


@app.route("/api/move-file", methods=["POST"])
def move_file():
    data       = request.json or {}
    src_folder = data.get("from_folder", "")
    dst_folder = data.get("to_folder",   "")
    filename   = data.get("filename",    "")
    if not all([src_folder, dst_folder, filename]):
        return jsonify({"error": "Missing parameters"}), 400
    src = Path(src_folder) / filename
    if not src.exists():
        return jsonify({"error": "File not found — it may have already been moved"}), 404
    try:
        dst_dir = Path(dst_folder)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / filename))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/diskspace")
def diskspace():
    """Return free and total bytes for the drive containing the given path."""
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400
    try:
        p = Path(path)
        # Use the drive root if the folder doesn't exist yet
        check = p
        while not check.exists() and check != check.parent:
            check = check.parent
        stat = shutil.disk_usage(str(check))
        return jsonify({"free": stat.free, "total": stat.total, "used": stat.used})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse")
def browse():
    """Return child directories of the given path for the folder picker UI."""
    raw  = request.args.get("path", "").strip()
    # Default: list drive letters on Windows
    if not raw:
        import string
        drives = [f"{d}:\\" for d in string.ascii_uppercase
                  if Path(f"{d}:\\").exists()]
        return jsonify({"path": "", "dirs": drives, "is_root": True})
    p = Path(raw)
    if not p.exists() or not p.is_dir():
        return jsonify({"error": "Path not found"}), 404
    try:
        dirs = sorted(
            [str(child) for child in p.iterdir()
             if child.is_dir() and not child.name.startswith(".")],
            key=lambda s: s.lower()
        )
    except PermissionError:
        dirs = []
    return jsonify({"path": str(p), "dirs": dirs, "is_root": False})


@app.route("/api/reference", methods=["POST"])
def set_reference():
    """Receive a reference image, encode it with CLIP, store for visual similarity search."""
    global _ref_embedding, _ref_name
    data    = request.json or {}
    img_b64 = data.get("image", "")
    name    = data.get("name", "reference")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400
    try:
        raw   = base64.b64decode(img_b64.split(",")[-1])
        arr   = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        load_clip()   # ensure model loaded (image encoder only, no text features)
        feat = encode_image_clip(frame)
        if feat is None:
            return jsonify({"error": "CLIP encoding failed — model may still be loading"}), 500
        _ref_embedding = feat
        _ref_name      = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(name).stem)[:30]
        return jsonify({"name": _ref_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reference/clear", methods=["POST"])
def clear_reference():
    """Clear the stored reference image so a new one can be uploaded."""
    global _ref_embedding, _ref_name
    _ref_embedding = None
    _ref_name      = ""
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
