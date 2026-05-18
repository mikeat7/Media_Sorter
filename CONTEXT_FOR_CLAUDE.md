# Context Document — Media Sorter Project
*Read this first if continuing development of this project in a new conversation.*

---

## Who the user is

Michael (mikeat7). Non-technical but very visionary. Comfortable asking questions,
quick to grasp concepts, focused on practical outcomes. Prefers plain explanations
over jargon. Has a trail camera hobby and a broader idea for a general-purpose
AI media sorter. Windows 11 Home, Python 3.13.5 installed.

---

## What was built

A local web application that scans a folder of photos/videos, runs AI detection,
and copies files into labelled subfolders. Started as a trail camera tool, evolved
into a general-purpose media organiser.

The app runs as a Flask web server on localhost:5000. The user opens their browser
to use it. No cloud, no accounts, fully private.

---

## File map

```
C:\Users\Dito\Documents\
  TrailCameraApp\
    app.py                    ← Flask backend (main program)
    templates\index.html      ← Web UI
    requirements.txt          ← pip dependencies
    install.bat               ← one-time setup for new users
    Launch Media Sorter.bat   ← daily launcher
    README.md                 ← user manual
    CONTEXT_FOR_CLAUDE.md     ← this file
    custom_categories.json    ← user-created CLIP categories (auto-generated)
    face_profiles\            ← face embeddings per person (*.npy files)
    md_v5a.0.0.pt             ← MegaDetector model (~290 MB)

  trail_camera_sorter.py      ← original CLI script (backup, keep it)

Output (configurable, default C:\MediaSorted\):
  Animals\  People\  Night\  Damaged\  Nature\  Buildings\ etc.
  People_Mike\  People_Sarah\  ← face recognition folders
  Collected\                    ← Scan For Photos output
  Unsorted\                     ← files that matched no category
  thumbs\                       ← JPEG thumbnails for UI grid
```

---

## Operation modes

| Mode | Label in UI | What it does |
|---|---|---|
| collect | Scan For Photos | Recursive scan → all files into Collected/ — no AI |
| sort | Sort By Category | AI → category subfolders, multi-copy optional |
| date | Sort by Date | EXIF date → Year/Month subfolders |
| location | Sort by Location | GPS → City_CC subfolders |
| filter | Category Matcher | AND logic — file must match ALL selected categories |

Default mode on load: **Scan For Photos** (collect).
All modes scan recursively through all subfolders of the source folder.
Output folder is excluded from scanning to prevent re-processing sorted files.
After any scan completes, the source field auto-updates to the output folder
so the user can immediately chain a second scan without retyping.

---

## Detection stack

### 1. Infrared / Night detection (fastest, no model)
Channel diff: mean(|R-G| + |R-B| + |G-B|) / 3 < 8 → infrared frame.
Majority vote across sampled frames. Always first priority.

### 2. MegaDetector v5a (Microsoft, ~290 MB)
YOLOv5 loaded via torch.hub (trust_repo=True). Classes: 0=animal, 1=person, 2=vehicle.
Default confidence: 25% (slider). At 15% catches more; at 30%+ stricter.

### 3. OpenCLIP ViT-B/32 (~350 MB, downloads on first use)
Text-image cosine similarity. Threshold scales with confidence slider:
clip_threshold = max(0.15, 0.21 + (slider - 0.15)).
At 25% slider → CLIP threshold = 0.31 (reduces false positives).
Adding a new category = writing a text prompt, zero retraining.

### 4. Blur/Damage detection (no model)
Laplacian variance < 80 on greyscale frame = blurry. Majority vote.
Images only (not videos).

### 5. EXIF tag detection (no model)
Screenshots: software field contains "screenshot" or resolution matches phone screen sizes.
Selfies: LensModel contains "front".
Panoramas: SceneCaptureType == 3.

### 6. Perceptual hash — Duplicates (imagehash library)
pHash difference ≤ 8 between frames = duplicate.

### 7. Facial recognition (facenet-pytorch, ~110 MB one-time download)
MTCNN face detection + InceptionResnetV1 (VGGFace2 pretrained).
User adds face profiles via the UI (name + 1-5 reference photos).
L2 distance threshold: 0.85. Files with matching faces copy to People_Name/.
Face checking is independent of category matching — runs on every file in sort mode.
Installed with: pip install facenet-pytorch --no-deps (numpy 2.x compatibility).
Profiles stored as: face_profiles\SafeName.npy (numpy array of embeddings).

---

## Video sampling strategy

Extract frames at 2, 4, 6, 8, 10 seconds (not evenly distributed).
Animals appear in first seconds of trail camera footage.
Face recognition checks first 2 frames only for speed.
Frame extraction uses actual FPS from video metadata.
Image files: load single frame with cv2.imread.

---

## Category priority order (Sort mode)

1. Night / IR
2. Damaged / Blurry (images only)
3. Screenshots, Selfies, Panoramas (EXIF, images only)
4. Duplicates (perceptual hash)
5. Animals, People, Vehicles (MegaDetector)
6. CLIP categories in list order (Pets, Nature, Sunset, Water, Buildings,
   Flowers, Food, Documents, Snow, Indoor, Events)
7. Custom user-created CLIP categories
8. Unsorted/ (copied here if nothing matched — no file left behind)

Face recognition runs independently AFTER the above — always checked in sort mode.
Multi-copy: when ON, file goes to ALL matched category folders, not just first.

---

## UI details

- Dark forest theme, category toggles, file type pills (Videos / Photos)
- Default: all categories OFF, multi-copy ON, confidence 25%
- Custom category form at bottom of category grid (name + description → CLIP prompt)
- Face Profiles card: name + file upload → stores embeddings, chips show saved profiles
- After scan completes: source field auto-updates to output folder for chaining
- Collect mode: source auto-updates to output\Collected\ specifically
- Folder open button uses shell start command for foreground focus
- Phone / tablet access: http://[local-IP]:5000 on same WiFi

---

## Key technical decisions

**torch.hub for MegaDetector:** MegaDetector is YOLOv5 format; ultralytics YOLO class
only loads v8+. trust_repo=True avoids interactive prompt.

**CLIP threshold scales with slider:** At 15% slider → 0.21 similarity.
At 25% → 0.31 (reduces false positives like karaoke mic in Food folder).
Formula: clip_threshold = max(0.15, 0.21 + (confidence - 0.15)).

**facenet-pytorch --no-deps:** Requires numpy<2.0 but works fine with numpy 2.x.
Install with --no-deps to skip the constraint check.

**Recursive rglob:** All modes now scan recursively through subfolders.
Output folder excluded from scan to prevent processing already-sorted files.

**No deletion policy:** Nothing is ever deleted. Unmatched files → Unsorted/.
Blurry/corrupted → Damaged/. User reviews and deletes manually.

**host='0.0.0.0':** Flask binds to all interfaces for phone/tablet access on WiFi.

**Windows case-insensitive FS fix (historical):** Original CLI used dedup dict
{p.name.upper(): p} to avoid counting same file twice via glob("*.MP4") and
glob("*.mp4"). No longer needed with rglob("*") + suffix check.

---

## Current state (as of May 2026)

### Working and tested:
- Full web UI with mode selector, category toggles, live log, thumbnails
- MegaDetector: animals, people, vehicles — confirmed working (deer 89%, person 93%)
- IR/Night detection — confirmed working
- 5-frame sampling at 2,4,6,8,10 seconds
- Flask server, SSE streaming, thumbnail serving
- Collect/Sort/Date/Location/Filter modes all implemented
- Recursive folder scanning
- Auto-populate source from output after scan completes
- Custom CLIP categories (persisted to custom_categories.json)
- Face profiles UI (add/delete) — code complete, not yet tested end-to-end
- Unsorted/ folder for unmatched files
- Multi-copy toggle (default ON)
- Confidence slider controls both MegaDetector and CLIP threshold
- Open folder brings window to foreground via shell start

### Not yet tested:
- Face recognition end-to-end (facenet-pytorch installed and imported OK)
- CLIP categories on diverse real-world photos (partial test — false positives at 15%)
- Blur detection end-to-end
- install.bat on a fresh machine
- By Date / By Location modes on photos with GPS/EXIF

---

## Future ideas (in rough priority order)

### 1. "Find Photos Like This" (visual similarity search)
Upload any reference photo → CLIP encodes it → finds all photos with similar
visual embedding. Works for specific flowers, food dishes, specific dogs, scenes.
Implementation: new route /api/find-similar, encode reference with CLIP,
compare against each file's image embedding, threshold ~0.85 cosine similarity.
No new models needed — uses existing CLIP infrastructure.

### 2. Per-category confidence thresholds
Some categories need looser settings (Animals on trail cam = 15%),
others need stricter (Food, Buildings = 30%+).
UI: expandable per-category slider, or presets ("Trail Camera" vs "General Photos").

### 3. Tune CLIP prompts based on real-world results
Food at 15%: karaoke mic and fashion photo triggered false matches.
Tighter prompts ("a photo of plated food or a meal being eaten") may help.
Consider adding negative prompts if CLIP API supports them.

### 4. PyInstaller single .exe for distribution
~2-3 GB bundle but plug-and-play for non-technical users.
Challenge: PyTorch + CLIP + facenet models are large.
Consider: installer that downloads models separately post-install.

### 5. HEIC support (iPhone photos)
Needs pillow-heif package. Install: pip install pillow-heif.
Add to IMAGE_EXTS and handle in load_frames().

### 6. Subfolder results in UI
When scanning a folder with subfolders, show which subfolder each file came from
in the log. Useful for understanding where photos originated.

### 7. Source folder tree browser
Instead of typing a path, let the user browse folders in the UI.
Flask route: /api/browse?path=C:\ returns directory listing.
Simple tree UI with clickable folders.

### 8. Duplicate detection improvements
Current: perceptual hash of first frame only.
Better: compare across all sampled frames, flag near-duplicates.
Consider: separate "Duplicates" review UI showing pairs side by side.

### 9. Android / iOS app
React Native or Flutter wrapper around a local Flask server.
Or: hosted version with cloud processing (different privacy model).
Gap in market: no offline, folder-based, configurable sorter on mobile.

### 10. Facial recognition improvements
- Allow multiple profiles to match the same photo (multi-copy for faces)
- Confidence display per match
- "Unknown faces" folder for detected faces that don't match any profile
- Option to name unknown faces from the UI

---

## Packages installed (on Michael's machine)

All installed to user directory (no admin needed):
flask, opencv-python, torch (CPU), ultralytics, open-clip-torch,
pillow, psutil, numpy (2.4.3), pandas, seaborn, tqdm, requests,
ImageHash, reverse_geocoder, facenet-pytorch (2.6.0, --no-deps)

YOLOv5 hub code: C:\Users\Dito\.cache\torch\hub\ultralytics_yolov5_master\
CLIP model cache: C:\Users\Dito\.cache\
facenet models: downloaded to torch cache on first face profile add

Python: C:\Python313\ (system install, added to PATH)

---

## How to resume

Tell Claude: "Read CONTEXT_FOR_CLAUDE.md in C:\Users\Dito\Documents\TrailCameraApp\
and continue the Media Sorter project."

Most useful first action in a new session: check what the user last tested,
then verify that feature works before adding anything new.
