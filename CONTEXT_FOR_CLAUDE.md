# Context Document — Media Sorter Project
*Read this first if continuing development of this project in a new conversation.*

---

## Who the user is

Michael (mikeat7). Non-technical but very visionary. Comfortable asking questions,
quick to grasp concepts, focused on practical outcomes. Prefers plain explanations
over jargon. Has a trail camera hobby and a broader idea for a general-purpose
AI media sorter. Windows 11 Home, Python 3.13.5 installed.

GitHub: https://github.com/mikeat7/Media_Sorter

---

## What was built

A local web application (Flask) that scans folders of photos/videos recursively,
runs AI detection, and copies files into labelled subfolders. Fully offline,
no accounts, no cloud. Started as a trail camera tool; now a general-purpose
media organiser positioned against Excire Foto ($199), DigiKam, ACDSee.

Key competitive advantage: user-definable CLIP categories (type a sentence,
create a new category instantly — no retraining). No competitor offers this.

Second major advantage: Visual Similarity Search — upload any reference photo,
find all visually similar photos in any folder. Any subject, zero configuration.

---

## File map

```
C:\Users\Dito\Documents\TrailCameraApp\
  app.py                      ← Flask backend (main program)
  templates\index.html        ← Web UI
  requirements.txt            ← pip dependencies
  install.bat                 ← one-time setup
  Launch Media Sorter.bat     ← daily launcher
  MediaSorter_Setup.iss       ← Inno Setup script (compile → distributable .exe)
  README.md                   ← user manual
  CONTEXT_FOR_CLAUDE.md       ← this file
  custom_categories.json      ← user-created CLIP categories (auto-generated)
  category_thresholds.json    ← per-category CLIP threshold overrides (auto-generated)
  face_profiles\              ← face embeddings per person (*.npy files)
  md_v5a.0.0.pt               ← MegaDetector model (~290 MB)

GitHub repo (distributable files only — models excluded):
  https://github.com/mikeat7/Media_Sorter

Output (default C:\MediaSorted\):
  Animals\   People\   Night\   Damaged\   Nature\   etc.
  People_Mike\              ← face recognition output
  Similar_to_[name]\        ← visual similarity search output
  Collected\                ← Scan For Photos output
  Unsorted\                 ← no-match fallback
  Trash\                    ← files moved from Duplicate Review (never auto-deleted)
  thumbs\                   ← UI thumbnails
```

---

## Operation modes (UI labels and internal keys)

| UI Label | Internal key | What it does |
|---|---|---|
| Scan For Media | collect | Recursive scan → all files into Collected/ — no AI |
| Sort By Category | sort | AI → category subfolders, multi-copy optional |
| Sort by Date | date | EXIF date → Year/Month subfolders |
| Sort by Location | location | GPS → City_CC subfolders |
| Find Photos With | filter | AND logic — must match ALL selected categories |
| Find Similar Photos | similar | Upload reference photo → CLIP image-to-image cosine similarity |
| Find Duplicates | dedupe | pHash scan → Unique/ and Duplicates/ — no AI, single-pass, very fast |
| Sort by Event | event | EXIF date sort → time-gap clustering (>8hr gap = new event) → Event_YYYY-MM-DD/ folders |

Default mode on load: NONE — user must click a mode button to begin.
Start button reads "▶ Select a mode above" and is disabled until a mode is chosen.
All modes scan recursively (rglob). Output folder excluded from scan.
Auto-source-update REMOVED — source field never changes automatically after a scan
(was confusing; replaced with a clickable opt-in hint text shown after scan completes).

**Sort mode extra option: Date in category path**
Checkbox "Include date in folder path" in Sort mode.
When ON: files go to Animals/2024/June/ instead of flat Animals/.
Backend param: dateInCat (bool). In run_sort: date_in_cat=False default.
folders[cat_key] always stores top-level category folder (not date subfolder) for the Open button.

**Workflow note (IMPORTANT):**
Always launch from C:\Users\Dito\Documents\TrailCameraApp\Launch Media Sorter.bat
That is the live development copy. The installed version in AppData is a frozen snapshot
for distributing to others. After code changes, close browser tab, Ctrl+C terminal, re-run bat.

---

## Detection stack

### 1. IR / Night (no model, fastest)
Mean channel diff < 12 across R/G/B → infrared frame. Majority vote. Always priority 1.
(Threshold raised from 8 → 12 to catch trail-cam photos with slight IR LED red cast.
B&W / grayscale photos always match this rule by design — zero channel difference.)

### 2. MegaDetector v5a (~290 MB, one-time download)
torch.hub.load('ultralytics/yolov5', 'custom', trust_repo=True).
Classes: 0=animal, 1=person, 2=vehicle.
Confidence threshold: from slider (default 25%).

### 3. OpenCLIP ViT-B/32 (~350 MB, one-time download)
Text-image cosine similarity. Global threshold scales with slider:
  clip_threshold = max(0.15, 0.21 + (slider_value - 0.15))
At 25% slider → 0.31 threshold (reduces false positives).
Per-category overrides stored in category_thresholds.json (take precedence over global).
User-definable categories: type a description → instant new category, zero retraining.

### 4. Visual Similarity Search (same CLIP model, image-to-image)
User uploads reference photo → CLIP image encoder → normalised feature vector stored in _ref_embedding.
For each file: encode with CLIP image encoder → cosine similarity vs reference.
Threshold: max(0.60, min(0.90, 0.70 + (confidence - 0.25) * 0.5))
  At 25% slider → 0.70 (default). Higher slider = stricter match.
Matches copied to Similar_to_{refname}/ folder.

### 5. Laplacian blur (no model)
Variance < 80 on greyscale → blurry. Images only. Majority vote.

### 6. EXIF tag detection (no model)
Screenshots: software field or screen-resolution match.
Selfies: LensModel contains "front". Panoramas: SceneCaptureType == 3.

### 7. Duplicate detection (split by file type)
**Videos**: exact-copy fingerprint — file size + MD5 of first and last 1 MB (compute_quick_hash).
  Two videos with matching (size, hash) tuple are byte-for-byte identical → true duplicate.
  pHash was abandoned for videos: trail cameras are stationary, so any two clips of the
  same duration produce nearly identical perceptual hashes regardless of content.
**Images**: pHash difference ≤ 8 = duplicate (catches same photo re-saved at different quality).

### 8. Facial recognition (facenet-pytorch, ~110 MB one-time download)
MTCNN + InceptionResnetV1 (VGGFace2). L2 threshold: 0.85.
User adds profiles via UI (name + 1-5 photos). Stored as face_profiles\Name.npy.
Runs independently of category matching on every file in sort mode.
Install: pip install facenet-pytorch --no-deps (numpy 2.x workaround).

### 9. HEIC support
pillow-heif registers a Pillow opener at startup (try/except — optional).
Allows iPhone HEIC photos to be processed by all detectors.

---

## Video sampling

Frames extracted at 2, 4, 6, 8, 10 seconds (animals appear early in trail cam footage).
Face recognition: first 2 frames only for speed.
FPS read from video metadata; images load as single frame.

---

## Category priority order

1. Night / IR
2. Damaged (images only)
3. Screenshot, Selfie, Panorama (EXIF, images only)
4. Duplicate (hash)
5. Animal, Person, Vehicle (MegaDetector)
6. Pets, Nature, Sunset, Water, Buildings, Flowers, Food, Documents, Snow, Indoor, Events (CLIP)
7. Custom user categories (CLIP)
8. Unsorted/ — nothing left behind

Face profiles: independent pass after above, always runs in sort mode.
Multi-copy (default ON): file copied to every matched folder.

---

## UI details

- Dark forest theme (#141814 background)
- Default: categories ALL OFF, multi-copy ON, confidence 25% (auto-bumps to 50% when entering Find Similar Photos, restores on exit)
- Copy non-matching to Unsorted/: checkbox, default OFF (skips unmatched files instead of bulk-copying)
- Include date in folder path: Sort mode checkbox — Animals/2024/June/ instead of flat Animals/
- Disk space indicator: shown next to Output field; pre-start warning if <5GB free
- Reference photo Clear button (✕ Clear): resets CLIP reference in Find Similar Photos
- "⚙ fixed rule" badge on Night/IR, Damaged, EXIF, Duplicates categories (confidence slider has no effect)
- 📖 Help button in header: opens built-in reference manual at /help in new tab
- Mode bar: 8 buttons — no default selected on load (user must choose)
- Category grid: colour-coded toggles, ● coloured = will sort / ● grey = skip
- Custom category form: name + description → new CLIP toggle
- Per-category threshold panel: collapsible ⚙️ section, sliders per CLIP category, saved to JSON
- Face Profiles card: name + file upload → embeddings stored → chip UI
- Find Similar Photos: file upload → CLIP encodes reference → scan finds visually similar
- After sort: "Review Uncertain Placements" button if any file sorted < 40% confidence
- After sort: "Review Duplicates" button if duplicates found — keep/trash grid, files go to Trash/ (never deleted)
- XMP sidecar export: optional checkbox — writes .xmp file alongside each sorted photo
- Folder browser: 📁 button on source and output fields — click-to-navigate tree modal
- After scan: clickable hint text offers to update source to output (opt-in, not automatic)
- Open folder: uses shell start for foreground focus
- Phone access: http://[local-IP]:5000 on same WiFi

---

## Competitive landscape (researched May 2026)

| App | Price | Custom AI cats | Offline | Folder export | Face |
|---|---|---|---|---|---|
| Excire Foto | $199 | ✗ | ✓ | ✗ | ✓ |
| DigiKam | Free | ✗ | ✓ | partial | ✓ |
| ACDSee | $80/yr | ✗ | ✓ | ✗ | ✓ |
| Mylio | $100/yr | ✗ | ✓ | ✗ | ✓ |
| Google Photos | Free | ✗ | ✗ | ✗ | ✓ |
| **Media Sorter** | **Free** | **✓** | **✓** | **✓** | **✓** |

User complaints across all competitors: cloud dependency, no folder output,
no custom categories, AI mis-tagging with no review workflow, two-tool workflow
(organise + edit), false positives with no correction path.

---

## Current state (May 2026)

### Phase 1 — FULLY BUILT, active testing in progress

All features implemented and bug-fixed. Active final testing in progress.

**Modes tested and confirmed working:**
- Find Duplicates (dedupe) — videos exact-copy, images pHash. Full 1926-file library: 1302 unique, 624 duplicates, all correct.
- Scan For Media (collect) — confirmed working
- Sort By Category (sort) — PARTIALLY tested. MegaDetector (animals/people/vehicles) and face recognition confirmed working on 2404-file library. CLIP categories broken at 50% slider (threshold too high — now fixed). Rerun needed at 25% slider to validate CLIP categories.

**Bugs fixed during Sort By Category test:**
- CLIP threshold cap: formula now capped at 0.38 max. Previously at 50% slider CLIP threshold hit 0.56 — effectively disabled. All CLIP categories (Nature, Food, Indoor, etc.) returned 0 matches. Fix: min(0.38, ...) cap in run_sort clip_thresh calculation.
- Source-inside-output: fixed for all modes (see Key technical decisions).
- These fixes are in app.py but NOT yet pushed to GitHub — accumulating fixes for clean release after full test pass.

**Modes not yet tested (continue from here):**
- Sort By Category (sort) — rerun at 25% slider to validate CLIP categories (Nature, Indoor, Buildings, etc.)
- Sort by Date (date) — EXIF → Year/Month folders
- Sort by Location (location) — GPS → City folders
- Find Photos With (filter) — AND logic multi-category
- Find Similar Photos (similar) — CLIP image-to-image
- Sort by Event (event) — time-gap clustering

**Next session starting point:**
1. Delete D:\c1_cleaned_and_sorted (incomplete test run, safe to delete — originals untouched in source)
2. Restart app (Ctrl+C → Launch Media Sorter.bat)
3. Run Sort By Category on D:\sorted_pictures\Unique_pictures → D:\c1_cleaned_and_sorted
   - Slider: 25% (default)
   - All categories ON
   - Face profiles: Mike and Nancy already saved
4. Let run overnight — ~13 hours for 2404 files
5. Audit results: expect Animals, People, Vehicles, Night, Damaged + all CLIP categories populated
6. Then proceed to test remaining modes one by one

**Do NOT push to GitHub until all modes pass testing.** v2.0 was released during testing — accumulate remaining fixes for a clean v2.1 release when Phase 1 is complete.

### Features implemented (test each):
- MegaDetector: animals, people, vehicles
- IR/Night detection
- 5-frame video sampling at 2,4,6,8,10 sec
- Flask server, SSE streaming, thumbnail serving
- All 8 modes implemented (including Find Similar Photos, Find Duplicates, Sort by Event)
- Recursive scanning with output exclusion
- Auto-populate source from output after scan
- Custom CLIP categories (persisted to JSON)
- Per-category CLIP threshold overrides (collapsible panel, persisted to JSON)
- Face profiles UI — add by photo, sort creates People_Name/ folders
- Unsorted/ fallback folder
- Multi-copy toggle (default ON)
- Confidence slider controls MegaDetector + CLIP
- Find Similar Photos — CLIP image-to-image cosine similarity
- Find Duplicates mode — pHash scan → Unique/ and Duplicates/ folders (no AI, no deletion)
- Sort by Event mode — EXIF date sort + 8hr gap clustering → Event_YYYY-MM-DD/ folders
- Date within Category — checkbox in Sort mode → Animals/2024/June/ subfolder structure
- Confidence Review Screen — post-sort grid of uncertain placements (<40%), keep/move
- Duplicate Review Screen — post-sort grid, keep/trash (Trash/ folder, never auto-deleted)
- HEIC support (iPhone photos via pillow-heif)
- XMP sidecar export (optional, writes .xmp alongside each sorted photo)
- Folder browser — 📁 click-to-navigate modal for source and output fields
- Open folder button brings Explorer to foreground
- Phone access: http://[local-IP]:5000
- GitHub repo live: https://github.com/mikeat7/Media_Sorter
- Inno Setup script ready to compile: MediaSorter_Setup.iss

### How to compile the installer (developer step, done once per release):
1. Download Inno Setup 6 free from https://jrsoftware.org/isdl.php
2. Open MediaSorter_Setup.iss in Inno Setup
3. Press F9 (or Build → Compile)
4. Output: dist\MediaSorter_Setup.exe (~5 MB)
5. Upload that .exe to GitHub → Releases → Create new release
End users download and run the .exe — they never touch Inno Setup or Python setup.

---

## IMPLEMENTATION PLAN

---

### PHASE 1 — PC App: Complete & Polished ✅ BUILT

All 10 steps implemented. Needs one complete test pass before declaring done.

| Step | Feature | Status |
|---|---|---|
| 1 | Validate everything end-to-end | ⏳ Test now |
| 2 | Visual Similarity Search | ✅ |
| 3 | Confidence Review Screen | ✅ |
| 4 | HEIC support | ✅ |
| 5 | Per-category confidence thresholds | ✅ |
| 6 | Duplicate/Burst Review UI | ✅ |
| 7 | Source folder browser | ✅ |
| 8 | XMP sidecar export | ✅ |
| 9 | Inno Setup installer script | ✅ |
| 10 | Polish and documentation | After testing |

**Phase 1 complete definition:**
All 8 modes tested and working. All review screens functional. HEIC, XMP, folder browser,
custom categories, face recognition all working. Installer compiles and installs cleanly.

---

### PHASE 2 — Android App (Google Play)
*Goal: a native Android app with full feature parity, publishable on Google Play.*

**Architecture decision (recommended: Flutter + on-device models)**

Option A — WiFi bridge (easiest, limited):
  Phone app connects to PC running Media Sorter. Works on home WiFi only.
  Publish as a "companion app." Fast to build but not standalone.

Option B — Flutter + on-device AI (recommended):
  Flutter app runs natively on Android. Models converted to TFLite/ONNX format
  and bundled with the app. Fully offline, no PC required. Publishable on Play Store.
  Estimated app size: ~800 MB with models (acceptable for modern phones).

**Step 2-1 — Model conversion**
- MegaDetector v5a: export to ONNX → convert to TFLite via ONNX-TF
- OpenCLIP ViT-B/32: export text/image encoders to ONNX separately
- facenet: already has TFLite weights available
- Estimated effort: 2-3 sessions

**Step 2-2 — Flutter project setup**
- Flutter SDK installation and Android Studio setup
- Project structure: lib/screens/, lib/models/, lib/services/
- Android permissions: READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, CAMERA
- Target: Android 10+ (API 29+)
- Estimated effort: 1 session

**Step 2-3 — Core scan and sort features**
- Folder picker → recursive media scan
- ONNX Runtime for Android → run MegaDetector and CLIP inference
- Same category logic as PC version (ported to Dart)
- Output folder creation and file copy
- Progress screen with live results
- Estimated effort: 3-4 sessions

**Step 2-4 — Face profiles on mobile**
- Camera capture or gallery picker for reference photos
- TFLite facenet inference on-device
- Store embeddings in local SQLite database
- Estimated effort: 1-2 sessions

**Step 2-5 — Visual similarity search on mobile**
- Same CLIP image-to-image approach as PC
- Reference photo picker → encode → scan library
- Estimated effort: 1 session

**Step 2-6 — Google Play Store submission**
- Create Play Console account ($25 one-time)
- App signing and release build
- Store listing: screenshots, description, privacy policy (required — app processes photos locally, nothing uploaded)
- Estimated effort: 1 session

**Step 2-7 — iOS version (App Store)**
- Flutter builds for iOS with minimal additional work once Android is done
- Requires Mac and Apple Developer account ($99/yr)
- Estimated effort: 1-2 sessions

---

## Key technical decisions (running log)

**torch.hub for MegaDetector:** YOLOv5 format; ultralytics YOLO class loads v8 only.
trust_repo=True avoids interactive terminal prompt.

**CLIP threshold scales with confidence slider:**
clip_threshold = max(0.15, 0.21 + (confidence - 0.15)).
At 25% → 0.31. Reduces false positives (karaoke mic in Food, etc.).
Per-category overrides in _cat_thresholds dict take precedence over global.

**Visual similarity threshold:**
sim_threshold = max(0.60, min(0.90, 0.70 + (confidence - 0.25) * 0.5))
At 25% slider → 0.70. Image-to-image cosine is higher than text-image.

**Uncertain threshold:** UNCERTAIN_THRESHOLD = 0.40. Files sorted below this appear
in the Review Uncertain screen after sort. Catch-and-review, not catch-and-reject.

**Duplicate Trash:** Files moved from duplicate review go to output\Trash\ folder.
Never auto-deleted — user always has final say.

**Video dedupe — pHash abandoned:** pHash was originally used for all files including videos.
Trail cameras are stationary so any two clips of the same duration produce nearly identical
perceptual hashes of the background → massive false positives (mink.MP4, groundhog.MOV etc.
all incorrectly flagged as duplicates of trail cam clips).
Fix: videos now use compute_quick_hash() → (file_size, md5_of_first+last_1MB) tuple.
Same file copied/renamed = identical bytes = caught. Different recordings = different size = safe.
Images keep pHash (correct for detecting same photo re-saved at different JPEG quality).
Validated with 18-file controlled test: 14 unique correctly → Unique, 4 duplicates correctly → Duplicates.
Full library test (~2000 videos) running at time of handoff.

**XMP sidecar:** Written as plain XML alongside each sorted copy (no library needed).
Contains dc:subject keywords (category labels) and xmp:Rating (scaled from confidence).
File: photo.xmp sits next to photo.jpg in the destination folder.

**facenet-pytorch --no-deps:** Package declares numpy<2.0 requirement but runs
fine on numpy 2.4.3. Install with --no-deps to skip the constraint check.

**Recursive rglob:** All modes scan recursively. Output folder excluded via
str(p).upper().startswith(out_upper) check to prevent re-processing sorted files.
Exception: if source is a subfolder of output (e.g. source=D:\sorted\Unique, output=D:\sorted),
the exclusion is bypassed because rglob is already scoped to source — no re-processing can occur.
Implemented via source_inside_output flag in the file-gathering block of run_sort.
Only exact-same-path (source == output) still correctly returns 0 files.

**No deletion policy:** Nothing auto-deleted. Unmatched → Unsorted/.
Blurry → Damaged/. Trash/ folder is a holding area, not auto-emptied.

**HEIC registration:** pillow_heif.register_heif_opener() called at import time
inside try/except — app starts normally if pillow-heif is not installed.

**host='0.0.0.0':** Flask binds all interfaces for phone/tablet access on WiFi.

**shell=True for open folder:** Routes through Windows start command which
grants foreground window focus. Direct subprocess.Popen(["explorer"]) opens behind browser.

**NetworkError / thread-pool exhaustion fix (May 2026):**
Root cause: Flask dev server uses one thread per request (threaded=True). The SSE stream
at /api/stream holds a long-lived connection for the full duration of a scan. If the browser
did not cleanly close the previous SSE connection (common with Firefox/Chrome EventSource
reconnect behaviour), old SSE generator threads piled up across re-runs, exhausting the
thread pool. New requests to /api/start or /api/stream then timed out → browser reported
"NetworkError when attempting to fetch resource."

Three changes in app.py:
1. _scan_id counter — incremented in /api/start on every new scan. Each SSE generator
   captures my_id = _scan_id at connect time; exits immediately if _scan_id != my_id.
   Stale generators from previous scans free their threads the instant a new scan starts.
2. GeneratorExit handler — try/except GeneratorExit in the generator so browser-side
   disconnect is always handled cleanly.
3. _run_sort_safe wrapper — wraps run_sort() in try/except; on crash pushes
   [ERROR] log message + done event and forces _status = 'done', so the SSE generator
   always terminates even if run_sort throws an unhandled exception.

Result: no restart required between scans. Multiple scans in the same session work reliably.

---

## Packages installed on Michael's machine

flask, opencv-python, torch (CPU), ultralytics, open-clip-torch,
pillow, pillow-heif, psutil, numpy (2.4.3), pandas, seaborn, tqdm, requests,
ImageHash, reverse_geocoder, facenet-pytorch (2.6.0, --no-deps)

Python: C:\Python313\ (system install, on PATH)
YOLOv5 hub cache: C:\Users\Dito\.cache\torch\hub\ultralytics_yolov5_master\
CLIP model cache: C:\Users\Dito\.cache\

---

## How to resume

Tell Claude: "Read CONTEXT_FOR_CLAUDE.md in C:\Users\Dito\Documents\TrailCameraApp\
and continue the Media Sorter project."

Always read this file before starting. Check current state section first —
test what is built before adding anything new.
