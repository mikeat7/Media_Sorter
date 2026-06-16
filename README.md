# Media Sorter — User Manual

A fully-offline AI photo and video organiser. Sorts trail camera footage, phone
photos, or any folder of media into labelled subfolders. Nothing is ever uploaded.

---

## What it does

Scans a folder of photos and videos and organises them into subfolders. It does this
two different ways, and it's worth understanding the difference:

- **Fact-based sorting (rock-solid):** dates, GPS location, duplicates, faces, and
  visual similarity. These read real data or compare pixels — they don't guess, and
  they're reliable.
- **AI scene recognition (smart best-effort):** categories like Nature, Buildings,
  Food, Indoor, and your own custom categories. These are ~90% accurate on clear
  subjects and occasionally miss or misfile ambiguous shots. Treat them as helpful
  **suggestions, not guarantees** — a free, fully-offline AI model has real limits.

You choose which modes and categories you want for each job.

---

## Setup — do this once per computer

**Step 1 — Install Python**

Download and install Python 3.10 or newer from https://www.python.org/downloads/
During install, check the box that says **"Add Python to PATH"** — this is important.

**Step 2 — Install packages**

Double-click `install.bat` inside the app folder. A window opens and installs
everything automatically (5–15 minutes). You only do this once.

**Step 3 — Done**

From now on, just double-click `Launch Media Sorter.bat`. Your browser opens
automatically at http://localhost:5000

---

## The operation modes

Pick a mode at the top of the app. Each does one job:

| Mode | What it does | Reliability |
|---|---|---|
| 📷 Scan For Media | Copies all media into one folder — no AI | Exact |
| 🔀 Sort By Category | Files into category subfolders (see below) | Mixed — see categories |
| 📅 Sort by Date | Reads each photo's date → Year/Month folders | Rock-solid |
| 📍 Sort by Location | Reads GPS data → City folders | Rock-solid (needs GPS in photo) |
| 🔍 Find Photos With | Finds photos matching **all** chosen categories at once | As reliable as the categories used |
| 🖼️ Find Similar Photos | Upload one photo → finds visually similar ones | Rock-solid |
| 🔁 Find Duplicates | Finds duplicate/near-identical files | Rock-solid |
| 🗓️ Sort by Event | Groups photos into events by time gaps | Rock-solid |

---

## Categories (for Sort By Category)

**Fact-based — reliable:**

| Category | What it finds |
|---|---|
| 🐾 Animals | Any animal — deer, bear, raccoon, birds, etc. (MegaDetector) |
| 🚶 People | People in frame (MegaDetector) |
| 🚗 Vehicles | Cars, trucks, ATVs (MegaDetector) |
| 🌙 Night / IR | Greyscale infrared / night-vision footage *(see note)* |
| ⚠️ Damaged | Genuinely blurry or out-of-focus images |
| 👤 Face profiles | People you've added by photo → their own folder |

**AI scene recognition — smart best-effort (~90% on clear subjects):**

| Category | What it finds |
|---|---|
| 🌿 Nature | Forest, fields, mountains, natural landscapes |
| 🌸 Flowers | Flowers and blooming plants |
| ❄️ Snow / Winter | Snow and winter scenes |
| 🌊 Water / Beach | Ocean, lake, river scenes |
| 🏛️ Buildings | Houses, structures, urban scenes |
| 🍽️ Food | Meals, food close-ups |
| 🏠 Indoor | Interior scenes |
| 📄 Documents | Receipts, menus, printed text |
| ➕ Your own | Type a description → instant custom category, no retraining |

> **Custom categories** are the standout feature. Type something like
> *"a deer with antlers"* or *"a child blowing out birthday candles"* and the app
> creates a matching category on the spot. They're best-effort like the other AI
> categories, but no other free tool lets you do this.

---

## The confidence slider — what it really controls

The slider **only affects People, Animals, and Vehicles.** It has **no effect** on
the AI scene categories (Nature, Cats, Flowers, etc.) — those use a fixed setting.
The app hides the slider automatically when it isn't relevant.

- **Drag left ("Catch more"):** flags anything that might be a person/animal —
  catches more, but with more false hits.
- **Drag right ("Only confident"):** only very sure matches — cleaner, but may miss
  distant or partial animals.
- **Leave it at 25%** unless People/Animals are being over- or under-detected. For
  trail cameras, 25% is a good default so you don't miss far-off wildlife.

---

## Trail camera tips

Media Sorter began as a trail-camera tool and still excels at it:

1. **Source:** your SD card folder (e.g. `D:\DCIM\100MEDIA`).
2. **Mode:** Sort By Category.
3. **Enable:** Animals, People, Vehicles, Night/IR (and Damaged to cull empty/blurry
   triggers).
4. **Tick the 🎬 Videos file type** — trail cams record MP4/MOV. The app samples
   frames at 2, 4, 6, 8, and 10 seconds (wildlife usually appears early).
5. **Leave the slider at 25%.**
6. **Afterward, run Find Duplicates** to collapse burst sequences into one copy each.

---

## Good to know

- **Each run starts clean.** When you sort into an output folder, the app clears its
  own previous results there first, so what's on disk always matches what the app
  reports. Your original photos are never touched — the app only ever *copies*.
- **Duplicates are skipped automatically** when sorting, so each unique photo is
  filed once. (Tick the Duplicates category, or use Find Duplicates mode, if you'd
  rather round them up for review instead.)
- **Nothing is ever deleted.** Unmatched files go to `Unsorted/`. Duplicate review
  moves files to a `Trash/` folder you control — never an automatic delete.

---

## First run note

The first sort downloads two AI models (one time, then stored locally):
- **MegaDetector** (~290 MB) — animals, people, vehicles
- **CLIP** (~350 MB) — scene categories

An internet connection is needed for this first download only. Everything after is
fully offline.

---

## Troubleshooting

**Browser doesn't open automatically** — go to http://localhost:5000 manually.

**Auto-detect can't find the SD card** — type the path in the Source box
(e.g. `D:\DCIM\100MEDIA`).

**App won't start** — run `install.bat` again to confirm all packages installed.

**Very slow processing** — this runs on the CPU, not a graphics card. Expect roughly
3–8 seconds per file. A PC with an NVIDIA GPU would be 10–20× faster.

**An animal/person isn't being detected** — drag the slider toward "Catch more".
(This only helps Animals/People/Vehicles — the slider doesn't affect scene categories.)

**An AI scene category misses or misfiles a photo** — that's expected occasionally;
these categories are ~90% best-effort, not perfect. Rewording a custom category's
description can help.

**Windows SmartScreen warning when installing** — because the installer isn't yet
code-signed, Windows may warn. Click **More info → Run anyway**. The app is safe and
fully offline.

---

## Privacy

Everything runs locally on your computer. No photos or videos are ever uploaded.
The only internet use is the one-time model download.
