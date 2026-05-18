# Media Sorter — User Manual

AI-powered photo and video organiser. Sorts trail camera footage, phone photos,
or any folder of media into labelled subfolders automatically.

---

## What it does

Scans a folder of photos and videos, uses AI to identify what is in each file,
and copies them into organised subfolders — Animals, People, Night, Nature,
Buildings, Damaged, and more. You choose which categories you want.

---

## Files on your computer

```
C:\Users\...\Documents\
  TrailCameraApp\           ← the app lives here
    app.py                  ← main program (do not delete)
    templates\index.html    ← the web interface
    requirements.txt        ← list of needed packages
    install.bat             ← first-time setup (run once)
    Launch Media Sorter.bat ← how you open the app every time

  trail_camera_sorter.py    ← original command-line backup script
  md_v5a.0.0.pt             ← MegaDetector AI model (~290 MB)

C:\Users\..\.cache\         ← CLIP AI model lives here (auto-managed)
C:\Users\..\AppData\Roaming\Python\   ← Python packages (auto-managed)
```

Your sorted output goes wherever you set the Output folder in the app.
Default: `C:\Users\...\Documents\MediaSorted\`

---

## Setup — do this once per computer

**Step 1 — Install Python**

Download and install Python 3.10 or newer from https://www.python.org/downloads/

During install, check the box that says **"Add Python to PATH"** — this is important.

**Step 2 — Install packages**

Double-click `install.bat` inside the TrailCameraApp folder.
A window will open and install everything automatically. Takes 5–15 minutes.
You only ever do this once.

**Step 3 — Done**

From now on, just double-click `Launch Media Sorter.bat` to open the app.
Your browser will open automatically at http://localhost:5000

---

## Using the app

**1. Set the source folder**

This is where your photos/videos are. Click the 🔍 button to auto-detect an
SD card or phone. If that doesn't work, type the path manually.

Common paths:
- SD card in card reader: `D:\DCIM\MOVIE` or `E:\DCIM\100MEDIA`
- Phone plugged in via USB: browse My Computer for the phone drive
- Old photo folder: `C:\Users\...\Pictures\Vacation2022`

**2. Set the output folder**

Where sorted subfolders will be created. Each category gets its own subfolder.
You can use a different output folder for each project.

**3. Choose file types**

Toggle Videos (MP4, MOV, AVI…) and/or Photos (JPG, PNG, BMP…) on or off.

**4. Choose categories**

Click each category chip to turn it on (green dot) or off. Only selected
categories will be sorted. Use "All on / All off" for quick switching.

| Category | What it finds |
|---|---|
| 🌙 Night / IR | Infrared and night-vision footage — identified by greyscale colour |
| ⚠️ Damaged | Blurry, out-of-focus, or corrupted images |
| 🐾 Animals | Any animal — deer, bear, raccoon, birds, etc. |
| 🚶 People | People in frame |
| 🚗 Vehicles | Cars, trucks, ATVs |
| 🌿 Nature | Forest, fields, mountains, natural landscapes |
| 🏛️ Buildings | Houses, structures, urban scenes |
| 🍽️ Food | Meals, food close-ups |
| 🏠 Indoor | Interior scenes |
| 🌊 Water / Beach | Ocean, lake, river scenes |

**5. Set confidence**

The slider controls how certain the AI must be before placing a file in a category.
- Drag left: catches more (may include some false matches)
- Drag right: only very confident matches (may miss some)
- 15% is a good starting point for trail cameras

**6. Press Start**

The log shows each file as it is processed, colour-coded by category.
Results appear as thumbnail previews in real time.
Press the 📂 Open Folder button on any result card to open that folder.

**7. Stop at any time**

Press Stop to halt mid-sort. Files already copied remain in place.
You can restart and it will process all files again (safe — just overwrites copies).

---

## First run note

The first time you run a sort, the app downloads two AI models:
- **MegaDetector** (~290 MB) — detects animals, people, vehicles
- **CLIP** (~350 MB) — detects nature, buildings, food, and other scene types

This happens once and is stored on your computer. All future sorts are instant to start.
An internet connection is required for the first run only.

---

## Troubleshooting

**Browser doesn't open automatically**
Open your browser and go to http://localhost:5000 manually.

**"Detection failed — enter path manually"**
The auto-detect button couldn't find your SD card or phone.
Just type the path in the Source box (e.g. `D:\DCIM\MOVIE`).

**App won't start / error in the launch window**
Run `install.bat` again to make sure all packages are installed.
If still failing, try right-clicking `install.bat` and choosing "Run as administrator".

**Very slow processing**
This computer uses the CPU (not a graphics card) for AI processing.
Expect roughly 3–8 seconds per file. 500 files takes 30–60 minutes.
A computer with an NVIDIA GPU would be 10–20x faster.

**A category I expected isn't being detected**
Try lowering the confidence slider toward "More finds".
Some animal types (e.g. deer) are detected with medium confidence — 10–12% may help.

---

## Giving this to someone else

Share the entire `TrailCameraApp` folder plus the `md_v5a.0.0.pt` model file.
The recipient follows Setup steps 1 and 2 above (Python + install.bat).
They do not need you, a terminal, or any technical knowledge after that.

---

## Privacy

Everything runs locally on your computer. No photos or videos are ever uploaded
to the internet. The only internet use is the one-time model download.
