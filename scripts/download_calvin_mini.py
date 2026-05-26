"""
download_calvin_mini.py
=======================
Downloads the CALVIN task_D_D zip, extracts ONLY the files needed for
visualization (lang_annotations + first N annotated episodes), then
creates a small uploadable zip (~30-80 MB instead of ~1 GB).

Usage:
    python scripts/download_calvin_mini.py

Output:
    calvin_mini.zip  (in current directory)
    → Upload this to Google Drive at: MyDrive/VERA_CALVIN/
    → Then extract it there (or let Colab extract it)

Upload instructions:
    1. Upload calvin_mini.zip to MyDrive/VERA_CALVIN/calvin_mini.zip
    2. In Colab, run:
         import zipfile, shutil
         with zipfile.ZipFile('/content/drive/MyDrive/VERA_CALVIN/calvin_mini.zip') as zf:
             zf.extractall('/content/')
         print('Done')
"""

import urllib.request
import zipfile
import os
import io
import time
import numpy as np
from pathlib import Path

CALVIN_URL   = 'http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip'
OUT_ZIP      = 'calvin_mini.zip'
N_EPISODES   = 4     # annotated episodes to keep (2 for figure + 2 spares)
SPLIT        = 'training'

# ── Step 1: Download full zip ──────────────────────────────────────────────
print('Downloading CALVIN task_D_D.zip …')
print(f'  URL: {CALVIN_URL}')
print('  This is ~1 GB — should take 2–10 min on home internet\n')

CACHE = Path('task_D_D.zip')

if not CACHE.exists():
    start = time.time()
    def _progress(count, block, total):
        pct   = min(100, count * block * 100 // total) if total > 0 else 0
        speed = count * block / (time.time() - start + 1e-9) / 1e6
        print(f'  {pct}%  {count*block/1e6:.0f}/{total/1e6:.0f} MB  {speed:.1f} MB/s',
              end='\r')
    urllib.request.urlretrieve(CALVIN_URL, CACHE, reporthook=_progress)
    print(f'\nDownload complete: {CACHE.stat().st_size/1e6:.0f} MB')
else:
    print(f'Using cached zip: {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)')

# ── Step 2: Read lang_annotations to find episode boundaries ──────────────
print('\nReading lang_annotations from zip …')

ANN_KEY = f'task_D_D/{SPLIT}/lang_annotations/auto_lang_ann.npy'

with zipfile.ZipFile(CACHE, 'r') as zf:
    all_names = zf.namelist()

    # Find annotation file (path may vary)
    ann_matches = [n for n in all_names if 'auto_lang_ann' in n]
    if not ann_matches:
        raise FileNotFoundError('auto_lang_ann.npy not found in zip')
    ann_key = ann_matches[0]
    print(f'  Found: {ann_key}')

    with zf.open(ann_key) as f:
        ann = np.load(io.BytesIO(f.read()), allow_pickle=True).item()

indx  = ann['info']['indx']   # list of (start_idx, end_idx) per annotated episode
tasks = ann['language']['task']

print(f'  {len(indx)} annotated episodes found')
print(f'  First few tasks: {tasks[:3].tolist()}')

# ── Step 3: Pick N diverse episodes ───────────────────────────────────────
# Pick episodes spaced evenly through the dataset for variety
step  = max(1, len(indx) // N_EPISODES)
picks = list(range(0, len(indx), step))[:N_EPISODES]

needed_files = set()
needed_files.add(ann_key)

for i in picks:
    start_idx, end_idx = indx[i]
    task = tasks[i]
    print(f'  Episode {i}: "{task}"  frames {start_idx}–{end_idx}')
    for frame_idx in range(start_idx, end_idx + 1):
        fname = f'task_D_D/{SPLIT}/episode_{frame_idx:07d}.npz'
        needed_files.add(fname)

print(f'\nFiles needed: {len(needed_files)} ({len(needed_files)-1} .npz + 1 annotation)')

# ── Step 4: Extract needed files into mini zip ─────────────────────────────
print(f'\nBuilding {OUT_ZIP} …')

with zipfile.ZipFile(CACHE, 'r') as src_zf:
    available = set(src_zf.namelist())
    with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as dst_zf:
        for i, fname in enumerate(sorted(needed_files)):
            if fname in available:
                data = src_zf.read(fname)
                dst_zf.writestr(fname, data)
                if i % 20 == 0:
                    print(f'  packed {i}/{len(needed_files)} …', end='\r')
            else:
                print(f'  ⚠️  missing: {fname}')

size_mb = Path(OUT_ZIP).stat().st_size / 1e6
print(f'\n✓ Created {OUT_ZIP} ({size_mb:.0f} MB)')
print(f'  Contains {N_EPISODES} annotated episodes for visualization')

# ── Step 5: Instructions ───────────────────────────────────────────────────
print(f"""
══════════════════════════════════════════════════════
  NEXT STEPS
══════════════════════════════════════════════════════
1. Upload {OUT_ZIP} to Google Drive:
     MyDrive/VERA_CALVIN/calvin_mini.zip

2. In a Colab cell, run:

     import zipfile
     print('Extracting …')
     with zipfile.ZipFile('/content/drive/MyDrive/VERA_CALVIN/calvin_mini.zip') as zf:
         zf.extractall('/content/')
     print('Done — CALVIN at /content/task_D_D')

3. Set CAL_DATA_ROOT = '/content/task_D_D' in the CONFIG cell
══════════════════════════════════════════════════════
""")
