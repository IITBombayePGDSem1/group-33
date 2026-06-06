# IPL Jersey Detection — Project README

**Course:** Programming for Machine Learning and Data Science  
**Institute:** CMInDS, Indian Institute of Technology Bombay  
**Submission Deadline:** June 06, 2026

---
**Instances for testing:**
-Annotation tool: http://35.184.217.3/
-Streamlit app:  http://35.184.217.3/streamlit/

## Team Members

| Name | Initials |
|---|---|
| Eklavya Bhardwaj | E |
| Shantanu | S |
| Prabhu | P |
| Mukund | M |

---

## Problem Statement

Given an image from an IPL match, the system must:

1. Divide the image into an **8×8 grid** (64 cells of 100×75 px each)
2. For each cell, predict which IPL team is present (or 0 for background)
3. Output a CSV row with columns `c01` to `c64` containing values 0–10

No player name identification is required — only team affiliation.

---

## Team Labels

| ID | Team |
|---|---|
| 0 | Background / No team |
| 1 | Chennai Super Kings (CSK) |
| 2 | Delhi Capitals (DC) |
| 3 | Gujarat Titans (GT) |
| 4 | Kolkata Knight Riders (KKR) |
| 5 | Lucknow Super Giants (LSG) |
| 6 | Mumbai Indians (MI) |
| 7 | Punjab Kings (PBKS) |
| 8 | Rajasthan Royals (RR) |
| 9 | Royal Challengers Bengaluru (RCB) |
| 10 | Sunrisers Hyderabad (SRH) |

---

## Dataset

### Image Sources

Images were collected from the following publicly available websites using automated web scraping:

- **ESPNcricinfo** — https://www.espncricinfo.com
- **Cricbuzz** — https://www.cricbuzz.com
- **IPL Official** — https://www.iplt20.com

All images are from the **IPL 2023 season (18th edition)**.

### Specifications

| Property | Value |
|---|---|
| Total images | ~2500 |
| Resolution | 800 × 600 px |
| Aspect ratio | 4:3 |
| Format | JPG / PNG |
| Min resolution policy | Images below 800×600 native resolution were discarded — not upscaled |
| Coverage | All 10 IPL franchises, ≥100 instances each |

### Image Diversity

The dataset covers:
- Multiple players from different franchises in a single frame
- Varying player poses, orientations, and partial/full body visibility
- Different lighting conditions, camera angles, and crowd backgrounds
- Single-player and multi-player scenes
- No-player images (empty pitch, crowd, background) for model generalisation

### Folder Structure

```
ipl_2023_master_dataset/
    ipl2023/                      ← All 800×600 match images
    real_world_references/        ← Per-team reference crops for auto-labelling
        1_CSK/
        2_DC/
        3_GT/
        4_KKR/
        5_LSG/
        6_MI/
        7_PBKS/
        8_RR/
        9_RCB/
        10_SRH/

Labels/
    final_predictions_test_e.csv  ← Eklavya's annotations
    final_predictions_test_m.csv  ← Mukund's annotations
    final_predictions_test_p.csv  ← Prabhu's annotations
    final_predictions_test_s.csv  ← Shantanu's annotations
    merged_predictions.csv        ← Final merged & verified labels
```

---

## Pipeline Overview

```
Web Scraping → Image Preprocessing → Auto-Labelling → Manual Verification → Feature Extraction → Model Training → Inference
```

---

## Step 1 — Web Scraping (`Webscrapping.ipynb`)

- Used **Selenium** (headless Chrome) + **BeautifulSoup** to scrape match photos from ESPNcricinfo, Cricbuzz, and iplt20.com
- Images were downloaded, deduplicated via MD5 hash, and saved to `ipl_2023_master_dataset/ipl2023/`
- Video frames were also extracted using `yt-dlp` + `cv2.VideoCapture` for additional diversity
- All images resized to **800×600 px** using PIL; images below native 800×600 were discarded

---

## Step 2 — Auto-Labelling (`AutoLabelling.ipynb`)

To generate initial labels for 2500 images at 64 cells each (160,000 cells total), an automated labelling pipeline was built:

**Human detection via HOG:**
- OpenCV's `HOGDescriptor` with the default people detector was run at multiple scales and strides to detect player bounding boxes
- Detected boxes were merged using Non-Maximum Suppression (`cv2.groupRectangles`)
- A boolean mask was created mapping detections to the 8×8 grid cells

**Feature extraction (266-dim per cell):**
- HSV histogram (H channel, 16 bins)
- Spatial 2×2 quadrant colour grid
- LBP texture (`P=8, R=1, uniform`)
- HOG edges

**Label assignment:**
- Per-team reference image crops (stored in `real_world_references/`) were used to compute average HSV signatures for each franchise
- Grid cells with detected humans were assigned the closest-matching team label based on colour distance

---

## Step 3 — Manual Verification (Flask Annotation Tool)

A custom **Flask + HTML annotation tool** (packaged as `AttonatorIITBombay.zip`, containing `app.py`, `templates/`, and `static/images/`) was built to allow all four team members to verify auto-generated labels simultaneously:

- Each annotator was assigned a separate CSV file (`_e`, `_m`, `_p`, `_s`)
- The tool displays each image with the 8×8 grid overlay and predicted labels
- Annotators could correct any cell label and mark the image as **Verified**
- A `/merge_all` endpoint merged all verified rows from all four annotators into `merged_predictions.csv`

---

## Step 4 — Model Training (`ModelTraining_final.ipynb`)

### Feature Engineering

Each 100×75 px grid cell is described by a **~424-dimensional handcrafted feature vector**:

| Feature | Description | Dims |
|---|---|---|
| HSV histogram | H(16) + S(8) + V(8) — full colour distribution | 32 |
| Spatial colour grid | 2×2 quadrant H histograms — WHERE the colour is | 32 |
| LBP texture | Local Binary Pattern, P=8, R=1, uniform | 10 |
| HOG edges | 9 orientations, 8×8 px/cell, 2×2 cells/block, L2-Hys | ~200 |
| ORB Bag-of-Visual-Words | 500 keypoints, vocab size 150 via MiniBatchKMeans | 150 |

> **No CNNs or deep learning** — all features are handcrafted, fully compliant with project constraints.

### Class Imbalance Handling

- Class 0 (background) makes up the majority of cells (~85%)
- **Undersampled** class 0 to 5× the average team class count
- `class_weight='balanced'` on the classifier for further correction

### Model

```
HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.03,
    max_depth=8,
    min_samples_leaf=20,
    max_bins=128,
    class_weight='balanced',
    random_state=42
)
```

### Train / Test Split

- 70% train / 30% test (random split, `np.random.seed(42)`)

### Per-Class Confidence Thresholds

After training, per-class confidence thresholds were optimised via a two-stage random search:

- **Stage 1:** 300 random trials, threshold range 0.25–0.55 per team class
- **Stage 2:** 200 fine-grained trials around the best Stage 1 result

If the predicted class confidence is below its threshold, the cell is relabelled as 0 (background). This significantly reduced false positives.

### Post-Processing

Two additional post-processing steps applied after threshold filtering:

1. **Person mask:** Cells not covered by the HOG person detector are forced to 0
2. **Isolated cell suppression:** Cells with no same-team neighbours in 8-directions are removed (two passes)

### Results

| Metric | Value |
|---|---|
| Final Macro F1 (test set) | **0.4881** |

---

## Step 5 — Model Comparison (`ModelTournament.ipynb`)

Multiple classifiers were evaluated before selecting the final model:

- Calibrated LinearSVC
- Random Forest
- HistGradientBoostingClassifier (selected)

HistGradientBoosting was chosen for its best macro F1 score and stable probability estimates needed for threshold optimisation.

---

## Step 6 — Inference Pipeline (`app.py` — Streamlit)

A **Streamlit web application** serves as the inference pipeline:

```bash
streamlit run app.py
```

**Features:**
- Upload any JPG/PNG image
- Automatically resizes to 800×600 and runs the full pipeline
- Toggle HOG person verification on/off
- Visualise the 8×8 grid overlay with team labels and confidence scores
- Displays detected teams with player counts (connected-blob BFS)
- Download prediction as a correctly formatted CSV

**Model file required:** `model_epsm.pkl` in the same directory.  
The pickle contains: `model`, `scaler`, `kmeans_bovw`, `vocab_size`, `thresholds`, `macro_f1`.

---

## Output CSV Format

```
Image File Name, Train Or Test, c01, c02, c03, ..., c64
match_photo_001.jpg, Train, 0, 0, 6, 0, 8, 0, ...
```

- Each `c01`–`c64` value is an integer 0–10
- Cells are numbered left-to-right, top-to-bottom across the 8×8 grid

---

## GitHub Repositories

| Repository | Description | Link |
|---|---|---|
| `group-33` | Main project — notebooks, model, predictions, inference app | https://github.com/IITBombayePGDSem1/group-33 |
| `attonatorIITBOMBAY` | Flask annotation & label verification tool | https://github.com/IITBombayePGDSem1/attonatorIITBOMBAY |

**Live annotation tool** (used by all 4 team members for collaborative label verification):  
http://35.184.217.3/

---

## Libraries Used

### Computer Vision & Image Processing

| Library | Usage |
|---|---|
| `opencv-python` (cv2) | HOG person detection, image resizing, ORB keypoint extraction, colour space conversion |
| `Pillow` (PIL) | Image loading, resizing, format conversion |
| `scikit-image` (skimage) | HOG features, LBP texture (`skimage.feature`), greyscale conversion (`skimage.color`), exposure utilities |

### Machine Learning

| Library | Usage |
|---|---|
| `scikit-learn` | `HistGradientBoostingClassifier`, `RandomForestClassifier`, `LinearSVC`, `CalibratedClassifierCV`, `LogisticRegression`, `MiniBatchKMeans`, `KMeans`, `StandardScaler`, `Pipeline`, `TSNE`, `f1_score`, `classification_report`, `silhouette_score`, `davies_bouldin_score`, `cosine_similarity` |

### Data Handling

| Library | Usage |
|---|---|
| `numpy` | Array operations, feature vectors, random sampling |
| `pandas` | CSV loading, label management, train/test splits |

### Web Scraping

| Library | Usage |
|---|---|
| `selenium` | Headless Chrome browser automation for scraping |
| `beautifulsoup4` (bs4) | HTML parsing and image URL extraction |
| `requests` | HTTP image downloads |
| `yt-dlp` | YouTube/video frame extraction for dataset diversity |

### Visualisation

| Library | Usage |
|---|---|
| `matplotlib` | Grid overlays, feature plots, confusion matrices (`matplotlib.pyplot`, `matplotlib.colors`, `matplotlib.patches`, `matplotlib.patheffects`) |
| `seaborn` | Heatmaps and distribution plots |

### Web Applications

| Library | Usage |
|---|---|
| `streamlit` | Inference pipeline UI (`app.py`) |
| `flask` | Annotation and label verification tool (`AttonatorIITBombay`) |
| `werkzeug` | Secure file uploads in Flask |

### Utilities

| Library | Usage |
|---|---|
| `tqdm` | Progress bars during feature extraction and training |
| `pickle` | Model serialisation / deserialisation |

```
submission/
│
├── Webscrapping.ipynb              ← Image collection from ESPNcricinfo, Cricbuzz, iplt20.com
├── AutoLabelling.ipynb             ← HOG-based auto-labelling pipeline
├── ModelTournament.ipynb           ← Classifier comparison and selection
├── ModelTraining_final.ipynb       ← Final model training, threshold optimisation, evaluation
├── app.py                          ← Streamlit inference pipeline (loads pkl, runs prediction, exports CSV)
│
├── AttonatorIITBombay.zip          ← Flask annotation tool (zipped)
│   ├── app.py                      ← Flask backend
│   ├── templates/                  ← HTML frontend templates
│   └── static/images/              ← Static assets
│
├── model_epsm.pkl                  ← Trained model (model + scaler + BoVW vocab + thresholds)
├── merged_predictions.csv          ← Final labelled dataset (train + test, c01–c64)
├── predictions_epsm.csv            ← Final model predictions output (all 64 cells per image)
├── README.txt                      ← Brief dataset summary
└── README.md                       ← This file
```
