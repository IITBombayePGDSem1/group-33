import streamlit as st
import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd

st.set_page_config(
    page_title="IPL Jersey Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 700; }
.page-title {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
    color: #f5f5f5; margin-bottom: 4px;
    background: linear-gradient(135deg, #8a8a8a 60%, #b87010);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-subtitle {
    font-size: 0.8rem; color: #666; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 24px; font-weight: 500;
}
.info-box {
    background: #13131f; border-left: 3px solid #e8a020;
    padding: 11px 16px; border-radius: 0 8px 8px 0;
    margin: 8px 0; font-size: 0.84rem; color: #bbb; line-height: 1.7;
}
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #666; margin-bottom: 10px;
}
.team-card {
    border-radius: 10px; padding: 18px 14px; text-align: center;
    margin: 4px; position: relative; overflow: hidden;
}
div[data-testid="stHorizontalBlock"] { gap: 12px; }
[data-testid="stMetric"] {
    background: #111118; border: 1px solid #1e1e2e;
    border-radius: 10px; padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants
TEAM_NAMES = {
    0: 'Background', 1: 'CSK', 2: 'DC', 3: 'GT', 4: 'KKR',
    5: 'LSG', 6: 'MI', 7: 'PBKS', 8: 'RR', 9: 'RCB', 10: 'SRH'
}
TEAM_FULL_NAMES = {
    0: 'Background', 1: 'Chennai Super Kings', 2: 'Delhi Capitals',
    3: 'Gujarat Titans', 4: 'Kolkata Knight Riders', 5: 'Lucknow Super Giants',
    6: 'Mumbai Indians', 7: 'Punjab Kings', 8: 'Rajasthan Royals',
    9: 'Royal Challengers Bengaluru', 10: 'Sunrisers Hyderabad'
}
TEAM_COLORS = {
    0: '#444444', 1: '#FDB913', 2: '#0078BC', 3: '#1D4F91',
    4: '#3B1F6E', 5: '#A0C2F0', 6: '#005DA0', 7: '#ED1B24',
    8: '#EA1A85', 9: '#EC1C24', 10: '#F7A721'
}
TEAM_TEXT_COLORS = {
    0: '#fff', 1: '#000', 2: '#fff', 3: '#fff', 4: '#fff',
    5: '#000', 6: '#fff', 7: '#fff', 8: '#fff', 9: '#fff', 10: '#000'
}


@st.cache_resource
def get_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def get_person_cells(img_bgr):
    """
    Detect humans using OpenCV HOG detector.
    Tuned for cricket images where players are smaller and further away.
    Returns (person_cells set, raw bounding boxes list, n_detections int).
    """
    hog_det = get_hog_detector()
    img     = cv2.resize(img_bgr, (800, 600))

    # Try multiple scales to catch far-away players
    # Smaller winStride = more thorough scan, scale closer to 1 = finer pyramid
    boxes_all = []
    for win_stride, padding, scale in [
        ((4, 4), (8, 8), 1.03),   # fine scan — catches small/distant players
        ((8, 8), (8, 8), 1.05),   # standard scan
        ((4, 4), (4, 4), 1.08),   # wide scan
    ]:
        boxes, weights = hog_det.detectMultiScale(
            img,
            winStride=win_stride,
            padding=padding,
            scale=scale
        )
        if len(boxes) > 0:
            for box in boxes:
                boxes_all.append(box)

    if len(boxes_all) == 0:
        return set(), [], 0

    # Non-maximum suppression to merge overlapping boxes
    boxes_np = np.array(boxes_all)
    boxes_np, _ = cv2.groupRectangles(
        [b.tolist() for b in boxes_np] * 2,
        groupThreshold=1,
        eps=0.3
    )

    if len(boxes_np) == 0:
        return set(), [], 0

    # Map bounding boxes to grid cells
    person_cells = set()
    for (x, y, w, h) in boxes_np:
        for r in range(8):
            for c in range(8):
                cx1, cy1 = c * 100, r * 75
                cx2, cy2 = cx1 + 100, cy1 + 75
                # Require at least 30% overlap to count as a person cell
                ix1 = max(x, cx1); iy1 = max(y, cy1)
                ix2 = min(x+w, cx2); iy2 = min(y+h, cy2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = (ix2-ix1)*(iy2-iy1)
                    cell_area = 100 * 75
                    if overlap / cell_area >= 0.20:
                        person_cells.add((r, c))

    return person_cells, boxes_np.tolist(), len(boxes_np)


def get_neighbor_cells(person_cells, grid_shape=(8, 8)):
    """
    Return the set of cells that are direct neighbours (4-dir) of
    any person cell — used to show cell numbers around players.
    """
    neighbors = set()
    for (r, c) in person_cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                if (nr, nc) not in person_cells:
                    neighbors.add((nr, nc))
    return neighbors


def extract_all_handcrafted_features(cell_image, kmeans_model, vocab_size, orb_detector):
    from skimage.feature import hog, local_binary_pattern

    hsv_img = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    hist_h, _ = np.histogram(hsv_img[:, :, 0], bins=16, range=(0, 180))
    hist_s, _ = np.histogram(hsv_img[:, :, 1], bins=8,  range=(0, 256))
    hist_v, _ = np.histogram(hsv_img[:, :, 2], bins=8,  range=(0, 256))
    hist_h = hist_h.astype(float) / (hist_h.sum() + 1e-7)
    hist_s = hist_s.astype(float) / (hist_s.sum() + 1e-7)
    hist_v = hist_v.astype(float) / (hist_v.sum() + 1e-7)
    hsv_features = np.concatenate([hist_h, hist_s, hist_v])

    h, w = hsv_img.shape[:2]
    spatial_feats = []
    for qr in range(2):
        for qc in range(2):
            quad = hsv_img[qr*h//2:(qr+1)*h//2, qc*w//2:(qc+1)*w//2, 0]
            q_hist, _ = np.histogram(quad, bins=8, range=(0, 180))
            q_hist = q_hist.astype(float) / (q_hist.sum() + 1e-7)
            spatial_feats.extend(q_hist)
    spatial_features = np.array(spatial_feats)

    gray_img = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist_lbp = hist_lbp.astype(float) / (hist_lbp.sum() + 1e-7)

    hog_features = hog(
        gray_img, orientations=9,
        pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualize=False
    )

    keypoints, descriptors = orb_detector.detectAndCompute(gray_img, None)
    hist_bovw = np.zeros(vocab_size, dtype=float)
    if descriptors is not None:
        descriptors = np.array(descriptors, dtype=np.float32)
        words = kmeans_model.predict(descriptors)
        for word in words:
            hist_bovw[word] += 1.0
        hist_bovw /= (hist_bovw.sum() + 1e-7)

    return np.concatenate([hsv_features, spatial_features, hist_lbp, hog_features, hist_bovw])


def apply_thresholds(probabilities, thresholds):
    preds = []
    for prob in probabilities:
        bc = int(np.argmax(prob))
        if bc != 0 and prob[bc] < thresholds[bc]:
            preds.append(0)
        else:
            preds.append(bc)
    return preds


def suppress_isolated_cells(preds):
    grid    = np.array(preds).reshape(8, 8)
    cleaned = grid.copy()
    # Pass 1: 4-dir no non-zero neighbours
    for r in range(8):
        for c in range(8):
            if grid[r, c] == 0: continue
            nb4 = []
            if r > 0: nb4.append(grid[r-1, c])
            if r < 7: nb4.append(grid[r+1, c])
            if c > 0: nb4.append(grid[r, c-1])
            if c < 7: nb4.append(grid[r, c+1])
            if all(n == 0 for n in nb4):
                cleaned[r, c] = 0
    # Pass 2: 8-dir no same-team neighbours
    cleaned2 = cleaned.copy()
    for r in range(8):
        for c in range(8):
            team = cleaned[r, c]
            if team == 0: continue
            same = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        same.append(cleaned[nr, nc] == team)
            if not any(same):
                cleaned2[r, c] = 0
    return cleaned2.flatten().tolist()


def apply_person_mask(preds, person_cells):
    if len(person_cells) == 0:
        return preds
    masked = list(preds)
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if masked[idx] != 0 and (r, c) not in person_cells:
                masked[idx] = 0
    return masked



def count_players_by_team(grid):
    # Count distinct players per team using 4-connected BFS blob detection.
    # One connected blob of same-team cells = one player.
    visited = np.zeros((8, 8), dtype=bool)
    team_player_counts = {}
    for r in range(8):
        for c in range(8):
            team = grid[r, c]
            if team == 0 or visited[r, c]:
                continue
            queue = [(r, c)]
            visited[r, c] = True
            while queue:
                cr, cc = queue.pop()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (0 <= nr < 8 and 0 <= nc < 8
                            and not visited[nr, nc]
                            and grid[nr, nc] == team):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            team_player_counts[team] = team_player_counts.get(team, 0) + 1
    return team_player_counts

@st.cache_resource
def load_model_from_path(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_artifact():
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_epsm.pkl'),
        os.path.join(os.getcwd(), 'model_epsm.pkl'),
    ]:
        if os.path.exists(candidate):
            return load_model_from_path(candidate), None
    return None, "model_epsm.pkl not found in working directory."


def run_inference(img_bgr, artifact, use_person_mask_flag=True):
    model      = artifact['model']
    scaler     = artifact['scaler']
    kmeans_bvw = artifact['kmeans_bovw']
    vocab_size = artifact['vocab_size']
    thresholds = artifact['thresholds']
    orb        = cv2.ORB_create(nfeatures=500)

    img = cv2.resize(img_bgr, (800, 600))

    if use_person_mask_flag:
        person_cells, hog_boxes, n_hog = get_person_cells(img_bgr)
    else:
        person_cells, hog_boxes, n_hog = set(), [], 0

    features_list = []
    for r in range(8):
        for c in range(8):
            cell_img = img[r*75:(r+1)*75, c*100:(c+1)*100]
            features_list.append(
                extract_all_handcrafted_features(cell_img, kmeans_bvw, vocab_size, orb)
            )

    X     = scaler.transform(np.array(features_list))
    probs = model.predict_proba(X)
    preds = apply_thresholds(probs, thresholds)

    if use_person_mask_flag:
        preds = apply_person_mask(preds, person_cells)

    preds = suppress_isolated_cells(preds)

    return preds, probs, person_cells, hog_boxes, n_hog


def make_grid_figure(img_bgr, grid, probs_array, person_cells,
                     hog_boxes, show_person_boxes):
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (800, 600)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.imshow(img_rgb, alpha=0.85)

    # Cells that have a team label
    active_cells = {
        (r, c) for r in range(8) for c in range(8) if grid[r, c] > 0
    }
    # Cells adjacent to active cells — show cell numbers here only
    neighbor_cells = get_neighbor_cells(active_cells)

    # Draw grid lines only around active + neighbour region
    relevant_rows = set()
    relevant_cols = set()
    for (r, c) in active_cells | neighbor_cells:
        relevant_rows.add(r)
        relevant_rows.add(r + 1)
        relevant_cols.add(c)
        relevant_cols.add(c + 1)

    for r in range(9):
        if r in relevant_rows:
            ax.axhline(r * 75, color='#e8a020', linestyle='--',
                       linewidth=0.6, alpha=0.7)
    for c in range(9):
        if c in relevant_cols:
            ax.axvline(c * 100, color='#e8a020', linestyle='--',
                       linewidth=0.6, alpha=0.7)

    # Draw HOG raw bounding boxes
    if show_person_boxes and len(hog_boxes) > 0:
        for (x, y, w, h) in hog_boxes:
            ax.add_patch(plt.Rectangle(
                (x, y), w, h,
                fill=False, edgecolor='#00ff99', linewidth=1.5,
                linestyle='-', alpha=0.6
            ))

    for r in range(8):
        for c in range(8):
            pc   = grid[r, c]
            cx   = c * 100 + 50
            cy   = r * 75 + 37.5
            conf = float(np.max(probs_array[r * 8 + c])) * 100
            cell_num = r * 8 + c + 1

            if pc > 0:
                # Colored fill for team cells
                ax.add_patch(plt.Rectangle(
                    (c*100, r*75), 100, 75,
                    fill=True, facecolor=TEAM_COLORS[pc], alpha=0.28, linewidth=0
                ))
                txt = ax.text(cx, cy - 7, TEAM_NAMES[pc],
                              color=TEAM_COLORS[pc], fontsize=12,
                              ha='center', va='center', fontweight='bold',
                              fontfamily='monospace')
                txt.set_path_effects([
                    PathEffects.withStroke(linewidth=2.5, foreground='black')
                ])
                ax.text(cx, cy + 11, f'{conf:.0f}%',
                        color='white', fontsize=7.5,
                        ha='center', va='center', alpha=0.75)
                # Cell number in corner of active cell
                ax.text(c*100 + 4, r*75 + 9, str(cell_num),
                        color='white', fontsize=6.5,
                        ha='left', va='top', alpha=0.6,
                        fontfamily='monospace')

            elif (r, c) in neighbor_cells:
                # Show cell number only for cells adjacent to a player cell
                ax.text(c*100 + 4, r*75 + 9, str(cell_num),
                        color='white', fontsize=6.5,
                        ha='left', va='top', alpha=0.35,
                        fontfamily='monospace')

    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ── Load model
artifact, load_error = get_artifact()

# ══════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0">
        <div style="font-size:1.3rem;font-weight:700;color:#f0f0f0">IPL Jersey Detector</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px">CMInDS · IIT Bombay</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px">- Eklavya Bhardwaj</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px">- Shantanu</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px">- Prabhu</div>
        <div style="font-size:0.75rem;color:#666;margin-top:2px">- Mukund</div>
    </div>
    """, unsafe_allow_html=True)

    if artifact is not None:
        macro_f1 = artifact.get('macro_f1', 0.4881)
        st.markdown(f"""
        <div style="background:#0f2a1a;border:1px solid #2a6a3a;border-radius:8px;
                    padding:12px 14px;margin-bottom:16px">
            <div style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;
                        text-transform:uppercase;color:#5ecf7a;margin-bottom:6px">
                Model Ready
            </div>
            <div style="font-size:0.82rem;color:#aaa;line-height:1.8">
                HistGradientBoosting<br>
                Macro F1 &nbsp;<b style="color:#f0f0f0">{macro_f1:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#2a1a0f;border:1px solid #6a3a2a;border-radius:8px;
                    padding:12px 14px;margin-bottom:8px">
            <div style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;
                        text-transform:uppercase;color:#cf7a5e;margin-bottom:4px">
                Model Not Found
            </div>
            <div style="font-size:0.82rem;color:#aaa">{load_error}</div>
        </div>
        """, unsafe_allow_html=True)
        model_file = st.file_uploader("Upload model_epsm.pkl", type=["pkl"],
                                      label_visibility="collapsed")
        if model_file is not None:
            tmp_path = "/tmp/uploaded_model.pkl"
            with open(tmp_path, "wb") as f:
                f.write(model_file.read())
            try:
                artifact = load_model_from_path(tmp_path)
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.divider()

    st.markdown("""
    <div style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;
                text-transform:uppercase;color:#666;margin-bottom:10px">
        Detection Settings
    </div>
    """, unsafe_allow_html=True)

    use_person_mask = st.toggle(
        "HOG Person Verification",
        value=True,
        help="Uses OpenCV's HOG person detector to verify each cell contains "
             "a human before assigning a team label."
    )
    show_person_boxes = st.toggle(
        "Show HOG Bounding Boxes",
        value=False,
        help="Draws the raw bounding boxes returned by the HOG detector "
             "(green outlines) on the grid overlay."
    )

    st.divider()

    st.markdown("""
    <div style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;
                text-transform:uppercase;color:#666;margin-bottom:10px">
        IPL Teams
    </div>
    """, unsafe_allow_html=True)

    html_teams = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">'
    for tid in range(1, 11):
        html_teams += (
            f'<div style="background:{TEAM_COLORS[tid]};color:{TEAM_TEXT_COLORS[tid]};'
            f'border-radius:6px;padding:6px 8px;font-size:0.78rem;font-weight:600;'
            f'text-align:center;line-height:1.3">'
            f'{TEAM_NAMES[tid]}<br>'
            f'<span style="font-size:0.65rem;font-weight:400;opacity:0.85">'
            f'{TEAM_FULL_NAMES[tid].replace(" ", "<br>")}</span>'
            f'</div>'
        )
    html_teams += '</div>'
    st.markdown(html_teams, unsafe_allow_html=True)

# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════
st.markdown('<div class="page-title">IPL Jersey Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Automated Team Detection &nbsp;&middot;&nbsp; '
    'Player Classification &nbsp;&middot;&nbsp; IIT Bombay &mdash; CMInDS</div>',
    unsafe_allow_html=True
)
st.divider()

uploaded_img = st.file_uploader(
    "Upload an IPL match image — JPG or PNG, recommended resolution 800x600 or higher",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_img is not None and artifact is not None:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="section-label">Original Image</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
        h, w = img_bgr.shape[:2]
        st.caption(f"Input resolution: {w}x{h}px — resized to 800x600 for inference")

    with col2:
        st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
        pipeline_info = (
            "HOG person verification <b>enabled</b> — only cells containing a "
            "detected human will be assigned a team label."
            if use_person_mask else
            "HOG person verification <b>disabled</b> — all 64 cells classified "
            "directly by the jersey classifier."
        )
        st.markdown(f'<div class="info-box">{pipeline_info}</div>',
                    unsafe_allow_html=True)
        run_btn = st.button("Run Detection")

    if run_btn:
        with st.spinner("Running inference — person detection + jersey classification..."):
            preds, probs, person_cells, hog_boxes, n_hog = run_inference(
                img_bgr, artifact, use_person_mask
            )

        grid = np.array(preds).reshape(8, 8)
        detected_cells = [
            (r * 8 + c + 1, grid[r, c])
            for r in range(8) for c in range(8) if grid[r, c] > 0
        ]
        team_counts = {}
        for _, tid in detected_cells:
            team_counts[tid] = team_counts.get(tid, 0) + 1

        # Connected-blob player count — one blob = one player
        player_counts = count_players_by_team(grid)

        st.divider()
        st.markdown('<div class="section-label">Detection Results</div>',
                    unsafe_allow_html=True)

        top_team = max(player_counts, key=player_counts.get) if player_counts else 0
        total_players = sum(player_counts.values())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Players Detected", total_players)
        m2.metric("Teams Identified", len(player_counts))
        m3.metric("HOG Detections", n_hog if use_person_mask else "—")
        m4.metric("Dominant Team", TEAM_NAMES.get(top_team, "None"))

        st.markdown(
            '<div class="section-label" style="margin-top:16px">Grid Overlay</div>',
            unsafe_allow_html=True
        )
        fig = make_grid_figure(
            img_bgr, grid, probs, person_cells, hog_boxes, show_person_boxes
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if player_counts:
            st.markdown(
                '<div class="section-label" style="margin-top:16px">Teams Detected</div>',
                unsafe_allow_html=True
            )
            sorted_teams = sorted(player_counts.items(), key=lambda x: -x[1])
            n_cols = min(len(sorted_teams), 5)
            cols   = st.columns(n_cols)
            for i, (tid, count) in enumerate(sorted_teams):
                with cols[i % n_cols]:
                    badge = (
                        '<div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;'
                        'text-transform:uppercase;opacity:0.75;margin-bottom:6px">Most Players</div>'
                        if i == 0 else '<div style="margin-bottom:20px"></div>'
                    )
                    plural = "players" if count > 1 else "player"
                    st.markdown(
                        f'<div class="team-card" style="background:{TEAM_COLORS[tid]};'
                        f'color:{TEAM_TEXT_COLORS[tid]}">'
                        f'{badge}'
                        f'<div style="font-size:1.5rem;font-weight:800;letter-spacing:-0.5px">'
                        f'{TEAM_NAMES[tid]}</div>'
                        f'<div style="font-size:0.72rem;font-weight:500;opacity:0.8;'
                        f'margin-top:3px;line-height:1.4">{TEAM_FULL_NAMES[tid]}</div>'
                        f'<div style="font-size:2rem;font-weight:800;margin-top:12px;'
                        f'line-height:1">{count}</div>'
                        f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:1.5px;'
                        f'text-transform:uppercase;opacity:0.75;margin-top:2px">{plural}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No player cells detected in this image.")

        with st.expander("Raw Grid Predictions (8x8 table)"):
            grid_df = pd.DataFrame(
                [[f"{grid[r,c]} - {TEAM_NAMES[grid[r,c]]}" for c in range(8)]
                 for r in range(8)],
                index=[f"Row {r+1}" for r in range(8)],
                columns=[f"Col {c+1}" for c in range(8)]
            )
            st.dataframe(grid_df, use_container_width=True)

        st.divider()
        csv_row = {"Image File Name": uploaded_img.name, "Train Or Test": "Test"}
        for i, p in enumerate(preds, 1):
            csv_row[f"c{i:02d}"] = p
        csv_bytes = pd.DataFrame([csv_row]).to_csv(index=False).encode()
        st.download_button(
            label="Download Prediction CSV",
            data=csv_bytes,
            file_name=f"prediction_{uploaded_img.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv"
        )

elif uploaded_img is not None and artifact is None:
    st.warning("Model not loaded. Please upload model_epsm.pkl in the sidebar.")

else:
    st.markdown("""
    <div style="margin-top:32px;border:1px dashed #333;border-radius:12px;
                padding:48px 32px;text-align:center;background:#111118">
        <div style="display:flex;justify-content:center;gap:6px;margin-bottom:20px">
            <div style="width:12px;height:12px;border-radius:50%;background:#FDB913"></div>
            <div style="width:12px;height:12px;border-radius:50%;background:#0078BC"></div>
            <div style="width:12px;height:12px;border-radius:50%;background:#3B1F6E"></div>
            <div style="width:12px;height:12px;border-radius:50%;background:#ED1B24"></div>
            <div style="width:12px;height:12px;border-radius:50%;background:#005DA0"></div>
        </div>
        <div style="font-size:1.3rem;font-weight:700;color:#e8e8e8;letter-spacing:-0.3px">
            Upload an IPL match image to detect players by team
        </div>
        <div style="font-size:0.83rem;color:#555;margin-top:10px;line-height:1.8;
                    max-width:520px;margin-left:auto;margin-right:auto">
            The model divides each image into an <b style="color:#888">8x8 grid</b>,
            detects players via HOG, and classifies each player cell as one of
            10 IPL franchises using handcrafted visual features.
        </div>
        <div style="display:flex;justify-content:center;gap:32px;margin-top:28px">
            <div style="text-align:center">
                <div style="font-size:1.5rem;font-weight:800;color:#e8a020">64</div>
                <div style="font-size:0.68rem;color:#555;text-transform:uppercase;
                            letter-spacing:1.5px;font-weight:600;margin-top:2px">Grid Cells</div>
            </div>
            <div style="width:1px;background:#2a2a3a"></div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;font-weight:800;color:#e8a020">10</div>
                <div style="font-size:0.68rem;color:#555;text-transform:uppercase;
                            letter-spacing:1.5px;font-weight:600;margin-top:2px">IPL Teams</div>
            </div>
            <div style="width:1px;background:#2a2a3a"></div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;font-weight:800;color:#e8a020">0.49</div>
                <div style="font-size:0.68rem;color:#555;text-transform:uppercase;
                            letter-spacing:1.5px;font-weight:600;margin-top:2px">Macro F1</div>
            </div>
            <div style="width:1px;background:#2a2a3a"></div>
            <div style="text-align:center">
                <div style="font-size:1.5rem;font-weight:800;color:#e8a020">424</div>
                <div style="font-size:0.68rem;color:#555;text-transform:uppercase;
                            letter-spacing:1.5px;font-weight:600;margin-top:2px">Features / Cell</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
