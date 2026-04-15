import os
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "Archives/Archives_1947"
OUTPUT_DIR = "resultats_detection"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

ENABLE_DESKEW = True

MIN_BOX_WIDTH = 250
MIN_BOX_HEIGHT = 80

TOP_CROP_RATIO = 0.03
SIDE_MARGIN_RATIO = 0.02

# Marge initiale sur la boîte détectée
REFINE_MARGIN = 14

# Expansion verticale légère
BOX_PAD_Y = 5

# Page gauche
LEFT_PAGE_LEFT_EXTRA = 30
LEFT_PAGE_RIGHT_SNAP_PAD = 8

# Page droite
RIGHT_PAGE_LEFT_SNAP_PAD = 8
RIGHT_PAGE_RIGHT_EXTRA = 24

# Recherche autour du cadre
SIDE_SEARCH_MARGIN = 40

# Stop si trop de colonnes vides d'affilée
EMPTY_RUN_STOP = 8

# Colonne considérée vide si très peu d'encre
MIN_INK_PER_COL = 1

# Fusion finale plus prudente
MERGE_Y_GAP = 10
MERGE_X_TOL = 60

# Re-découpage des boîtes trop hautes
ENABLE_SECONDARY_SPLIT = True
SECONDARY_SPLIT_SMOOTH = 17
SECONDARY_SPLIT_THRESHOLD_RATIO = 0.22
SECONDARY_SPLIT_MIN_GAP_LEN = 10
SECONDARY_SPLIT_MIN_PART_HEIGHT = 70
SECONDARY_SPLIT_TALL_BOX_FACTOR = 1.35


# ============================================================
# OUTILS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(folder):
    files = []
    for name in os.listdir(folder):
        p = Path(folder) / name
        if p.suffix.lower() in VALID_EXTENSIONS:
            files.append(str(p))
    files.sort()
    return files


def resize_if_needed(img, max_width=2200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def find_intervals(signal, threshold, min_len=20):
    mask = signal > threshold
    intervals = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                intervals.append((start, i))
            start = None

    if start is not None:
        if len(signal) - start >= min_len:
            intervals.append((start, len(signal)))

    return intervals


def find_low_intervals(signal, threshold, min_len=5):
    mask = signal <= threshold
    intervals = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                intervals.append((start, i))
            start = None

    if start is not None:
        if len(signal) - start >= min_len:
            intervals.append((start, len(signal)))

    return intervals


def smooth_signal(signal, ksize=31):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones(ksize, dtype=np.float32) / ksize
    return np.convolve(signal, kernel, mode="same")


def deskew_image(gray):
    coords = np.column_stack(np.where(gray < 220))
    if len(coords) < 100:
        return gray

    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.3:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def crop_useful_region(gray):
    h, w = gray.shape[:2]

    top = int(h * TOP_CROP_RATIO)
    left = int(w * SIDE_MARGIN_RATIO)
    right = int(w * (1 - SIDE_MARGIN_RATIO))

    cropped = gray[top:h, left:right]
    return cropped, (left, top)


def binarize(gray):
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    return bw


def remove_small_components(bw, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ============================================================
# DETECTION COLONNES
# ============================================================

def detect_text_columns(bw):
    vertical_projection = bw.sum(axis=0) / 255.0
    vertical_smooth = smooth_signal(vertical_projection, ksize=61)

    threshold = max(40, vertical_smooth.max() * 0.55)
    raw_cols = find_intervals(vertical_smooth, threshold=threshold, min_len=180)

    return raw_cols, vertical_projection, vertical_smooth


# ============================================================
# DETECTION PARAGRAPHES
# ============================================================

def refine_box_with_content(bw, x1, y1, x2, y2, margin=REFINE_MARGIN):
    sub = bw[y1:y2, x1:x2]
    ys, xs = np.where(sub > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    xx1 = x1 + max(xs.min() - margin, 0)
    xx2 = x1 + min(xs.max() + margin, sub.shape[1] - 1)
    yy1 = y1 + max(ys.min() - margin, 0)
    yy2 = y1 + min(ys.max() + margin, sub.shape[0] - 1)

    if xx2 - xx1 < MIN_BOX_WIDTH or yy2 - yy1 < MIN_BOX_HEIGHT:
        return None

    return (xx1, yy1, xx2, yy2)


def detect_paragraphs_in_column(bw, x1, x2):
    column = bw[:, x1:x2]

    horizontal_projection = column.sum(axis=1) / 255.0
    horizontal_smooth = smooth_signal(horizontal_projection, ksize=21)

    threshold = max(5, horizontal_smooth.max() * 0.18)
    spans = find_intervals(horizontal_smooth, threshold=threshold, min_len=40)

    boxes = []
    for y1, y2 in spans:
        box = refine_box_with_content(bw, x1, y1, x2, y2, margin=REFINE_MARGIN)
        if box is not None:
            boxes.append(box)

    return boxes, horizontal_projection, horizontal_smooth


# ============================================================
# AJUSTEMENT HORIZONTAL
# ============================================================

def expand_side_until_blank(bw, start_x, direction, limit_x, y1, y2,
                            min_ink=MIN_INK_PER_COL,
                            empty_run_stop=EMPTY_RUN_STOP):
    h, w = bw.shape[:2]
    y1 = max(0, y1)
    y2 = min(h, y2)

    x = start_x
    best = start_x
    empty_run = 0

    while True:
        nx = x + direction

        if direction < 0 and nx < limit_x:
            break
        if direction > 0 and nx > limit_x:
            break

        ink = np.count_nonzero(bw[y1:y2, nx])

        if ink > min_ink:
            best = nx
            empty_run = 0
        else:
            empty_run += 1
            if empty_run >= empty_run_stop:
                break

        x = nx

    return best


def adjust_box_to_column(bw, box, col_x1, col_x2):
    h, w = bw.shape[:2]
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    mid = w // 2

    band_y1 = max(0, y1 - 2)
    band_y2 = min(h, y2 + 2)

    new_y1 = max(0, y1 - BOX_PAD_Y)
    new_y2 = min(h - 1, y2 + BOX_PAD_Y)

    if cx < mid:
        left_limit = max(0, min(col_x1, x1) - SIDE_SEARCH_MARGIN)
        best_left = expand_side_until_blank(
            bw,
            start_x=x1,
            direction=-1,
            limit_x=left_limit,
            y1=band_y1,
            y2=band_y2
        )

        new_x1 = max(0, best_left - LEFT_PAGE_LEFT_EXTRA)
        new_x2 = min(w - 1, col_x2 + LEFT_PAGE_RIGHT_SNAP_PAD + 30)

    else:
        new_x1 = max(0, col_x1 - RIGHT_PAGE_LEFT_SNAP_PAD)

        right_limit = min(w - 1, max(col_x2, x2) + SIDE_SEARCH_MARGIN)
        best_right = expand_side_until_blank(
            bw,
            start_x=x2,
            direction=1,
            limit_x=right_limit,
            y1=band_y1,
            y2=band_y2
        )

        new_x2 = min(w - 1, best_right + RIGHT_PAGE_RIGHT_EXTRA)

    if new_x2 - new_x1 < MIN_BOX_WIDTH:
        return box

    return (new_x1, new_y1, new_x2, new_y2)


# ============================================================
# RE-DECOUPAGE DES BOITES TROP HAUTES
# ============================================================

def split_tall_box_if_needed(bw, box, typical_height):
    x1, y1, x2, y2 = box
    height = y2 - y1

    if height < max(MIN_BOX_HEIGHT * 1.8, typical_height * SECONDARY_SPLIT_TALL_BOX_FACTOR):
        return [box]

    sub = bw[y1:y2, x1:x2]
    row_proj = sub.sum(axis=1) / 255.0
    row_smooth = smooth_signal(row_proj, ksize=SECONDARY_SPLIT_SMOOTH)

    thr = max(2, row_smooth.max() * SECONDARY_SPLIT_THRESHOLD_RATIO)
    low_gaps = find_low_intervals(
        row_smooth,
        threshold=thr,
        min_len=SECONDARY_SPLIT_MIN_GAP_LEN
    )

    if not low_gaps:
        return [box]

    valid_cuts = []
    for gy1, gy2 in low_gaps:
        gap_mid = (gy1 + gy2) // 2

        top_h = gap_mid
        bot_h = height - gap_mid

        if top_h >= SECONDARY_SPLIT_MIN_PART_HEIGHT and bot_h >= SECONDARY_SPLIT_MIN_PART_HEIGHT:
            valid_cuts.append((gy1, gy2, gap_mid))

    if not valid_cuts:
        return [box]

    best = max(valid_cuts, key=lambda g: g[1] - g[0])
    cut_y = y1 + best[2]

    top_box = refine_box_with_content(bw, x1, y1, x2, cut_y, margin=8)
    bot_box = refine_box_with_content(bw, x1, cut_y, x2, y2, margin=8)

    result = []
    if top_box is not None:
        result.append(top_box)
    if bot_box is not None:
        result.append(bot_box)

    if len(result) == 2:
        return result

    return [box]


def secondary_split_boxes(bw, boxes):
    if not boxes:
        return boxes

    heights = [y2 - y1 for (_, y1, _, y2) in boxes]
    typical_height = int(np.median(heights)) if heights else MIN_BOX_HEIGHT

    new_boxes = []
    for box in boxes:
        split_boxes = split_tall_box_if_needed(bw, box, typical_height)
        new_boxes.extend(split_boxes)

    return new_boxes


# ============================================================
# FUSION ET TRI
# ============================================================

def merge_close_boxes(boxes, y_gap=MERGE_Y_GAP, x_tol=MERGE_X_TOL):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = []

    current = list(boxes[0])

    for b in boxes[1:]:
        same_column = abs(b[0] - current[0]) < x_tol and abs(b[2] - current[2]) < x_tol
        close_vertically = 0 <= b[1] - current[3] <= y_gap

        if same_column and close_vertically:
            current[0] = min(current[0], b[0])
            current[1] = min(current[1], b[1])
            current[2] = max(current[2], b[2])
            current[3] = max(current[3], b[3])
        else:
            merged.append(tuple(current))
            current = list(b)

    merged.append(tuple(current))
    return merged


def sort_reading_order(boxes):
    return sorted(boxes, key=lambda b: (b[0], b[1]))


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def detect_paragraph_boxes(gray):
    if ENABLE_DESKEW:
        gray = deskew_image(gray)

    useful_gray, (offset_x, offset_y) = crop_useful_region(gray)
    bw = binarize(useful_gray)
    bw = remove_small_components(bw, min_area=20)

    columns, _, _ = detect_text_columns(bw)

    all_boxes = []

    for col_x1, col_x2 in columns:
        col_boxes, _, _ = detect_paragraphs_in_column(bw, col_x1, col_x2)

        if ENABLE_SECONDARY_SPLIT:
            col_boxes = secondary_split_boxes(bw, col_boxes)

        adjusted_boxes = []
        for box in col_boxes:
            adjusted = adjust_box_to_column(bw, box, col_x1, col_x2)
            adjusted_boxes.append(adjusted)

        all_boxes.extend(adjusted_boxes)

    all_boxes = merge_close_boxes(all_boxes, y_gap=MERGE_Y_GAP, x_tol=MERGE_X_TOL)
    all_boxes = sort_reading_order(all_boxes)

    restored_boxes = []
    for x1, y1, x2, y2 in all_boxes:
        restored_boxes.append((
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y
        ))

    return restored_boxes, useful_gray, bw


# ============================================================
# AFFICHAGE
# ============================================================

def draw_boxes(image_bgr, boxes):
    out = image_bgr.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 4)

        label = f"Acte {i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        text_x = x1 + 5
        text_y = max(25, y1 - 8)

        cv2.rectangle(
            out,
            (text_x - 4, text_y - th - 8),
            (text_x + tw + 4, text_y + 4),
            (255, 255, 255),
            -1
        )
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return out


def save_crops(gray, boxes, output_folder, base_name):
    crops_dir = Path(output_folder) / "crops" / base_name
    ensure_dir(crops_dir)

    saved = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = gray[y1:y2, x1:x2]
        out_path = crops_dir / f"{base_name}_acte_{i:02d}.png"
        cv2.imwrite(str(out_path), crop)
        saved.append(str(out_path))

    return saved


# ============================================================
# EXECUTION
# ============================================================

def process_image(image_path, output_dir):
    print(f"Traitement : {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Impossible de lire l'image.")
        return

    image_bgr = resize_if_needed(image_bgr, max_width=2200)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    boxes, _, _ = detect_paragraph_boxes(gray)

    base_name = Path(image_path).stem
    annotated = draw_boxes(image_bgr, boxes)

    annotated_dir = Path(output_dir) / "annotated"
    ensure_dir(annotated_dir)

    annotated_path = annotated_dir / f"{base_name}_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    crop_paths = save_crops(gray, boxes, output_dir, base_name)

    print(f"  {len(boxes)} actes détectés")
    print(f"  Image annotée : {annotated_path}")
    print(f"  Crops enregistrés : {len(crop_paths)}")


def main():
    ensure_dir(OUTPUT_DIR)

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"Aucune image trouvée dans : {INPUT_DIR}")
        return

    for image_path in image_paths:
        process_image(image_path, OUTPUT_DIR)

    print("\nTerminé.")


if __name__ == "__main__":
    main()