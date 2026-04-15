"""
Test visuel du pipeline de prétraitement.

Affiche l'image à chaque étape de la transformation avant passage à l'OCR.

Usage :
    python test_preprocessing_visual.py
    python test_preprocessing_visual.py chemin/vers/image.jpg
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from app_core.pipeline.decoupage import (
    resize_if_needed,
    deskew_image,
    crop_useful_region,
    binarize,
    remove_small_components,
    detect_text_columns,
    detect_paragraphs_in_column,
    secondary_split_boxes,
    adjust_box_to_column,
    merge_close_boxes,
    sort_reading_order,
    smooth_signal,
    ENABLE_DESKEW,
    MERGE_Y_GAP,
    MERGE_X_TOL,
)
from app_core.pipeline.decoupage import draw_boxes


def bgr_to_rgb(img):
    """OpenCV BGR → matplotlib RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gray_to_rgb(img):
    """Niveaux de gris → RGB pour matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def show(title, img, cmap=None, figsize=(14, 8)):
    """Affiche une image dans une fenêtre matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def draw_boxes_on_gray(gray, boxes, color_bgr=(0, 0, 255), thickness=3):
    """Dessine des boîtes sur une image en niveaux de gris (retourne BGR)."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)
        label = str(i)
        cv2.putText(out, label, (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
    return out




def run_pipeline_visual(image_path):
    print(f"\n{'='*60}")
    print(f"  Image source : {image_path}")
    print(f"{'='*60}\n")

    # ── Étape 0 : lecture ──────────────────────────────────────
    image_bgr_orig = cv2.imread(image_path)
    if image_bgr_orig is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        sys.exit(1)

    h0, w0 = image_bgr_orig.shape[:2]
    print(f"[0] Image originale       : {w0} × {h0} px")
    show("Étape 0 — Image originale", bgr_to_rgb(image_bgr_orig))

    # ── Étape 1 : resize ──────────────────────────────────────
    image_bgr = resize_if_needed(image_bgr_orig, max_width=2200)
    h1, w1 = image_bgr.shape[:2]
    print(f"[1] Après resize          : {w1} × {h1} px  (max_width=2200)")
    show("Étape 1 — Après resize (max 2200 px de large)", bgr_to_rgb(image_bgr))

    # ── Étape 2 : conversion en niveaux de gris ───────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"[2] Conversion en gris    : {gray.shape[1]} × {gray.shape[0]} px")
    show("Étape 2 — Niveaux de gris (BGR → GRAY)", gray, cmap='gray')

    # ── Étape 3 : deskew ──────────────────────────────────────
    if ENABLE_DESKEW:
        gray_deskewed = deskew_image(gray)
        print(f"[3] Deskew activé         : correction de l'inclinaison")
    else:
        gray_deskewed = gray
        print(f"[3] Deskew désactivé      : image inchangée")
    show("Étape 3 — Deskew (redressement)", gray_deskewed, cmap='gray')

    # ── Étape 4 : crop de la région utile ─────────────────────
    useful_gray, (offset_x, offset_y) = crop_useful_region(gray_deskewed)
    print(f"[4] Crop région utile     : offset=({offset_x}, {offset_y})  "
          f"taille={useful_gray.shape[1]} × {useful_gray.shape[0]} px")
    show("Étape 4 — Crop région utile (marges retirées)", useful_gray, cmap='gray')

    # ── Étape 5 : binarisation ────────────────────────────────
    bw = binarize(useful_gray)
    print(f"[5] Binarisation          : seuillage adaptatif + ouverture morpho")
    show("Étape 5 — Binarisation (seuillage adaptatif)", bw, cmap='gray')

    # ── Étape 6 : suppression des petites composantes ─────────
    bw_clean = remove_small_components(bw, min_area=20)
    removed = int(bw.sum() / 255) - int(bw_clean.sum() / 255)
    print(f"[6] Suppression bruit     : ~{removed} pixels supprimés (composantes < 20 px²)")
    show("Étape 6 — Suppression du bruit (composantes < 20 px²)", bw_clean, cmap='gray')

    # ── Étape 7 : projection verticale + détection colonnes ───
    columns, v_proj, v_smooth = detect_text_columns(bw_clean)
    print(f"[7] Détection colonnes    : {len(columns)} colonne(s) trouvée(s) → {columns}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle("Étape 7 — Détection des colonnes de texte", fontsize=14, fontweight='bold')

    # Image avec colonnes colorées
    col_vis = cv2.cvtColor(bw_clean, cv2.COLOR_GRAY2BGR)
    colors = [(255, 100, 0), (0, 180, 255), (0, 255, 120), (200, 0, 200)]
    for idx, (cx1, cx2) in enumerate(columns):
        c = colors[idx % len(colors)]
        overlay = col_vis.copy()
        cv2.rectangle(overlay, (cx1, 0), (cx2, col_vis.shape[0]), c, -1)
        cv2.addWeighted(overlay, 0.25, col_vis, 0.75, 0, col_vis)
        cv2.rectangle(col_vis, (cx1, 0), (cx2, col_vis.shape[0]), c, 3)

    axes[0].imshow(bgr_to_rgb(col_vis))
    axes[0].set_title(f"{len(columns)} colonne(s) détectée(s)", fontsize=11)
    axes[0].axis('off')

    # Projection verticale
    axes[1].plot(v_proj, np.arange(len(v_proj)), color='steelblue', linewidth=0.8, label='projection')
    axes[1].plot(v_smooth, np.arange(len(v_smooth)), color='red', linewidth=1.5, label='signal lissé')
    axes[1].invert_yaxis()
    axes[1].set_title("Projection\nverticale", fontsize=11)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ── Étape 8 : détection des paragraphes par colonne ───────
    all_raw_boxes = []
    for col_x1, col_x2 in columns:
        para_boxes, h_proj, h_smooth = detect_paragraphs_in_column(bw_clean, col_x1, col_x2)
        all_raw_boxes.extend(para_boxes)

    print(f"[8] Détection paragraphes : {len(all_raw_boxes)} boîte(s) brute(s)")
    vis8 = draw_boxes_on_gray(useful_gray, all_raw_boxes)
    show("Étape 8 — Paragraphes bruts détectés par colonne", bgr_to_rgb(vis8))

    # ── Étape 9 : re-découpage secondaire ─────────────────────
    all_split_boxes = []
    for col_x1, col_x2 in columns:
        col_boxes, _, _ = detect_paragraphs_in_column(bw_clean, col_x1, col_x2)
        split = secondary_split_boxes(bw_clean, col_boxes)
        all_split_boxes.extend(split)

    print(f"[9] Split secondaire      : {len(all_raw_boxes)} → {len(all_split_boxes)} boîtes")
    vis9 = draw_boxes_on_gray(useful_gray, all_split_boxes)
    show("Étape 9 — Après re-découpage des boîtes trop hautes", bgr_to_rgb(vis9))

    # ── Étape 10 : ajustement horizontal des boîtes ───────────
    adjusted_boxes = []
    for col_x1, col_x2 in columns:
        col_boxes, _, _ = detect_paragraphs_in_column(bw_clean, col_x1, col_x2)
        col_boxes = secondary_split_boxes(bw_clean, col_boxes)
        for box in col_boxes:
            adj = adjust_box_to_column(bw_clean, box, col_x1, col_x2)
            adjusted_boxes.append(adj)

    print(f"[10] Ajustement horizontal : {len(adjusted_boxes)} boîtes ajustées")
    vis10 = draw_boxes_on_gray(useful_gray, adjusted_boxes)
    show("Étape 10 — Après ajustement horizontal (snap aux colonnes)", bgr_to_rgb(vis10))

    # ── Étape 11 : fusion des boîtes proches ──────────────────
    merged_boxes = merge_close_boxes(adjusted_boxes, y_gap=MERGE_Y_GAP, x_tol=MERGE_X_TOL)
    merged_boxes = sort_reading_order(merged_boxes)
    print(f"[11] Fusion               : {len(adjusted_boxes)} → {len(merged_boxes)} boîtes")
    vis11 = draw_boxes_on_gray(useful_gray, merged_boxes)
    show("Étape 11 — Après fusion des boîtes proches", bgr_to_rgb(vis11))

    # ── Étape 12 : restauration des coordonnées originales ────
    restored_boxes = [
        (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        for x1, y1, x2, y2 in merged_boxes
    ]
    print(f"[12] Restauration offset  : offset=({offset_x}, {offset_y}) réappliqué")

    # ── Étape 13 : résultat final annoté ──────────────────────
    final_annotated = draw_boxes(image_bgr, restored_boxes)
    print(f"[13] Résultat final       : {len(restored_boxes)} acte(s) détecté(s)")
    show(f"Étape 13 — Résultat final : {len(restored_boxes)} acte(s) détecté(s) sur l'image originale",
         bgr_to_rgb(final_annotated))

    # ── Récapitulatif : tous les crops ────────────────────────
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    n = len(restored_boxes)
    if n > 0:
        cols_grid = min(4, n)
        rows_grid = (n + cols_grid - 1) // cols_grid
        fig, axes = plt.subplots(rows_grid, cols_grid,
                                 figsize=(cols_grid * 4, rows_grid * 4))
        fig.suptitle(f"Récapitulatif — {n} acte(s) découpé(s) (crops envoyés à l'OCR)",
                     fontsize=14, fontweight='bold')

        if n == 1:
            axes = np.array([[axes]])
        elif rows_grid == 1:
            axes = axes.reshape(1, -1)

        for i, (x1, y1, x2, y2) in enumerate(restored_boxes):
            r, c = divmod(i, cols_grid)
            h_img, w_img = gray_full.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w_img, x2), min(h_img, y2)
            crop = gray_full[y1c:y2c, x1c:x2c]
            axes[r][c].imshow(crop, cmap='gray')
            axes[r][c].set_title(f"Acte {i+1}", fontsize=10)
            axes[r][c].axis('off')

        # Masquer les cases vides
        for j in range(n, rows_grid * cols_grid):
            r, c = divmod(j, cols_grid)
            axes[r][c].axis('off')

        plt.tight_layout()
        plt.show()

    print(f"\n{'='*60}")
    print(f"  Pipeline terminé — {n} acte(s) détecté(s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    DEFAULT_IMAGE = r"Archives\Archives_1937\Archives_1937_01.JPG"

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, DEFAULT_IMAGE)
        print(f"Aucune image fournie — utilisation de l'image par défaut :\n  {path}")

    run_pipeline_visual(path)
