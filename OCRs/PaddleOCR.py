import cv2
import re
import os
import time
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

# CONFIG
BASE_PATH    = "Archives/Archives_1937"
USE_GPU      = False  # Mettre True si tu as un GPU CUDA disponible
CACHE_DIR    = os.path.join(BASE_PATH, "ocr_output")   # dossier cache .txt
FORCE_RERUN  = False  # Mettre True pour forcer le re-traitement même si cache existe

# CACHE
def get_cache_path(filename: str) -> str:
    """Retourne le chemin du fichier cache .txt correspondant à une image."""
    name = os.path.splitext(filename)[0]  # retire l'extension .JPG / .jpg
    return os.path.join(CACHE_DIR, f"{name}.txt")

def load_from_cache(cache_path: str) -> str | None:
    """Lit le cache si le fichier existe, sinon retourne None."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def save_to_cache(cache_path: str, text: str):
    """Sauvegarde le texte OCR dans le cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

# PRÉTRAITEMENT
def preprocess_image(image_path: str):
    """Charge et binarise l'image via seuillage Otsu. Retourne GRAY (1 canal) comme l'original."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh  # GRAY — la conversion BGR est faite en aval

# OCR + SÉPARATION GAUCHE / DROITE
def extract_text_left_right(image_bgr, ocr_engine):
    """
    Lance l'OCR UNE seule fois sur l'image entière,
    puis trie les mots détectés selon leur position X
    pour séparer page gauche / page droite.
    """
    # Si l'image est GRAY (1 canal), on la convertit en BGR pour PaddleOCR
    if len(image_bgr.shape) == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    middle = w * 0.5
    result = ocr_engine.predict(image_bgr)
    left_words, right_words   = [], []
    left_conf,  right_conf    = {}, {}

    if not result:
        return "Aucun texte détecté.", {}

    for res in result:
        if res is None:
            continue

        rec_boxes  = res.get('rec_boxes',  [])
        rec_texts  = res.get('rec_texts',  [])
        rec_scores = res.get('rec_scores', [])

        for box, text, conf in zip(rec_boxes, rec_texts, rec_scores):
            text = str(text).strip()
            if not text:
                continue

            # box = [x_min, y_min, x_max, y_max]
            x_center = (box[0] + box[2]) / 2
            entry    = f"[{conf:.2f}] {text}"

            if x_center < middle:
                left_words.append(entry)
                left_conf[text]  = round(float(conf), 3)
            else:
                right_words.append(entry)
                right_conf[text] = round(float(conf), 3)

    all_confidences = {**left_conf, **right_conf}

    full_text = (
        "===== PAGE GAUCHE =====\n"
        + "\n".join(left_words)
        + "\n\n\n===== PAGE DROITE =====\n"
        + "\n".join(right_words)
    )
    return full_text, all_confidences

# NETTOYAGE
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# STATISTIQUES DE CONFIANCE
def print_confidence_stats(confidences: dict):
    if not confidences:
        print("Aucune donnée de confiance.")
        return
    values  = list(confidences.values())
    moyenne = sum(values) / len(values)
    mini    = min(values)
    maxi    = max(values)
    faibles = [t for t, c in confidences.items() if c < 0.7]

    print(f"  Moyenne  : {round(moyenne, 3)}")
    print(f"  Min      : {round(mini, 3)}")
    print(f"  Max      : {round(maxi, 3)}")
    if faibles:
        print(f"  ⚠ Mots < 0.70 : {faibles}")

# PIPELINE PRINCIPAL
def main():
    # Collecte des images
    images = sorted([f for f in os.listdir(BASE_PATH) if f.lower().endswith(".jpg")])
    if not images:
        print(f"Aucune image JPG trouvée dans : {BASE_PATH}")
        return
    print(f"{len(images)} image(s) trouvée(s) dans {BASE_PATH}\n")

    # ── Vérifie si toutes les images sont déjà en cache ──
    all_cached = all(
        os.path.exists(get_cache_path(f)) for f in images
    ) and not FORCE_RERUN

    # ── Chargement du modèle (uniquement si nécessaire) ──
    ocr = None
    if not all_cached:
        print("Chargement PaddleOCR (une seule fois)...")
        t0 = time.time()
        ocr = PaddleOCR(
            use_textline_orientation=False,
            lang="fr",
            device="gpu" if USE_GPU else "cpu",
        )
        print(f"Modèle chargé en {time.time() - t0:.1f}s\n")
    else:
        print("✅ Toutes les images sont en cache — OCR ignoré.\n")

    # ── Boucle sur toutes les images ─────────────
    for i, filename in enumerate(images, 1):
        image_path  = os.path.join(BASE_PATH, filename)
        cache_path  = get_cache_path(filename)
        print(f"[{i}/{len(images)}] {filename}")

        # ── Lecture depuis le cache ──
        if not FORCE_RERUN:
            cached_text = load_from_cache(cache_path)
            if cached_text is not None:
                print("  ⚡ Depuis le cache")
                print("\n===== TEXTE =====")
                print(cached_text)
                print("─" * 50)
                continue

        # ── OCR complet ──
        t1 = time.time()
        try:
            image_bgr         = preprocess_image(image_path)
            text, confidences = extract_text_left_right(image_bgr, ocr)
            text              = clean_text(text)
        except Exception as e:
            print(f"  ❌ Erreur : {e}")
            continue

        print(f"  ✅ OCR terminé en {time.time() - t1:.1f}s")

        # Affichage
        print("\n===== TEXTE =====")
        print(text)
        print("\n===== CONFIANCE =====")
        print_confidence_stats(confidences)
        print()

        # Sauvegarde dans le cache
        save_to_cache(cache_path, text)
        print(f"  💾 Mis en cache : {cache_path}")

        print("─" * 50)

if __name__ == "__main__":
    main()