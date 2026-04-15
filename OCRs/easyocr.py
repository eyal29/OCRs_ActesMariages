import easyocr
import cv2

IMAGE_PATH = "Archives/Archives_1937/Archives_1937_01.JPG"

# 1. CHARGEMENT IMAGE
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Image introuvable")
    exit()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. DÉTECTION RÉGIONS
def detect_text_regions(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 10:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[1])
    return boxes

# 3. OCR PAR RÉGION (reader injecté depuis l'extérieur)
def run_ocr(image, reader):
    boxes = detect_text_regions(image)
    texts = []
    confidences = []
    for (x, y, w, h) in boxes:
        roi = image[y:y+h, x:x+w]
        results = reader.readtext(roi, text_threshold=0.4, low_text=0.3)
        for (_, text, conf) in results:
            if not text:
                continue
            texts.append(f"[{conf:.2f}] {text}")
            try:
                confidences.append(float(conf))
            except:
                pass
    full_text = "\n".join(texts)
    conf_moy = sum(confidences) / len(confidences) if confidences else 0
    return full_text, conf_moy

# 4. PIPELINE PRINCIPAL (script terminal autonome)
if __name__ == "__main__":
    h, w = img.shape
    left  = img[:, :w//2]
    right = img[:, w//2:]

    # Reader chargé UNE seule fois
    print("Chargement EasyOCR...")
    reader = easyocr.Reader(['fr'], gpu=False)

    print("OCR en cours...")
    left_text,  left_conf  = run_ocr(left,  reader)
    right_text, right_conf = run_ocr(right, reader)

    final_text = (
        "===== PAGE GAUCHE =====\n\n"
        + left_text +
        "\n\n===== PAGE DROITE =====\n\n"
        + right_text
    )

    conf_global = (left_conf + right_conf) / 2

    print("\n===== TEXTE =====\n")
    print(final_text)
    print("\n===== CONFIANCE =====")
    print(f"Gauche : {round(left_conf,  3)}")
    print(f"Droite : {round(right_conf, 3)}")
    print(f"Global : {round(conf_global, 3)}")