import cv2
import pytesseract
import pandas as pd
import re
from PIL import Image
import matplotlib.pyplot as plt
import os


base_path = "Archives/Archives_1937"

images = sorted([f for f in os.listdir(base_path) if f.endswith(".JPG")])
first_image_path = os.path.join(base_path, images[0])
print("Image utilisée :", first_image_path)


def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# trouver automatiquement où se trouve le texte dans l’image
def detect_text_regions(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > 100 and h > 50:
            boxes.append((x, y, w, h))

    # ordre de lecture
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


# 🔥 OCR avec texte + dictionnaire de confiance
def ocr_with_confidence(roi, config):
    data = pytesseract.image_to_data(
        roi,
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    # nettoyage
    data = data.dropna(subset=['text'])
    data['text'] = data['text'].astype(str)
    data = data[data['text'].str.strip() != ""]
    data = data[data['conf'] != -1]

    words = []
    confidences = {}

    for _, row in data.iterrows():
        word = row['text'].strip()
        conf = int(row['conf'])

        words.append(word)
        confidences[word] = conf

    text = " ".join(words)

    return text, confidences


def extract_text_left_right(image):
    h, w = image.shape
    middle = int(w * 0.5)

    left_img = image[:, :middle]
    right_img = image[:, middle:]

    config = "--oem 3 --psm 6 -l fra"

    left_boxes = detect_text_regions(left_img)
    right_boxes = detect_text_regions(right_img)

    left_texts = []
    right_texts = []

    # 🔥 dictionnaire global
    all_confidences = {}

    # GAUCHE
    for (x, y, bw, bh) in left_boxes:
        roi = left_img[y:y+bh, x:x+bw]

        text, conf_dict = ocr_with_confidence(roi, config)

        left_texts.append(text)
        all_confidences.update(conf_dict)

    # DROITE
    for (x, y, bw, bh) in right_boxes:
        roi = right_img[y:y+bh, x:x+bw]

        text, conf_dict = ocr_with_confidence(roi, config)

        right_texts.append(text)
        all_confidences.update(conf_dict)

    left_text = "\n".join(left_texts)
    right_text = "\n".join(right_texts)

    full_text = (
        "===== PAGE GAUCHE =====\n"
        + left_text +
        "\n\n\n"
        + "===== PAGE DROITE =====\n"
        + right_text
    )

    return full_text, all_confidences


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# pipeline
processed = preprocess_image_for_ocr(first_image_path)

text, confidences = extract_text_left_right(processed)
text = clean_text(text)

print("===== TEXTE =====")
print(text)

print("\n===== DICTIONNAIRE CONFIANCE =====")
print(confidences)