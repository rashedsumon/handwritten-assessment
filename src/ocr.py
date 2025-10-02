# src/ocr.py
import pytesseract
from PIL import Image
import easyocr
import os
import tempfile
from pdf2image import convert_from_path

# Initialize EasyOCR (slow first init)
_reader = None
def _get_easyocr_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available
    return _reader

def _image_to_text_tesseract(img_path):
    try:
        pil = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(pil)
        return text
    except Exception as e:
        return f"[tesseract_error]{e}"

def _image_to_text_easyocr(img_path):
    try:
        reader = _get_easyocr_reader()
        res = reader.readtext(img_path, detail=0, paragraph=True)
        return "\n".join(res)
    except Exception as e:
        return f"[easyocr_error]{e}"

def extract_text_from_image(path_or_image):
    # if PDF, convert first page
    if str(path_or_image).lower().endswith(".pdf"):
        pages = convert_from_path(path_or_image, first_page=1, last_page=1)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pages[0].save(tmp.name, format="PNG")
            text = _image_to_text_easyocr(tmp.name)
            os.unlink(tmp.name)
            return text
    # else try easyocr first, fallback to tesseract
    text = _image_to_text_easyocr(path_or_image)
    if text.startswith("[easyocr_error]") or len(text.strip()) < 10:
        text = _image_to_text_tesseract(path_or_image)
    return text

def extract_text_from_image_batch(paths):
    results = {}
    for p in paths:
        fname = os.path.basename(p)
        results[fname] = extract_text_from_image(p)
    return results
