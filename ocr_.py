from PIL import Image
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pschaefe\AppData\Local\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\jmaniar\AppData\Local\Tesseract-OCR\tesseract.exe'
def ocr(filename):
    im = Image.open(filename)
    text = pytesseract.image_to_string(im, lang = 'eng')
    return text