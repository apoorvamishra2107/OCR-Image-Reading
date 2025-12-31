import cv2
import pytesseract

# Path to your image
image_path = 'ocr img.jpg'

# Read the image using OpenCV
img = cv2.imread(image_path)

# Preprocessing functions
def get_grayscale(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image): 
    return cv2.medianBlur(image, 5)

def thresholding(image): 
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# OCR function
def ocr_image(img):
    text = pytesseract.image_to_string(img)
    return text

# Apply preprocessing
img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)

# Run OCR
print(ocr_image(img))




