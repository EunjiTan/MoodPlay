import cv2
import numpy as np
from PIL import Image
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def convert_to_grayscale(image_path, output_path):
    img = Image.open(image_path).convert('L')
    img.save(output_path)
    return output_path

def generate_segmentation(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((3,3), np.uint8)
    seg = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(output_path, seg)
    return output_path

def colorize_placeholder(gray_path, seg_path, output_path):
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colored[:,:,0] = np.clip(colored[:,:,0] * 0.8, 0, 255)
    colored[:,:,1] = np.clip(colored[:,:,1] * 0.9, 0, 255)
    colored[:,:,2] = np.clip(colored[:,:,2] * 1.0, 0, 255)
    cv2.imwrite(output_path, colored)
    return output_path