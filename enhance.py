import cv2
import numpy as np

def auto_enhance(img):
    # CLAHE di channel LAB (kecerahan)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)

    lab_clahe = cv2.merge([L2, A, B])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Sedikit RGB Boost (ringan + aman)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.10, beta=5)

    return enhanced
