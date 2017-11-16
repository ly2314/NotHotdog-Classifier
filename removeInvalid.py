import os
import cv2
import numpy as np
from PIL import Image  

def is_image_ok(fn):
    try:
        Image.open(fn)
        img = cv2.imread(fn)
        if img is None:
            return False
        return True
    except:
        return False

def remove(dirPaths):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            current_image_path = str(dirPath)+'/'+str(img)
            if is_image_ok(current_image_path) == False:
                os.remove(current_image_path)
                continue
            for invalid in os.listdir('invalid'):
                try:
                    invalid = cv2.imread('invalid/'+str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).any()):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))

    
paths = ['images/not-car/animals', \
        'images/not-car/people', \
        'images/not-car/food', \
        'images/not-car/plants', \
        'images/car/cars', \
        'images/not-car/structures', \
        'images/car/trucks', \
        'images/car/mortorcycles', \
        'images/not-car/airplane', \
        'images/not-car/ships', \
        'images/car/jeeps', \
        'images/car/old-cars']

remove(paths)