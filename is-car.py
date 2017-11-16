import cv2
import numpy as np
import os
import sys

from utils import loadBlurImg, preprocessData

def load_img(img_path):
    x = []
    if not os.path.exists(img_path):
        print("Image not found!")
        exit(1)
    
    img = loadBlurImg(img_path, (64, 64))

    if img is None:
        print("Invalid image")
        exit(1)
    
    x.append(img)
    y = np.array(x)
    return y

def main(img_path):

    if not os.path.exists('car.h5'):
        print("Model not found")
        exit(1)

    img = load_img(img_path)
    img = preprocessData(img)

    from keras.models import load_model
    model = load_model('car.h5')

    predictions = model.predict_classes(img)
    
    if predictions[0] == 0:
        print("Is car")
    else:
        print("Not car")

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Insufficient arguments")
        exit(1)
    
    main(sys.argv[1])