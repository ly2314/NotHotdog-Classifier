import numpy as np
import cv2
from skimage import exposure

def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    
    images = 0.2989*images[:,:,:,0] + 0.5870*images[:,:,:,1] + 0.1140*images[:,:,:,2]
    return images

def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)
    
    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])
    
    images = images.reshape(images.shape + (1,)) 
    return images

def normalizeImages2(images):
    for i in range(images.shape[0]):
        cv2.normalize(images[i],images[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # if convert to gray scale use this after
    print("images has shape before", images.shape)
    #images = images.reshape(images.shape + (1,)) 
    #print("images has shape after", images.shape)
    return images

def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)

def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols,rows))

def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    if img is None:
        return None
    #angle = np.random.randint(-180, 180)
    #img = rotateImage(img, angle)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    #showImage(img)
    return img