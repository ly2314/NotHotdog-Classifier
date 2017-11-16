import urllib
import urllib.request
import cv2
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool 
import itertools
from PIL import Image

pic_num = 1

def store_raw_images(paths, links):
    global pic_num
    for link, path in zip(links, paths):
        if not os.path.exists(path):
            os.makedirs(path)
        image_urls = str(urllib.request.urlopen('http://image-net.org/api/text/imagenet.synset.geturls?wnid=' + link).read())
        
        pool = ThreadPool(32)
        pool.starmap(loadImage, zip(itertools.repeat(path),image_urls.split('\\n'),itertools.count(pic_num))) 
        pool.close() 
        pool.join()
                    
def loadImage(path,link, counter):
    global pic_num
    if pic_num < counter:
        pic_num = counter+1;
    try:
        import socket
        socket.setdefaulttimeout(5)
        urllib.request.urlretrieve(link, path+"/"+str(counter)+".jpg")
        img = cv2.imread(path+"/"+str(counter)+".jpg")             
        if img is not None:
            cv2.imwrite(path+"/"+str(counter)+".jpg",img)
            print(counter)

    except Exception as e:
        print(str(e))  

def is_image_ok(fn):
    try:
        Image.open(fn)
        img = cv2.imread(fn)
        if img is None:
            return False
        return True
    except:
        return False
    
def removeInvalid(dirPaths):
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
  
def main():
    ids = [ 
            'n00015388', \
            'n00007846', \
            'n00021265', \
            'n00017222', \
            'n02958343', \
            'n04341686', \
            'n04490091', \
            'n03790512'
             ]
    
    paths = ['images/not-car/animals', \
             'images/not-car/people', \
             'images/not-car/food', \
             'images/not-car/plants', \
             'images/car/cars', \
             'images/not-car/structures', \
             'images/car/trucks', \
             'images/car/mortorcycles']

    store_raw_images(paths, ids)
    removeInvalid(paths)


if __name__ == "__main__":

    main()
