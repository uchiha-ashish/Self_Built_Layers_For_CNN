import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

train_dir = "/content/dataset/train"
categories = os.listdir(train_dir)
for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img))  
        plt.imshow(img_array, cmap='gray')  
        plt.show() 
        break  
    break


training_data = []
IMG_SIZE=244
def create_training_data():
    for category in categories:  
        path = os.path.join(train_dir,category)  
        class_num = CATEGORIES.index(category)  
        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                training_data.append([img_array, class_num])  
            except Exception as e:  
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))
create_training_data()




test_dir = "/content/dataset/test"
categories = os.listdir(test_dir)
for category in categories:  
    path = os.path.join(test_dir,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img))  
        plt.imshow(img_array)  
        plt.show() 
        break  
    break


test_data = []
IMG_SIZE=224
def create_test_data():
    for category in categories:  
        path = os.path.join(test_dir,category)  
        class_num = categories.index(category)  
        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                test_data.append([img_array, class_num])  
            except Exception as e:  
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))
create_test_data()


import random
random.shuffle(training_data)
random.shuffle(test_data)
