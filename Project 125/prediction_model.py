import cv2
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
  ssl._create_default_https_context=ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes = len(classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scale = X_train/255
X_test_scale = X_test/255

lr = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scale, y_train)

def get_prediction(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert('L')
    img_rz = img_bw.resize((22, 30), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_rz, pixel_filter)
    img_scaled = np.clip(img_rz-min_pixel, 0, 255)
    max_pixel = np.max(img_rz)
    img_scaled = np.asarray(img_scaled)/max_pixel
    test_sample = np.array(img_scaled).reshape(1, 784)
    test_prediction = lr.predict(test_sample)
    # accuracy = accuracy_score(test_sample, test_prediction)
    # print(accuracy)
    return test_prediction[0]