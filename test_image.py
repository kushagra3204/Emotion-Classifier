from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

model = load_model(os.path.join('models','happysadmodel_enhanced.h5'))
img = cv2.imread('sadimg.jpg')
resize = tf.image.resize(img,(256,256))
yhat = model.predict(np.expand_dims(resize/255,0))
if yhat>0.5:
    print(f'Sad image (yhat: {yhat})')
else:
    print(f'Happy image (yhat: {yhat})')