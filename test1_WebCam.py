from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

model = load_model(os.path.join('models','happysadmodel1.h5'))

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize image
    img = frame.copy()
    # img = cv2.imread('madelportrait.jpg')
    resize = tf.image.resize(img,(256,256))
    yhat = model.predict(np.expand_dims(resize/255,0))
    if yhat>0.5:
        print(f'Sad image (yhat: {yhat})')
    else:
        print(f'Happy image (yhat: {yhat})')
    
    cv2.imshow('Happy & Sad Classifier', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()