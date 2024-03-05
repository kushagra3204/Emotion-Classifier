from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
import threading
import time
import cv2
import os


####### GLOBAL VARIABLES #######
DATA_DIR = 'data'
IMG_EXT = ['jpeg','jpg','bmp','png']
MODEL = Sequential()
PRE = Precision()
RE = Recall()
ACC = BinaryAccuracy()


####### ENABLE GPU #######
def enable_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)


####### REMOVE UNNECESSARY PICTURES #######
def remove_redundant_pics():
    for datadir in os.listdir(DATA_DIR):
        for image in os.listdir(os.path.join(DATA_DIR,datadir)):
            img_path = os.path.join(DATA_DIR,datadir,image)
            try:
                with Image.open(img_path) as img:
                    tip = img.format.lower()
                if tip not in IMG_EXT:
                    print("Not desired image format ",format(img_path),tip)
                    os.remove(img_path)
            except Exception as e:
                print(f"Issue with image {format(img_path)}")


def show_batch_data(batch):
    fig,ax = plt.subplots(ncols=4,figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
    plt.show()


def plot_graph(a,a_name,b,b_name,title):
    fig = plt.figure()
    plt.plot(a,color='teal',label=a_name)
    plt.plot(b,color='orange',label=b_name)
    fig.suptitle(title,fontsize=20)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join('models','model_analysis',f'{a_name}_VS_{b_name}_trial.png'))
    plt.show()

    
####### BUILDING DEEP LEARNING MODEL #######
def build_model():
    MODEL.add(Conv2D(32,(3,3),1,activation='relu',input_shape=(256,256,3)))
    MODEL.add(MaxPooling2D())
    
    MODEL.add(Conv2D(32,(3,3),1,activation='relu'))
    MODEL.add(MaxPooling2D())
    
    MODEL.add(Conv2D(16,(3,3),1,activation='relu'))
    MODEL.add(MaxPooling2D())
    
    MODEL.add(Flatten())
    
    MODEL.add(Dense(256,activation='relu'))
    MODEL.add(Dense(1,activation='sigmoid'))
    
    MODEL.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
    MODEL.summary()


def evaluate_performance(test):
    for batch in test.as_numpy_iterator():
        x,y = batch
        yhat = MODEL.predict(x)
        PRE.update_state(y,yhat)
        RE.update_state(y,yhat)
        ACC.update_state(y,yhat)
    print(f'Precision: {PRE.result().numpy()}, Recall: {RE.result().numpy()}, Accuracy: {ACC.result().numpy()}')
    

####### TRAIN & PREPROCESS DATA #######
def preprocess_data_and_train(data):
    data = data.map(lambda x,y: (x/255,y))
    scaled_iterator = data.as_numpy_iterator()
    
    show_batch_data(scaled_iterator.next())
    
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2) + 1
    test_size = int(len(data)*.1) + 1
    
    train = data.take(train_size)
    
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    # Train the data
    hist = MODEL.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])
    
    plot_graph(hist.history['loss'],'loss',hist.history['val_loss'],'val_loss','Loss')
    plot_graph(hist.history['accuracy'],'accuracy',hist.history['val_accuracy'],'val_accuracy','Accuracy')
    
    evaluate_performance(test)
    

####### MAIN #######
if __name__ == "__main__":
    
    build_model()
    
    remove_redundant_pics()
    
    data = tf.keras.utils.image_dataset_from_directory(DATA_DIR)
    
    preprocess_data_and_train(data)
    
    MODEL.save(os.path.join('models','happysadmodel_trial.h5'))