import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
import tensorflow.keras.losses
import tensorflow.keras.metrics
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2


def load_images(num_img): #import all the images
    num_test=400
    Dossier_Complet= os.listdir('./train/')
    Dossier_Complet= [re.split('_|\.', i) for i in Dossier_Complet]

    original=sorted([i[:2] for i in Dossier_Complet[:num_img] if len(i)==3])
    original=['_'.join(i) for i in original]
    
    set_train = [imread('./train/'+str(i)+'.tif') for i in original[num_test:]]
    set_mask = [imread('./train/'+str(i)+'_mask.tif') for i in original[num_test:]]
    set_test = [imread('./train/'+str(i)+'.tif') for i in original[:num_test]]
    set_test_mask = [imread('./train/'+str(i)+'_mask.tif') for i in original[:num_test]]
    return set_train, set_mask, set_test, set_test_mask

def preprocessing(imgs, rows, cols, normalized=False, mask=False): 
    """
    preprocessing to add one channels and to normalize
    """

    img_1 = np.asarray([resize(i, (rows, cols, 1), preserve_range=True) for i in imgs])
   
    if normalized:
        img_1= (img_1/255.)#-0.5
    if mask: 
        img_1/=255.
        
    return img_1


def dice_coef(y_true, y_pred, smooth=1): 
    """
    to evaluate the model
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred,  smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true)+K.sum(y_pred)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def dice_coef1(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred): 
    """
    to make the average with the dice coef of the 2 classes (sofmax)
    """
    dice=0
    for index in range(2):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/2 # taking average

def mask2channels(imgs):
    """
    to give 2 channels for the mask and use the softmax
    """
    img_1 = np.ndarray((np.asarray(imgs).shape[0],np.asarray(imgs).shape[1], np.asarray(imgs).shape[2], 2), dtype=np.float)

    img_1[:,:,:,1]= [1-i[:,:,0] for i in imgs]
    img_1[:,:,:,0]= 1-img_1[:,:,:,1]
    return img_1

def get_unet(img_rows, img_cols):

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer= Adam(lr=1e-3), loss= [tensorflow.keras.losses.binary_crossentropy], metrics=[dice_coef_multilabel])
    
    return model


def auto_canny(image, sigma=0.33):
    
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged