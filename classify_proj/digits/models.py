from django.db import models
from django.conf import settings
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from tensorflow.keras.models import load_model

import tensorflow as tf
import cv2
import os
import numpy as np

# Create your models here.
class Digit(models.Model):
    image = models.ImageField(upload_to='images')
    result = models.CharField(max_length=2, blank=True)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):
        print(self.image)
        img = Image.open(self.image)
        img_array = image.img_to_array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        prepar_img = np.expand_dims(img_array, axis=2)
        img_array = np.expand_dims(prepar_img, axis=0)
        print(img_array.shape)
        print('BASE_DIR', settings.BASE_DIR)
        try:
            file_model = os.path.join(settings.BASE_DIR, 'CNN_convnet.h5')
            model = load_model(file_model)
            pred = np.argmax(model.predict(img_array))
            self.result = str(pred)
            print(f'Digito Classificado como {pred}')

        except:
            print('Falha para classificar : ')
            self.result = 'Falha na predição'

        #file_model = ("/home/dev/mnist/classify/src/classify_proj/classify_proj/neural_cnn/CNN_convnet.h5")
        #print('file model: ', file_model)
        #model = load_model(file_model)
        #print(model)
        #pred = np.argmax(model.predict(img_array))
        #self.result = str(pred)

        return super().save(*args, **kwargs)