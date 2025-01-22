import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.applications import VGG16

from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

import numpy as np
import time
import os

root = os.path.dirname(os.path.abspath(__file__)) 

class vgg16_SVHN():
    def __init__(self) -> None:
        self.model_name = 'vgg16_SVHN'
        self.dataset = 'SVHN'
        self.original_epochs = 10
        self.load_dataset()
        self.batch_size = 128
    
    def load_dataset(self):
        # Load Fashion-MNIST dataset
        
        train_raw = loadmat(os.path.join(root,'..','datasets','SVHN', 'train_32x32.mat'))
        test_raw = loadmat(os.path.join(root,'..','datasets','SVHN', 'test_32x32.mat'))
        
        x_train = np.array(train_raw['X'])
        x_test = np.array(test_raw['X'])
        y_train = train_raw['y']
        y_test = test_raw['y']
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        
        x_test= x_test.reshape (-1,32,32,3)
        x_train= x_train.reshape (-1,32,32,3)
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.fit_transform(y_test)
        

        # y_train = tf.keras.utils.to_categorical(y_train, 10)
        # y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (x_test, y_test)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
        
        
        
    def fit_model(self, epochs = 1):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)


        batch_size = self.batch_size
        epochs = epochs
        
        # VGG 16  (https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16)
        # paper: https://arxiv.org/abs/1409.1556
        # Create an instance of VGG16 with pre-trained weights
        model = VGG16(
            include_top=True,
            weights = None,
            input_shape=(32, 32, 3),  # Adjust the input shape according to your dataset
            classes=10,  # Set the number of classes in your dataset
            classifier_activation="softmax",
        )
                
                

        optimizer = keras.optimizers.Adadelta(learning_rate = 1.0)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        # optimizer = Adam(learning_rate=0.01) # default of adam
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # opt = Adam(learning_rate=0.001) # default of adam
        # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        if not os.path.exists(os.path.join(root,'output', 'epoch_models', self.model_name)):
            os.makedirs(os.path.join(root,'output', 'epoch_models', self.model_name))
            
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epochs}.h5')
        model.save(model_path)
        
        return model
    
    def train_an_epoch_more(self,model,epoch_number):
        loaded_model = self.load_model_of_each_epoch(epoch_number)
        if loaded_model:
            print('+++++++++++++')
            return loaded_model
        
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = self.batch_size
        print('fitting...')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epoch_number}.h5')
        model.save(model_path)
        
        return model


    def load_original_model(self):
        original_model_path = os.path.join('models',self.model_name,'original_model.h5')
        print(original_model_path)
        
        if os.path.exists(original_model_path):
            model = load_model(original_model_path,compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        else:
            print('start training')
            model = self.fit_model(epochs = self.original_epochs)
            model.save(original_model_path)
            return model
    
    def load_model_of_each_epoch(self, epoch = 0):
        if epoch:
            if epoch == self.original_epochs:
                model = self.load_original_model()
                return model
            
            model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epoch}.h5')
            if os.path.isfile(model_path):
                model = load_model(model_path, compile=False)
                model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
                return model
        else:
            return None
        
    def execute_model(self, model):
        score = model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])

        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])