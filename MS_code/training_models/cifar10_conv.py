import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import time
import os

root = os.path.dirname(os.path.abspath(__file__)) 

class cifar10_conv():
    def __init__(self) -> None:
        self.model_name = 'cifar10' # conv-8 layer (vgg-8)
        self.dataset = 'cifar10'
        self.original_epochs = 20
        self.load_dataset()
        self.batch_size = 128
    
    def load_dataset(self):
        # Load cifar10 dataset
        CLIP_MIN = -0.5
        CLIP_MAX = 0.5

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (x_test, y_test)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
        
        
        
    def fit_model(self, epochs = 1):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)


        batch_size = self.batch_size
        epochs = epochs
        
        # VGG 16 (VGG 8 - 3 blocks) (https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16)
        # paper: https://arxiv.org/abs/1409.1556
        model = Sequential()
        # -- block 1 ----
        # Convolutional layer 1
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # -- block 2 ----
        # layer 3
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        # layer 4
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        # 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # -- block 3 ----
        # layer 5
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        # layer 6
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))



        model.add(Flatten())

    
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(Dense(512, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.25))

        # layer 7
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # layer 8
        model.add(Dense(10, activation='softmax'))
        
        
        
        opt = Adam(learning_rate=0.001) # default of adam
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        if not os.path.exists(os.path.join(root,'output', 'epoch_models', self.model_name)):
            os.makedirs(os.path.join(root,'output', 'epoch_models', self.model_name))
            
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epochs}.h5')
        model.save(model_path)
        
        return model
    
    def train_an_epoch_more(self,model,epoch_number):
        loaded_model = self.load_model_of_each_epoch(epoch_number)
        if loaded_model:
            return loaded_model
        
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = self.batch_size
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epoch_number}.h5')
        model.save(model_path)
        
        return model


    def load_original_model(self):
        original_model_path = os.path.join('models',self.model_name,'original_model.h5')
        print(original_model_path)
        
        if os.path.exists(original_model_path):
            model = load_model(original_model_path,compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
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
                model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
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