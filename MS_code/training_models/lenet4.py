import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import time
import os

root = os.path.dirname(os.path.abspath(__file__)) 

class lenet4():
    def __init__(self) -> None:
        self.model_name = 'lenet4'
        self.dataset = 'fashion_mnist'
        self.original_epochs = 20
        self.load_dataset()
    
    def load_dataset(self):
        # Load Fashion-MNIST dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        CLIP_MIN = -0.5
        CLIP_MAX = 0.5

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


        ## Define LeNet-4 model (reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791) (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833432)
        batch_size = 100
        epochs = epochs
        
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))) 
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2))) 
        model.add(Flatten()) 
        model.add(Dense(120, activation='relu')) 
        model.add(Dense(10, activation='softmax')) 

        optimizer = tf.keras.optimizers.Adadelta(learning_rate=1)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        if not os.path.exists(os.path.join(root,'output', 'epoch_models', self.model_name)):
            os.makedirs(os.path.join(root,'output', 'epoch_models', self.model_name))
            
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epochs}.h5')
        model.save(model_path)
        
        return model
    
    def train_an_epoch_more(self,model,epoch_number):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = 100
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epoch_number}.h5')
        model.save(model_path)
        
        return model


    def load_original_model(self):
        original_model_path = os.path.join('models',self.model_name,'original_model.h5')
        print(original_model_path)
        
        if os.path.exists(original_model_path):
            model = load_model(original_model_path,compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1), loss='categorical_crossentropy', metrics=['accuracy'])
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
                model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1), loss='categorical_crossentropy', metrics=['accuracy'])
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