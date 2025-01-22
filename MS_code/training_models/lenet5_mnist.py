import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import os

import keras



root = os.path.dirname(os.path.abspath(__file__)) 

### This model is used in deepcrime as well.
class lenet5_mnist():
    
    shifted_datasets = ['mnist_brightness','mnist_contrast', 'mnist_rotation', 'mnist_scale', 'mnist_shear','mnist_translation', 'mnist_combined',
                        'mnist_c_brightness','mnist_c_contrast', 'mnist_c_rotation', 'mnist_c_scale', 'mnist_c_shear','mnist_c_translation']
    
    
    def __init__(self, dataset_name = 'mnist') -> None:
        self.model_name = 'lenet5_mnist'
        self.dataset = dataset_name
        self.original_epochs = 12
        self.number_of_classes = 10
        self.load_dataset()
    
    def load_dataset(self, diff_test=''):
        
        if self.dataset in lenet5_mnist.shifted_datasets:
            CLIP_MIN = -0.5
            CLIP_MAX = 0.5

            (x_train, y_train), (_, y_test) = mnist.load_data() ## because we want to use shifted test input to test in different distribution of train inputs

            (img_rows, img_cols) = (28, 28)
            num_classes = 10

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_train = x_train.astype('float32')
            x_train = x_train / 255

            transformed_test_set_path = os.path.join(root,'..','datasets', 'transformed', 'mnist', f'{self.dataset}_test.npy')
            target_test_x = np.load(transformed_test_set_path)
            target_test_x = target_test_x.reshape(target_test_x.shape[0], img_rows, img_cols, 1)
            target_test_x = target_test_x.astype('float32')
            target_test_x = target_test_x / 255
            
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (target_test_x, y_test)
            
        
        else:
            ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
            (img_rows, img_cols) = (28, 28)
            num_classes = 10

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)


            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train = x_train / 255
            x_test = x_test / 255

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (x_test, y_test)
            
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
        
        
    def fit_model(self, epochs = 1):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)


        batch_size = 128
        epochs = epochs
        
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))) # layer 1
        model.add(MaxPooling2D(pool_size=(2, 2))) # layer 2
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu')) # layer 3
        model.add(MaxPooling2D(pool_size=(2, 2))) # layer 4
        model.add(Flatten()) # layer 5
        model.add(Dense(120, activation='relu')) # layer 6
        model.add(Dense(84, activation='relu')) # layer 7
        model.add(Dense(10, activation='softmax')) # layer 8
        
        optimizer = keras.optimizers.Adadelta(learning_rate = 0.05)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        
        if not os.path.exists(os.path.join(root,'output', 'epoch_models', self.model_name)):
            os.makedirs(os.path.join(root,'output', 'epoch_models', self.model_name))
            
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epochs}.h5')
        model.save(model_path)
        
        return model
    
    def train_an_epoch_more(self,model, num_of_epoch):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = 128
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{num_of_epoch}.h5')
        model.save(model_path)
        
        return model


    def load_original_model(self):
        original_model_path = os.path.join(root,'..', 'models', self.model_name,'original_model.h5')
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
    
    