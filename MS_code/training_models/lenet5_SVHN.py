import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.callbacks import LearningRateScheduler
from keras.models import load_model



root = os.path.dirname(os.path.abspath(__file__)) 


class lenet5_SVHN():
    def __init__(self) -> None:
        self.model_name = 'lenet5_SVHN'
        self.dataset = 'SVHN'
        self.original_epochs = 20
        self.load_dataset()
    
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


        batch_size = 128
        epochs = epochs
        
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3))) # layer 1
        model.add(MaxPooling2D(pool_size=(2, 2))) # layer 2
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu')) # layer 3
        model.add(MaxPooling2D(pool_size=(2, 2))) # layer 4
        model.add(Flatten()) # layer 5
        model.add(Dense(120, activation='relu')) # layer 6
        model.add(Dense(84, activation='relu')) # layer 7
        model.add(Dense(10, activation='softmax')) # layer 8
        
        # def lr_schedule(epoch):
        #     """Learning rate scheduler function."""
        #     learning_rate = 1.0
        #     # if epoch > 10:
        #     #     learning_rate = 0.5
        #     # if epoch > 15:
        #     #     learning_rate = 0.1
        #     return learning_rate
        
        optimizer = keras.optimizers.Adadelta(learning_rate = 1.0)
        
        # lr_scheduler = LearningRateScheduler(lr_schedule)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        if not os.path.exists(os.path.join(root,'output', 'epoch_models', self.model_name)):
            os.makedirs(os.path.join(root,'output', 'epoch_models', self.model_name))
            
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epochs}.h5')
        model.save(model_path)
        
        return model
    
    def train_an_epoch_more(self, model, epoch_number):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = 128
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        
        model_path = os.path.join(root,'output', 'epoch_models', self.model_name,f'model_epoch_{epoch_number}.h5')
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
        
        print(self.x_test.shape)
        print(self.y_test)
        print(np.argmax(self.y_test, axis=1))
        
        predicted_y_by_original = model.predict(self.x_test, verbose=0)
        print(np.argmax(predicted_y_by_original, axis=1))
    
    