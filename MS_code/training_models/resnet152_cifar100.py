import tensorflow as tf
from keras.models import load_model
import numpy as np
import os




root = os.path.dirname(os.path.abspath(__file__)) 
import sys

sys.path.insert(1, os.path.join(root, '..'))
from preprocess_data import *



class resnet152_cifar100():
    def __init__(self, dataset_name='') -> None:
        self.model_name = 'resnet152_cifar100'
        self.dataset = dataset_name
        self.number_of_classes = 100
        self.original_epochs = 10 ## for finetuning
    
    def load_dataset(self, diff_test = False):
        # Load cifar100 and normalization

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        # data preprocessing (it is necessary based on pretrained model here is ResNet)
        x_train = tf.keras.applications.resnet.preprocess_input(x_train)
        x_test = tf.keras.applications.resnet.preprocess_input(x_test)
        
        y_train = tf.keras.utils.to_categorical(y_train, self.number_of_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.number_of_classes)
        
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (x_test, y_test)
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
        
        
        
    def fit_model(self, epochs = 1):
        '''
        ## we used the following code to finetune cifar100 on ResNet152 (you can use resnet152_finetuning.py) - trained model is original_model.h5
        
        num_of_classes = 100
        
        (x_train, y_train), (x_test, y_test) = self.load_dataset()
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

        base_model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        model = Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(.25),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(num_of_classes, activation='softmax')
        ])

        # Freeze the base model's layers
        base_model.trainable = False


        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        #early stopping to monitor the validation loss and avoid overfitting
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)

        #reducing learning rate on plateau
        rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)


        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train,validation_data = (x_val, y_val), epochs=num_of_epochs, batch_size=64, verbose=2,
                callbacks=[rlrop])
        
        '''
        
        
        pass
        
    
    def train_an_epoch_more(self, model, epoch_number):
        # return model
        pass


    def load_original_model(self):
        # if self.dataset:
        #     model_name = self.model_name+'_'+self.dataset
        # else:
        #     model_name = self.model_name
        
        model_name = self.model_name
        
        original_model_path = os.path.join(root,'..', 'models', model_name,'original_model.h5')
        print(original_model_path)

        if os.path.exists(original_model_path):
            model = load_model(original_model_path,compile=False)
            model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1), loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        # else:
        #     print('start training')
        #     model = self.fit_model(epochs = self.original_epochs)
        #     model.save(original_model_path)
        #     return model
        
        return None 
        
    def load_model_of_each_epoch(self, epoch = 0):
        
        # if self.dataset:
        #     model_name = self.model_name+'_'+self.dataset
        # else:
        #     model_name = self.model_name
            
        if epoch:
            print('===========')
            if epoch == self.original_epochs:
                model = self.load_original_model()
                return model
            model_path = os.path.join(root,'output', 'epoch_models', model_name,f'model_epoch_{epoch}.h5')
            print(model_path)
            if os.path.isfile(model_path):
                model = load_model(model_path, compile=False)
                model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=1), loss='categorical_crossentropy', metrics=['accuracy'])
                return model
            else:
                print('not found model with path:',model_path)
                return None
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
    
    