import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
import keras.backend as K
import os
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer


root = os.path.dirname(os.path.abspath(__file__)) 


def load_dataset(subject, data_type):
    # load data
    if data_type == 'mnist':
        num_classes = 10
        CLIP_MAX = 0.5
        ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
        (img_rows, img_cols) = (28, 28)
        
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)


        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train, x_test = color_preprocessing(x_train, x_test, 0, 255)
        # x_test = x_test.reshape(len(x_test), 28, 28, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # x_train = (x_train / 255) - (1 - CLIP_MAX)
        # x_test = (x_test / 255) - (1 - CLIP_MAX)

        x_train = x_train / 255
        x_test = x_test / 255
        
    elif data_type == 'cifar10':
        num_classes = 10
        
        if 'resnet50' in subject:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            # data preprocessing (it is necessary based on pretrained model here is ResNet50)
            x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
            x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
        
        else:   
            CLIP_MAX = 0.5

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

            # # y_train = np_utils.to_categorical(y_train, 10)
            # # y_test = np_utils.to_categorical(y_test, 10)
            
    elif data_type == 'fashion_mnist':
        num_classes = 10
        
        CLIP_MAX = 0.5

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
        
    elif data_type == 'SVHN':
        num_classes = 10
        
        # object = lenet5_SHVN.lenet5_SVHN()
        # (x_train, y_train), (x_test, y_test) = object.load_model()
        
        train_raw = loadmat(os.path.join(root,'datasets','SVHN', 'train_32x32.mat'))
        test_raw = loadmat(os.path.join(root,'datasets','SVHN', 'test_32x32.mat'))
        
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
        
        y_train = np.argmax(y_train,axis=1)
        y_test = np.argmax(y_test,axis=1)
        
        
        
    elif data_type == 'cifar100':
        num_classes = 100
        
        if 'resnet' in subject:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            
            # data preprocessing (it is necessary based on pretrained model here is ResNet)
            x_train = tf.keras.applications.resnet.preprocess_input(x_train)
            x_test = tf.keras.applications.resnet.preprocess_input(x_test)    
    
    
    return (x_train, y_train), (x_test, y_test), num_classes