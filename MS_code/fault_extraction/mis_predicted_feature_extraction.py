from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import json
import os
import sys
# sys.path.append("model_training")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from training_models.lenet4 import lenet4 # fashion_mnist
from training_models.lenet5_SVHN import lenet5_SVHN
# from training_models.lenet5 import lenet5 # mnist
from training_models.cifar10_conv import cifar10_conv # cifar10 conv-8
from training_models.resnet20_cifar10 import resnet20_cifar10 
from training_models.vgg16_SVHN import vgg16_SVHN 
# from training_models.resnet50 import resnet50 
from training_models.lenet5_mnist import lenet5_mnist 
# from training_models.resnet20_cifar10_retrain import resnet20_cifar10_retrain
# from training_models.lenet5_mnist_retrain import lenet5_mnist_retrain
# from training_models.lenet5_fashion_mnist_retrain import lenet5_fashion_mnist_retrain
from training_models.resnet152_cifar100 import resnet152_cifar100


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA as sklearnPCA
from keras.models import load_model, Model

from datetime import datetime

# ------------------------------ #
### ---- Feature Extraction ---- ####

from keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.decomposition import PCA as sklearnPCA
import argparse
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist, cifar10 , fashion_mnist, cifar100

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img




parent_dir_name = 'fault_extraction'

shifted_test_input = False ## this flag is used when you want to test on test input with different distribution

# dataset = 'fasion_mnist'
# model = 'lenet4'

retrain_subjects = ['retrain_resnet20_cifar10', 'retrain_lenet5_mnist', 'retrain_lenet5_fashion_mnist']

def get_model_object(model_name, dataset_name=''):
    print('model_name: ', model_name)
    print('dataset_name: ', dataset_name)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    
    if model_name == 'lenet4':
        object_of_model = lenet4()
        
    elif model_name == 'lenet5_SVHN':
        object_of_model = lenet5_SVHN()
        
    # elif model_name == 'lenet5':
    #     object_of_model = lenet5()
    
    elif model_name == 'cifar10':
        object_of_model = cifar10_conv()
    
    elif model_name == 'resnet20_cifar10':
        object_of_model = resnet20_cifar10(dataset_name)
        
    elif model_name == 'vgg16_SVHN':
        object_of_model = vgg16_SVHN()
    
    # elif model_name == 'resnet50_amazon':
    #     object_of_model = resnet50('amazon')
    
    # elif model_name == 'resnet50_office31_mix':
    #     object_of_model = resnet50('office31_mix')
    
    # elif model_name == 'resnet50_cifar10':
    #     object_of_model = resnet50('cifar10')
    
    elif model_name == 'lenet5_mnist':
        object_of_model = lenet5_mnist(dataset_name)
        
    # elif model_name == 'resnet50_caltech256':
    #     object_of_model = resnet50('caltech256')
    
    # elif model_name == 'resnet50_caltech256_8020':
    #     object_of_model = resnet50('caltech256_8020')
        
    # elif model_name == 'resnet50_office31':
    #     object_of_model = resnet50('office31')
    
    ### retrain models
    # elif model_name == 'retrain_resnet20_cifar10':
    #     if dataset_name:
    #         object_of_model = resnet20_cifar10_retrain(dataset_name)
    #     else:
    #         raise 'please set dataset_name'
    
    # elif model_name == 'retrain_lenet5_mnist':
    #     if dataset_name:
    #         object_of_model = lenet5_mnist_retrain(dataset_name)
    #     else:
    #         raise 'please set dataset_name'
        
    
    # elif model_name == 'retrain_lenet5_fashion_mnist':
    #     if dataset_name:
    #         object_of_model = lenet5_fashion_mnist_retrain(dataset_name)
    #     else:
    #         raise 'please set dataset_name'
    
    
    ## after revise
    elif model_name == 'resnet152_cifar100':
        print('here we have resnet152_cifar100')
        object_of_model = resnet152_cifar100(dataset_name)
        
        
    
    else:
        object_of_model = None
        print("model_name:", model_name)
        print("dataset:", dataset_name)
    
    
    return object_of_model


def get_mispredicteds_of_last_epoch(model_name, dataset_name=''):
    all_mis_predicteds = {}
    
    subject = get_model_object(model_name, dataset_name)
    if subject == None:
        return all_mis_predicteds
    
    model = subject.load_original_model()
    (x_train, y_train), (x_test, y_test) = subject.load_dataset()
        
    y_test_predict = model.predict(x_test)
    
    y_test_predict = np.argmax(y_test_predict, axis=1)

    y_test_actual = np.argmax(y_test, axis=1)
    test_mis_predicteds = []
    i = 0
    for t,p in zip(list(y_test_actual), list(y_test_predict)):
        if t != p:
            test_mis_predicteds.append(i)
        i += 1
    
    # get "train" set mispredicteds
    y_train_predict = model.predict(x_train)
    y_train_predict = np.argmax(y_train_predict, axis=1)

    y_train_actual = np.argmax(y_train, axis=1)
    train_mis_predicteds = []
    i = 0
    for t,p in zip(list(y_train_actual), list(y_train_predict)):
        if t != p:
            train_mis_predicteds.append(i)
        i += 1

    score = model.evaluate(x_test, y_test, verbose=0)
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print('score train:', score_train)
    print('size of train:',len(x_train))
    print('\n')
    print('score test:', score)
    print('size of test:',len(x_test))
    
    all_mis_predicteds[f"epoch_{subject.original_epochs}"] = {'score':score, 'test_mispredicted':test_mis_predicteds, 'train_mispredicted':train_mis_predicteds, 
                                            'y_test_predicted':y_test_predict.tolist(), 'y_train_predicted':y_train_predict.tolist()}
    return all_mis_predicteds


def get_mispredicteds_of_all_epochs(model_name, dataset_name = ''):
    all_mis_predicteds = {}
    subject = get_model_object(model_name, dataset_name)
    
    if subject == None:
        return all_mis_predicteds
    
    
    number_of_original_epochs = subject.original_epochs
    (x_train, y_train), (x_test, y_test) = subject.load_dataset()
    

    # # create folder for saving model on each epoch
    # saved_models_on_each_epchs = 'saved_models_on_each_epochs' 
    # save_path_dir = os.path.join(os.getcwd(), saved_models_on_each_epchs, subject)

    # if not os.path.exists(save_path_dir):
    #     os.makedirs(save_path_dir)

    for epoch in range(1,number_of_original_epochs+1): # Number of epochs 
        print(epoch)
        
        # model = tf.keras.models.load_model(os.path.join(save_path_dir, f'model_epoch_{epoch}.h5'))
        temp_model = subject.load_model_of_each_epoch(epoch)
        if temp_model:
            model = temp_model
            print(f'load model successfully epoch: {epoch}')
            
        else:    
            if epoch == 1:
                model = subject.fit_model(1)
            else:
                model = subject.train_an_epoch_more(model, epoch)
            
        
        # get "test" set mispredicteds
        y_test_predict = model.predict(x_test)
        y_test_predict = np.argmax(y_test_predict, axis=1)

        y_test_actual = np.argmax(y_test, axis=1)
        test_mis_predicteds = []
        i = 0
        for t,p in zip(list(y_test_actual), list(y_test_predict)):
            if t != p:
                test_mis_predicteds.append(i)
            i += 1
        
        # get "train" set mispredicteds
        y_train_predict = model.predict(x_train)
        y_train_predict = np.argmax(y_train_predict, axis=1)

        y_train_actual = np.argmax(y_train, axis=1)
        train_mis_predicteds = []
        i = 0
        for t,p in zip(list(y_train_actual), list(y_train_predict)):
            if t != p:
                train_mis_predicteds.append(i)
            i += 1

        score = model.evaluate(x_test, y_test, verbose=0)
        
        all_mis_predicteds[f"epoch_{epoch}"] = {'score':score, 'test_mispredicted':test_mis_predicteds, 'train_mispredicted':train_mis_predicteds, 
                                                'y_test_predicted':y_test_predict.tolist(), 'y_train_predicted':y_train_predict.tolist()}
        
    return all_mis_predicteds
    


def get_mispredictions_input(subject, last_epoch = True, dataset_name = ''):

    if subject in retrain_subjects:
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject+'_'+dataset_name)
    
    elif shifted_test_input: ## this is for transformed test inputs to test TEASMA on distribution shift inputs
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', 'transformed', subject, dataset_name)
        
    
    else:
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject)
    
    
     
    if os.path.isfile(os.path.join(output_path_dir,'mis_predicted_info.json')):
        print('load from exist file')
        with open(os.path.join(output_path_dir,'mis_predicted_info.json'),'r') as f:
            all_mispredicteds = json.load(f)

    else:
        if last_epoch:
            all_mispredicteds = get_mispredicteds_of_last_epoch(subject, dataset_name)
        else:
            all_mispredicteds = get_mispredicteds_of_all_epochs(subject, dataset_name)

        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir) 

        with open(os.path.join(output_path_dir,'mis_predicted_info.json'),'w') as f:
                json.dump(all_mispredicteds, f)


    for epoch, value in all_mispredicteds.items():
        print(f'--------------- {epoch} ---------------')
        print(f'Acuracy: {round(value["score"][1],4)}')
        print(f'Number of mis-predicteds on test: {len(value["test_mispredicted"])}')
        print(f'Number of mis-predicteds on train: {len(value["train_mispredicted"])}')
        print('\n')
    


        
# ----------------------------------------------------------------- #
### ---------------------- Do Normalization --------------------------------------- ###
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def scale_one(X):
  nom = (X-X.min())*(1)
  denom = X.max() - X.min()
  scaled = nom/denom 
  return scaled



def vgg16_features_GD(dataset_name, model_object=None):
    CLIP_MIN = -0.5
    CLIP_MAX = 0.5
    print('****************************')
    print(dataset_name)
    print('****************************')
    if (dataset_name=="cifar10" or dataset_name=="cifar100" or dataset_name=="SVHN"):
        if(dataset_name=="cifar10"):
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if(dataset_name=="cifar100"):
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        if (dataset_name=="SVHN"):
            print('load SVHN dataset')
            (x_train, y_train), (x_test, y_test) = lenet5_SVHN().load_dataset()
    

        # lb = LabelBinarizer()
        # train_labels = lb.fit_transform(train_labels)
        # test_labels = lb.fit_transform(test_labels)

        x_test1= x_test.reshape (-1,32,32,3)
        x_train= x_train.reshape(-1,32,32,3)
        x_test1 = x_test1.astype("float32")
        x_train= x_train.astype("float32")
        x_test1 = (x_test1 / 255.0) - (1.0 - CLIP_MAX)
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        

        model = VGG16(weights='imagenet', include_top=False)
        features = model.predict(x_test1)
        features= features.reshape((len(x_test1),512))
        X_scf_test = scale(features, 0, 1)
        fe = model.predict(x_train)
        fe= fe.reshape((len(x_train),512))
        X_scf_train = scale(fe, 0, 1)
        print("shape of feature matrix", X_scf_test.shape)
        print("rank of feature matrix", np.linalg.matrix_rank(X_scf_test))


    if (dataset_name =="mnist" or dataset_name=="fashion_mnist"):
        if (dataset_name=="mnist"):
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        elif(dataset_name=="fashion_mnist"):
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train1=np.dstack([x_train]*3)
        x_test1=np.dstack([x_test]*3)
        x_test1= x_test1.reshape(-1,28,28,3)
        x_train1= x_train1.reshape(-1,28,28,3)
        #Resize the images 48*48 as required by VGG16
        x_test1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test1])
        x_train1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train1])


        x_test1 = x_test1.astype("float32")
        x_test1 = (x_test1 / 255.0) - (1.0 - CLIP_MAX)
        x_train1 = x_train1.astype("float32")
        x_train1 = (x_train1 / 255.0) - (1.0 - CLIP_MAX)


        input_layer=layers.Input(shape=(48,48,3))
        model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16.summary()
    
        FF = model_vgg16.predict(x_test1)
        features= FF.reshape((len(x_test1),1*512))
        # print("rank of feature matrix", np.linalg.matrix_rank(features))
        X_scf_test = scale(features, 0, 1)
        # print("rank of feature matrix", np.linalg.matrix_rank(X_scf_test))
        print("shape of feature matrix", X_scf_test.shape)
        FTrain = model_vgg16.predict(x_train1)
        fe_train= FTrain.reshape((len(x_train1),1*512))
        # print("rank of feature matrix", np.linalg.matrix_rank(features))
        X_scf_train= scale(fe_train, 0, 1)
        
  
    if (dataset_name in ['amazon', 'office31_mix']):
        print('load office31 dataset')
        (x_train, y_train), (x_test, y_test) = resnet50(dataset_name).load_dataset()

        image_size = 224
        x_test= x_test.reshape (-1,image_size,image_size,3)
        x_train= x_train.reshape(-1,image_size,image_size,3)

        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
    
    
    if (dataset_name in ['caltech256', 'caltech256_8020','office31']):
        print('load dataset for feature extraction of ', dataset_name)
        (x_train, y_train), (x_test, y_test) = resnet50(dataset_name).load_dataset()

        image_size = 224
        x_test= x_test.reshape (-1,image_size,image_size,3)
        x_train= x_train.reshape(-1,image_size,image_size,3)

        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
    
    
    elif (model_object.model_name == 'retrain_resnet20_cifar10'):
        print('load dataset for feature extraction of ', dataset_name)
        
        (x_train, y_train), (x_test, y_test) = model_object.load_dataset()

        image_size = 32 ## for cifar10
        
        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test, verbose=2)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train, verbose=2)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
    
    
    elif (model_object.model_name == 'retrain_lenet5_mnist' or model_object.model_name == 'retrain_lenet5_fashion_mnist'):
        print('load dataset for feature extraction of ', dataset_name)
        
        (x_train, y_train), (x_test, y_test) = model_object.load_dataset()

        # Resize images to 224x224
        image_size = 224 ## for mnist
        x_train = tf.image.resize(x_train, [224, 224])
        x_test = tf.image.resize(x_test, [224, 224])
        
        x_train = tf.repeat(x_train, 3, axis=-1)
        x_test = tf.repeat(x_test, 3, axis=-1)
        
        print('shape of mnist:', x_train.shape)
        
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test, verbose=2)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train, verbose=2)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
        
    
    #######################################################################################
    # we added following to avaluate of shifted test inputs on TEASMA (predicted model)   #
    #######################################################################################
    
    elif (model_object.model_name == 'resnet20_cifar10' and dataset_name in ['cifar10_brightness','cifar10_contrast', 'cifar10_rotation', 'cifar10_scale', 'cifar10_shear','cifar10_translation']):
        print('load dataset for feature extraction of resnet20_cifar10', dataset_name)
        
        (x_train, y_train), (x_test, y_test) = model_object.load_dataset()

        image_size = 32 ## for cifar10
        
        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test, verbose=2)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train, verbose=2)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
        
    elif (model_object.model_name == 'lenet5_mnist' and dataset_name in lenet5_mnist.shifted_datasets):
        print('load dataset for feature extraction of lenet5_mnist', dataset_name)
        
        (x_train, y_train), (x_test, y_test) = model_object.load_dataset()
        
        ## from zohre
        x_train = x_train - (1.0 - CLIP_MAX)
        x_test = x_test - (1.0 - CLIP_MAX)
        
        image_size = 48 ## for mnist
        x_train = tf.image.resize(x_train, [image_size, image_size])
        x_test = tf.image.resize(x_test, [image_size, image_size])
        
        x_train = tf.repeat(x_train, 3, axis=-1)
        x_test = tf.repeat(x_test, 3, axis=-1)
        
        print('shape of mnist:', x_train.shape)
        
        
        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test, verbose=2)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train, verbose=2)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
    
    
    elif (model_object.model_name == 'resnet152_cifar100'):
        print('load dataset for feature extraction of resnet152_cifar100', dataset_name)
        
        (x_train, y_train), (x_test, y_test) = model_object.load_dataset()

        image_size = 32 ## for cifar10
        
        #   input_layer=layers.Input(shape=(image_size, image_size, 3))
        #   model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
        model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)
        
        FF = model_vgg16.predict(x_test, verbose=2)
        features= FF.reshape(FF.shape[0], -1)
        X_scf_test = scale(features, 0, 1)
        print("shape of feature matrix test", X_scf_test.shape)
        
        FTrain = model_vgg16.predict(x_train, verbose=2)
        fe_train= FTrain.reshape(FTrain.shape[0], -1)
        X_scf_train= scale(fe_train, 0, 1)
        print("shape of feature matrix train", X_scf_train.shape)
        
        
    return X_scf_train, X_scf_test


####################################
# call feature extraction function #
####################################




def mis(VGGinputtest, VGGinputtrain, subject, epoch, dataset_name = ''):

    """
    Returns misclassified samples and their corresponding labels from the test set as well as the training set.

    Args:
    test_features: numpy array of shape (n_test_samples, feature_dim) containing features of test samples
    train_features: numpy array of shape (n_train_samples, feature_dim) containing features of training samples

    Returns:
    x_mis: numpy array of shape (n_misclassified_samples, feature_dim) containing misclassified samples
    y_true: list of true labels of misclassified samples from both test and training set
    y_pred: list of predicted labels of misclassified samples from both test and training set
    mis_test_index: list of indices of misclassified samples from the test set
    mis_train_index: list of indices of misclassified samples from the training set
    """

    # def feature_extraction(subject):
    model = subject
    model_object = get_model_object(subject, dataset_name)
    dataset = model_object.dataset



    (x_train, y_train), (x_test, y_test) = model_object.load_dataset()

    # you can pass test_mis_predicteds and train_mis_predicteds as arguments because we saved it before in a dictionaruy

    y_actual_test_and_train = [] # tt
    y_predicted_test_and_train = [] # tst

    # saved_models_on_each_epchs = 'saved_models_on_each_epchs'
    # load_path_dir = os.path.join(os.getcwd(), saved_models_on_each_epchs, model_name)
    if subject in retrain_subjects:
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds',subject+'_'+dataset_name)
    
    elif shifted_test_input: ## this is for transformed test inputs to test TEASMA on distribution shift inputs
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', 'transformed', subject, dataset_name)
        
    else:
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject)
        
    with open(os.path.join(output_path_dir,'mis_predicted_info.json'),'r') as f:
        all_mis_predicteds = json.load(f)


    test_mis_predicteds = np.array(all_mis_predicteds[f'epoch_{epoch}']["test_mispredicted"])
    train_mis_predicteds = np.array(all_mis_predicteds[f'epoch_{epoch}']["train_mispredicted"])

    y_test_predict = np.array(all_mis_predicteds[f'epoch_{epoch}']["y_test_predicted"])
    y_train_predict = np.array(all_mis_predicteds[f'epoch_{epoch}']["y_train_predicted"])



    ### load model here : ...
    
    # y_test_predict = model.predict(x_test) 
    # y_test_predict = np.argmax(y_test_predict, axis=1)

    y_test_actual = np.argmax(y_test, axis=1)
    # test_mis_predicteds = []
    # i = 0
    # for t,p in zip(list(y_test_actual), list(y_test_predict)):
    #     if t != p:
    #         test_mis_predicteds.append(i) # mis test index ( old name is mis_index)

    #         y_actual_test_and_train.append(y_test_actual[i])
    #         y_predicted_test_and_train.append(y_test_predict[i])
        
    #     i +=1


    # y_train_predict = model.predict(x_train)
    # y_train_predict = np.argmax(y_train_predict, axis=1)

    y_train_actual = np.argmax(y_train, axis=1)
    # train_mis_predicteds = []
    # i = 0
    # for t,p in zip(list(y_train_actual), list(y_train_predict)):
    #     if t != p:
    #         train_mis_predicteds.append(i) # mis train index ( old name is mis_tindex)

    #         y_actual_test_and_train.append(y_train_actual[i])
    #         y_predicted_test_and_train.append(y_train_predict[i])

    #     i += 1

    y_actual_test_and_train = np.concatenate((y_test_actual[test_mis_predicteds], y_train_actual[train_mis_predicteds]), axis=0) 
    y_predicted_test_and_train = np.concatenate((y_test_predict[test_mis_predicteds], y_train_predict[train_mis_predicteds]), axis=0)
    
        
    x_vgg_mis = np.concatenate((VGGinputtest[test_mis_predicteds], VGGinputtrain[train_mis_predicteds]), axis=0) # x_s,


        
    return x_vgg_mis, y_actual_test_and_train , y_predicted_test_and_train , test_mis_predicteds, train_mis_predicteds # x_mis, y_true, y_pred, mis_test_index, mis_train_index



def feature_extraction(subject, dataset_name= ''):

    # model = subject
    model_object = get_model_object(subject, dataset_name)

    print('start feature extraction')
    
    inputtrain, inputtest = vgg16_features_GD(model_object.dataset, model_object)

    print(inputtrain.shape)
    print(inputtest.shape)


    data_for_clustering = {}
    for ep in range(1, model_object.original_epochs+1):
        try:
            x_mis, tt , tst , mis_index, mis_tindex = mis(inputtest , inputtrain, subject=subject, epoch=ep, dataset_name=dataset_name)
            print(f'epoch_{ep} done!')
            data_for_clustering[f'epoch_{ep}'] = {'x_mis': x_mis, 
                                                'y_actual_test_and_train': tt,
                                                'y_predicted_test_and_train':tst,
                                                'mis_test_index':mis_index,
                                                'mis_train_index':mis_tindex}
        except:
            print(f'The file of the epoch {ep} not found :( , please check mispredicteds indeces of test set')
    
    
    print('==========================')
    print('subject:', subject)
    
    if subject in retrain_subjects:
        print(' *************** here')
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject+'_'+dataset_name)   
    
    elif shifted_test_input: ## this is for transformed test inputs to test TEASMA on distribution shift inputs
        print(' *************** test on different distribution')
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', 'transformed', subject, dataset_name)
         
    else:
        print('************ there')
        output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject)    
    
    print(output_path_dir)  
    np.save(os.path.join(output_path_dir, 'data_for_clustering_for_epoch.npy'), data_for_clustering)




if __name__ == "__main__":
    subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN', 'resnet152_cifar100']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        help="subject_dataset")
    
    parser.add_argument("--dataset_name", "-dataset_name",
                        type=str,
                        default='',
                        help="dataset name")
    
    parser.add_argument("--shifted_test_input", "-shifted_test_input",
                        type=int,
                        default=0,
                        help="if you want to test the prediction model on shifted data just set up this to 1")
    
    
    
    args = parser.parse_args()
    subject = args.subject
    dataset_name = args.dataset_name
    shifted_test_input = args.shifted_test_input
    use_last_epoch = True
    

    ### this line is for diff test data if you want to use diff test that acheived from finetuning
    ### please be careful that the load data from training models should be change if you want to use diff data
    
    
    # for subject,last_epoch in zip(subjects, use_last_epoch):
        
    start_time = datetime.now()
    
    get_mispredictions_input(subject, dataset_name=dataset_name)
    feature_extraction(subject, dataset_name=dataset_name)
    
    end_time = datetime.now()
    with open('cost_of_TEASMA.txt', 'a') as file:
        file.write(f"Time for extract misprediction input and features of subject {subject} dataset {dataset_name} is = {end_time - start_time}.\n")
        
    

