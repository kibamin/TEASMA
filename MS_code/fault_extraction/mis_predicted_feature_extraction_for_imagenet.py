from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import json
import os
import sys
# sys.path.append("model_training")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# from training_models.lenet4 import lenet4 # fashion_mnist
# from training_models.lenet5_SVHN import lenet5_SVHN
# from training_models.lenet5 import lenet5 # mnist
# from training_models.cifar10_conv import cifar10_conv # cifar10 conv-8
# from training_models.resnet20_cifar10 import resnet20_cifar10 
# from training_models.vgg16_SVHN import vgg16_SVHN 
# from training_models.resnet50 import resnet50 
# from training_models.lenet5_mnist import lenet5_mnist 
# from training_models.resnet20_cifar10_retrain import resnet20_cifar10_retrain
# from training_models.lenet5_mnist_retrain import lenet5_mnist_retrain
# from training_models.lenet5_fashion_mnist_retrain import lenet5_fashion_mnist_retrain
# from training_models.resnet152_cifar100 import resnet152_cifar100
from training_models.inceptionV3_imagenet import inceptionV3_imagenet
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from sklearn.preprocessing import scale
from keras.preprocessing import image

from tqdm import tqdm

# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from sklearn.decomposition import PCA as sklearnPCA
# from keras.models import load_model, Model

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


import tensorflow_datasets as tfds



root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')






parent_dir_name = 'fault_extraction'

shifted_test_input = False ## this flag is used when you want to test on test input with different distribution

# dataset = 'fasion_mnist'
# model = 'lenet4'

retrain_subjects = ['retrain_resnet20_cifar10', 'retrain_lenet5_mnist', 'retrain_lenet5_fashion_mnist']

def get_model_object(model_name, dataset_name=''):
    print('model_name: ', model_name)
    print('dataset_name: ', dataset_name)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    object_of_model = inceptionV3_imagenet()
    
    # if model_name == 'lenet4':
    #     object_of_model = lenet4()
        
    # elif model_name == 'lenet5_SVHN':
    #     object_of_model = lenet5_SVHN()
        
    # # elif model_name == 'lenet5':
    # #     object_of_model = lenet5()
    
    # elif model_name == 'cifar10':
    #     object_of_model = cifar10_conv()
    
    # elif model_name == 'resnet20_cifar10':
    #     object_of_model = resnet20_cifar10(dataset_name)
        
    # elif model_name == 'vgg16_SVHN':
    #     object_of_model = vgg16_SVHN()
    
    # elif model_name == 'resnet50_amazon':
    #     object_of_model = resnet50('amazon')
    
    # elif model_name == 'resnet50_office31_mix':
    #     object_of_model = resnet50('office31_mix')
    
    # elif model_name == 'resnet50_cifar10':
    #     object_of_model = resnet50('cifar10')
    
    # elif model_name == 'lenet5_mnist':
    #     object_of_model = lenet5_mnist(dataset_name)
        
    # elif model_name == 'resnet50_caltech256':
    #     object_of_model = resnet50('caltech256')
    
    # elif model_name == 'resnet50_caltech256_8020':
    #     object_of_model = resnet50('caltech256_8020')
        
    # elif model_name == 'resnet50_office31':
    #     object_of_model = resnet50('office31')
    
    # ### retrain models
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
    
    
    # ## after revise
    # elif model_name == 'resnet152_cifar100':
    #     print('here we have resnet152_cifar100')
    #     object_of_model = resnet152_cifar100(dataset_name)
        
        
    
    # else:
    #     object_of_model = None
    #     print("model_name:", model_name)
    #     print("dataset:", dataset_name)
    
    
    return object_of_model





def load_train_test_set_imagenet():
    data_dir = os.path.join(root, 'datasets', 'imagenet')
    write_dir = os.path.join(root, 'datasets', 'tf-imagenet-dirs')

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
                        extract_dir=os.path.join(write_dir, 'extracted'),
                        manual_dir=data_dir
                    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config}
    
    train_data, validation_data = tfds.load('imagenet2012', 
            data_dir=os.path.join(write_dir, 'data'),         
            split=['train', 'validation'], 
        #    split='validation', 
            shuffle_files=False, 
            download=False, 
            as_supervised=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs)


    def preprocess_image(image, label):
        # logger.info(f'process_image function started for subject:{subject}')
        # Resize the smaller side to 256 while maintaining aspect ratio
        initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        ratio = 256.0 / tf.reduce_min(initial_shape)
        new_shape = tf.cast(initial_shape * ratio, tf.int32)
        image = tf.image.resize(image, new_shape)
        
        ## inceptionV3
        image = tf.image.resize_with_crop_or_pad(image, 299, 299)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        
        return image, label

    
    train_data = train_data.map(preprocess_image).batch(256)
    val_data = validation_data.map(preprocess_image).batch(256)
    
    return train_data, val_data

def load_train_test_set_imagenet_for_vgg16(batch_size = 256):
    data_dir = os.path.join(root, 'datasets', 'imagenet')
    write_dir = os.path.join(root, 'datasets', 'tf-imagenet-dirs')

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
                        extract_dir=os.path.join(write_dir, 'extracted'),
                        manual_dir=data_dir
                    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config}
    
    train_data, validation_data = tfds.load('imagenet2012', 
            data_dir=os.path.join(write_dir, 'data'),         
            split=['train', 'validation'], 
        #    split='validation', 
            shuffle_files=False, 
            download=False, 
            as_supervised=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs)


    def preprocess_image(image, label):
        '''
        Preprocess the image by resizing and applying the specific preprocessing function for VGG16.
        '''
        # Resize the smaller side to 256 while maintaining aspect ratio
        initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        ratio = 256.0 / tf.reduce_min(initial_shape)
        new_shape = tf.cast(initial_shape * ratio, tf.int32)
        image = tf.image.resize(image, new_shape)
        # Crop the central 224x224 for VGG16
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        # Apply the specific preprocess input function for VGG16
        image = vgg16_preprocess_input(image)
        return image, label

    
    train_data = train_data.map(preprocess_image).batch(batch_size)
    val_data = validation_data.map(preprocess_image).batch(batch_size)
    
    return train_data, val_data







def get_mispredicteds_of_last_epoch(model_name, dataset_name=''):
    all_mis_predicteds = {}
    model_name = 'inceptionV3_imagenet'
    subject = get_model_object(model_name, 'imagenet')
    
    train_pred_output_path = os.path.join(os.getcwd(), 'mutants', 'mutant_prediction_outputs', model_name, 'train_set', 'original_model_prediction_outputs.npy')
    test_pred_output_path = os.path.join(os.getcwd(), 'mutants', 'mutant_prediction_outputs', model_name, 'test_set', 'original_model_prediction_outputs.npy')

    train_pred_output = np.load(train_pred_output_path)
    test_pred_output = np.load(test_pred_output_path)
    
    train_actual_labels = np.load(os.path.join(os.getcwd(), 'models', model_name, 'actual_labels','all_actual_labels_train.npy'))
    test_actual_labels = np.load(os.path.join(os.getcwd(), 'models', model_name, 'actual_labels', 'all_actual_labels_validation.npy'))
    
    index_of_image = -1
    train_mis_predicteds = []
    for actual_y, predicted_y in zip(train_actual_labels, train_pred_output):
        index_of_image += 1
        if actual_y != predicted_y:
            train_mis_predicteds.append(index_of_image)
            
    train_score = (len(train_actual_labels) - len(train_mis_predicteds)) / len(train_actual_labels) ## train accuracy
    print('train score', train_score)
    
    index_of_image = -1
    test_mis_predicteds = []
    for actual_y, predicted_y in zip(test_actual_labels, test_pred_output):
        index_of_image += 1
        if actual_y != predicted_y:
            test_mis_predicteds.append(index_of_image)
            
    test_score = (len(test_actual_labels) - len(test_mis_predicteds)) / len(test_actual_labels) ## test accuracy
    print('test score', test_score)
    
    
           
    all_mis_predicteds[f"epoch_{subject.original_epochs}"] = {'score':[0, test_score], 'test_mispredicted':test_mis_predicteds, 'train_mispredicted':train_mis_predicteds, 
                                            'y_test_predicted':test_pred_output.tolist(), 'y_train_predicted':train_pred_output.tolist(), 'train_score':[0, train_score]}
    return all_mis_predicteds



def get_mispredicteds_of_all_epochs(model_name, dataset_name = ''):
    print('if you want use this function please uncomment bellow lines')
#     all_mis_predicteds = {}
#     subject = get_model_object(model_name, dataset_name)
    
#     if subject == None:
#         return all_mis_predicteds
    
    
#     number_of_original_epochs = subject.original_epochs
#     (x_train, y_train), (x_test, y_test) = subject.load_dataset()
    

#     # # create folder for saving model on each epoch
#     # saved_models_on_each_epchs = 'saved_models_on_each_epochs' 
#     # save_path_dir = os.path.join(os.getcwd(), saved_models_on_each_epchs, subject)

#     # if not os.path.exists(save_path_dir):
#     #     os.makedirs(save_path_dir)

#     for epoch in range(1,number_of_original_epochs+1): # Number of epochs 
#         print(epoch)
        
#         # model = tf.keras.models.load_model(os.path.join(save_path_dir, f'model_epoch_{epoch}.h5'))
#         temp_model = subject.load_model_of_each_epoch(epoch)
#         if temp_model:
#             model = temp_model
#             print(f'load model successfully epoch: {epoch}')
            
#         else:    
#             if epoch == 1:
#                 model = subject.fit_model(1)
#             else:
#                 model = subject.train_an_epoch_more(model, epoch)
            
        
#         # get "test" set mispredicteds
#         y_test_predict = model.predict(x_test)
#         y_test_predict = np.argmax(y_test_predict, axis=1)

#         y_test_actual = np.argmax(y_test, axis=1)
#         test_mis_predicteds = []
#         i = 0
#         for t,p in zip(list(y_test_actual), list(y_test_predict)):
#             if t != p:
#                 test_mis_predicteds.append(i)
#             i += 1
        
#         # get "train" set mispredicteds
#         y_train_predict = model.predict(x_train)
#         y_train_predict = np.argmax(y_train_predict, axis=1)

#         y_train_actual = np.argmax(y_train, axis=1)
#         train_mis_predicteds = []
#         i = 0
#         for t,p in zip(list(y_train_actual), list(y_train_predict)):
#             if t != p:
#                 train_mis_predicteds.append(i)
#             i += 1

#         score = model.evaluate(x_test, y_test, verbose=0)
        
#         all_mis_predicteds[f"epoch_{epoch}"] = {'score':score, 'test_mispredicted':test_mis_predicteds, 'train_mispredicted':train_mis_predicteds, 
#                                                 'y_test_predicted':y_test_predict.tolist(), 'y_train_predicted':y_train_predict.tolist()}
        
#     return all_mis_predicteds
    print('')
    


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
        print(f'test_Acuracy: {round(value["score"][1],4)}')
        print(f'train_Acuracy: {round(value["train_score"][1],4)}')
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
    print('****************************')
    print(dataset_name)
    print('****************************')
    
    
    
    if (model_object.model_name == 'inceptionV3_imagenet'):
        print('load dataset for feature extraction of inceptionV3_imagenet', dataset_name)
        

    subject = 'inceptionV3_imagenet'
    
    train_data, validation_data = load_train_test_set_imagenet_for_vgg16(batch_size = 512)

    # Define image size for ImageNet
    image_size = 224  # VGG16 expects 224x224 images for ImageNet

    # Load the pre-trained VGG16 model
    model_vgg16 = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False)

    # Extract features for the validation set
    # temp_ff = []
    # for batch, _ in validation_data:
    #     ff = model_vgg16.predict(batch, verbose=2)
    #     temp_ff.append(ff)
    #     if len(temp_ff) == 4:
    #         break
        
    # FF = np.vstack(temp_ff)
    # features = FF.reshape(FF.shape[0], -1)
    # X_scf_test = scale(features, 0, 1)
    # print("Shape of feature matrix test:", X_scf_test.shape)
    # Process and save features in smaller batches
    
    test_output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject, 'validation_vgg16_features') 
    if not os.path.exists(test_output_path_dir):
        os.makedirs(test_output_path_dir)
        
        
    
     # Extract features for the validation set
    batch_index = 0
    for batch, _ in validation_data:
        print(batch_index)
        FF_batch = model_vgg16.predict(batch, verbose=2)
        features_batch = FF_batch.reshape(FF_batch.shape[0], -1)
        X_scf_test_batch = scale(features_batch, 0, 1)
        np.save(os.path.join(test_output_path_dir, f'X_scf_test_batch_{batch_index}.npy'), X_scf_test_batch)
        batch_index += 1
        
    print('Validation set done!')
    
    # FF = np.vstack([model_vgg16.predict(batch, verbose=2) for batch, _ in validation_data])
    # features = FF.reshape(FF.shape[0], -1)
    # X_scf_test = scale(features, 0, 1)
    # print("Shape of feature matrix test:", X_scf_test.shape)
    # np.save(os.path.join(output_path_dir, 'X_scf_test.npy'), X_scf_test)


    # Extract features for the training set
    train_output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject, 'train_vgg16_features') 
    if not os.path.exists(train_output_path_dir):
        os.makedirs(train_output_path_dir)
        
    batch_index = 0
    for batch, _ in train_data:
        print(batch_index)
        FF_batch = model_vgg16.predict(batch, verbose=2)
        features_batch = FF_batch.reshape(FF_batch.shape[0], -1)
        X_scf_train_batch = scale(features_batch, 0, 1)
        np.save(os.path.join(train_output_path_dir, f'X_scf_train_batch_{batch_index}.npy'), X_scf_train_batch)
        batch_index += 1
        
    print('Train set done!')
    
    # FTrain = np.vstack([model_vgg16.predict(batch, verbose=2) for batch, _ in train_data])
    # fe_train = FTrain.reshape(FTrain.shape[0], -1)
    # X_scf_train = scale(fe_train, 0, 1)
    # print("Shape of feature matrix train:", X_scf_train.shape)
    # np.save(os.path.join(output_path_dir, 'X_scf_train.npy'), X_scf_train)

    X_scf_test = [np.load(os.path.join(test_output_path_dir, f)) for f in sorted(os.listdir(test_output_path_dir)) if f.startswith('X_scf_test_batch_')]
    X_scf_test = np.vstack(X_scf_test)
    np.save(os.path.join(test_output_path_dir, 'X_scf_test_total.npy'), X_scf_test)
    
    print(X_scf_test.shape)
    print('save total test')
    
    
    X_scf_test = [np.load(os.path.join(train_output_path_dir, f)) for f in sorted(os.listdir(train_output_path_dir)) if f.startswith('X_scf_train_batch_')]
    X_scf_test = np.vstack(X_scf_test)
    np.save(os.path.join(train_output_path_dir, 'X_scf_train_total.npy'), X_scf_test)
    print(X_scf_test.shape)
    print('save total train')
        
    return X_scf_test, X_scf_test


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

   
    model = subject
    model_object = get_model_object(subject, dataset_name)
    dataset = model_object.dataset

    model_name = 'inceptionV3_imagenet'
    subject = model_name
    
    train_pred_output_path = os.path.join(os.getcwd(), 'mutants', 'mutant_prediction_outputs', model_name, 'train_set', 'original_model_prediction_outputs.npy')
    test_pred_output_path = os.path.join(os.getcwd(), 'mutants', 'mutant_prediction_outputs', model_name, 'test_set', 'original_model_prediction_outputs.npy')

    train_pred_output = np.load(train_pred_output_path)
    test_pred_output = np.load(test_pred_output_path)
    
    train_actual_labels = np.load(os.path.join(os.getcwd(), 'models', model_name, 'actual_labels','all_actual_labels_train.npy'))
    test_actual_labels = np.load(os.path.join(os.getcwd(), 'models', model_name, 'actual_labels', 'all_actual_labels_validation.npy'))

    y_train = train_actual_labels
    y_test = test_actual_labels

    # you can pass test_mis_predicteds and train_mis_predicteds as arguments because we saved it before in a dictionaruy

    y_actual_test_and_train = [] # tt
    y_predicted_test_and_train = [] # tst

    # saved_models_on_each_epchs = 'saved_models_on_each_epchs'
    # load_path_dir = os.path.join(os.getcwd(), saved_models_on_each_epchs, model_name)

    output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject)
        
    with open(os.path.join(output_path_dir,'mis_predicted_info.json'),'r') as f:
        all_mis_predicteds = json.load(f)


    test_mis_predicteds = np.array(all_mis_predicteds[f'epoch_{epoch}']["test_mispredicted"]) ## index of mispredicteds of test
    train_mis_predicteds = np.array(all_mis_predicteds[f'epoch_{epoch}']["train_mispredicted"]) ## index of mispredicteds of trian

    y_test_predict = np.array(all_mis_predicteds[f'epoch_{epoch}']["y_test_predicted"])
    y_train_predict = np.array(all_mis_predicteds[f'epoch_{epoch}']["y_train_predicted"])



    ### load model here : ...
    

    y_test_actual = y_test
    y_train_actual = y_train
    


    
    y_actual_test_and_train = np.concatenate((y_test_actual[test_mis_predicteds], y_train_actual[train_mis_predicteds]), axis=0) 
    y_predicted_test_and_train = np.concatenate((y_test_predict[test_mis_predicteds], y_train_predict[train_mis_predicteds]), axis=0)
    
    # for imagenet we can consider batch size = 1 and then extract features based on test_mis_predicteds (indeces)
    '''
    for i in test_mis_predicteds:
        batch, label = validation_data(i)
        ff = predict by vgg16 on batch
        and save to a array
    '''    
    # test_mis_index_ff = load_vgg_features(test_mis_predicteds, data_type= 'test')
    # train_mis_index_ff = load_vgg_features(train_mis_predicteds, data_type= 'train')
    test_mis_index_ff = []
    train_mis_index_ff = []
    
    x_vgg_mis = np.concatenate((test_mis_index_ff, train_mis_index_ff), axis=0) # x_s,
    # x_vgg_mis = np.concatenate((VGGinputtest[test_mis_predicteds], VGGinputtrain[train_mis_predicteds]), axis=0) # x_s,


        
    return x_vgg_mis, y_actual_test_and_train , y_predicted_test_and_train , test_mis_predicteds, train_mis_predicteds # x_mis, y_true, y_pred, mis_test_index, mis_train_index



def load_vgg_features(test_indexes, data_type = 'test'):
    
    
    test_output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject, 'validation_vgg16_features') 
    if data_type =='train':
        test_output_path_dir = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject, 'train_vgg16_features') 

    batch_size = 512
    
    # Original index positions before sorting
    original_positions = list(range(len(test_indexes)))

    # Sort the test indexes and keep track of the original order
    sorted_indexes_with_positions = sorted(zip(test_indexes, original_positions), key=lambda x: x[0])
    sorted_test_indexes = [x[0] for x in sorted_indexes_with_positions]

    # Function to get batch number for a given index
    def get_batch_number(index, batch_size):
        return index // batch_size

    # Find batch numbers that need to be loaded
    # batch_numbers = sorted(set(get_batch_number(index, batch_size) for index in sorted_test_indexes))
    
    desired_feature_output = np.empty((len(sorted_indexes_with_positions), 25088)) ### 7x7x512 is shape of output of vgg16 
    old_batch_number = -1
    for image_index, pos_index in sorted_indexes_with_positions:
        batch_number = get_batch_number(image_index, batch_size)
        if batch_number != old_batch_number:
            ff_batch_data = np.load(os.path.join(test_output_path_dir, f"X_scf_{data_type}_batch_{batch_number}.npy"))
        # else:
        #     use previuos ff_batch_data
        
        old_batch_number = batch_number
        
        index_in_batch_data = image_index - (batch_number*batch_size)
        desired_feature_output[pos_index] = ff_batch_data[index_in_batch_data]
    
    desired_feature_output = np.vstack(desired_feature_output)
    
    save_output_path = os.path.join(os.getcwd(), parent_dir_name, 'output','mispredicteds', subject, 'features_of_mispredicted_inputs') 
    
    np.save(os.path.join(save_output_path, f'mispredicted_X_scf_{data_type}.npy'), desired_feature_output)
    
    print('save for ', data_type)    
    return desired_feature_output

    

def feature_extraction(subject, dataset_name= ''):

    # model = subject
    model_object = get_model_object(subject, dataset_name)

    print('start feature extraction')
    
    # inputtrain, inputtest = vgg16_features_GD(model_object.dataset, model_object)

    # print(inputtrain.shape)
    # print(inputtest.shape)
    inputtrain = []
    inputtest = []

    data_for_clustering = {}
    ep = 0
    # try:
    x_mis, tt , tst , mis_index, mis_tindex = mis(inputtest , inputtrain, subject=subject, epoch=ep, dataset_name=dataset_name)
    print(f'epoch_{ep} done!')
    data_for_clustering[f'epoch_{ep}'] = {'x_mis': x_mis, 
                                        'y_actual_test_and_train': tt,
                                        'y_predicted_test_and_train':tst,
                                        'mis_test_index':mis_index,
                                        'mis_train_index':mis_tindex}
    # except:
    #     print(f'The file of the epoch {ep} not found :( , please check mispredicteds indeces of test set')
    
    
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
   
    subject = 'inceptionV3_imagenet'
    dataset_name = 'imagenet'
    shifted_test_input = 0
    use_last_epoch = True
    
    
    # for subject,last_epoch in zip(subjects, use_last_epoch):
        
    start_time = datetime.now()
    
    # get_mispredictions_input(subject, dataset_name=dataset_name)
    print('all mispredicted input from trian and test set collected and saved!')
    feature_extraction(subject, dataset_name=dataset_name)
    
    end_time = datetime.now()
    with open('cost_of_TEASMA.txt', 'a') as file:
        file.write(f"Time for extract misprediction input and features of subject {subject} dataset {dataset_name} is = {end_time - start_time}.\n")
        
    

