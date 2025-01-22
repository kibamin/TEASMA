import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from cnn_operator import *
from keras.models import load_model
import argparse
from utils import summary_model, color_preprocessing, model_predict
from termcolor import colored
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
import gc #garbage collector
import keras.backend as K
# from progressbar import *
import csv
# from keras.utils import np_utils
import properties_handler
import time
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import json
import pandas as pd

import logging

from datetime import datetime
import pickle

import tensorflow_datasets as tfds





logs_path = "app_logs"
os.makedirs(logs_path, exist_ok=True)

class MyLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        log_file=f'{name}.log'
        file_handler = logging.FileHandler(os.path.join(logs_path, log_file))
        file_handler.setLevel(logging.DEBUG)  # Set the log level for the file handler
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.addHandler(file_handler)


logger = MyLogger('generator.py') 


root = os.path.dirname(os.path.abspath(__file__)) 
import sys

sys.path.insert(1, os.path.join(root, '..','..'))
from preprocess_data import *
print(root)


def cnn_mutants_generation(ori_model, operator, ratio, standard_deviation=0.5):
    """

    :param ori_model:
    :param operator:
    :param ratio:
    :param standard_deviation:
    :return:
    """
    if operator < 5:
        cnn_operator(ori_model, operator, ratio, standard_deviation) # all operators of this condition checked
    else:
        new_model = cnn_operator(ori_model, operator, ratio, standard_deviation)
        return new_model
    return ori_model


def generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        help="original model path")
    
    parser.add_argument("--subject_name", "-subject_name",
                        type=str,
                        help="model name for example (lenet5, cifar10, ...)")
                        
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        help="mnist or cifar-10 or other dataset")
    parser.add_argument("--operator", "-operator",
                        type=int,
                        help="mutator (-1 means all operators)")
    parser.add_argument("--ratio", "-ratio",
                        type=float,
                        help="mutation ratio")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="mutants save path")
    parser.add_argument("--threshold", "-threshold",
                        type=float,
                        default=0.9,
                        help="ori acc * threshold must > mutants acc")
    parser.add_argument("--num", "-num",
                        type=int,
                        default=1,
                        help="mutants number")
    parser.add_argument("--standard_deviation", "-standard_deviation",
                        type=float,
                        default=0.5,
                        help="standard deviation for gaussian fuzzing")
    
    
    args = parser.parse_args()
    model_path = args.model_path
    subject = args.subject_name
    data_type = args.data_type
    operator = args.operator
    ratio = args.ratio
    save_path = args.save_path
    threshold = args.threshold
    num = args.num
    standard_deviation = args.standard_deviation


    print('=-===================')
    # load data
    if data_type == 'mnist':
        CLIP_MAX = 0.5
        ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
        (img_rows, img_cols) = (28, 28)
        num_classes = 10
        
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
        
        logger.info(y_test)   
    elif data_type == 'cifar10':
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

            # logger.info(y_test.flatten())
    elif data_type == 'fashion_mnist':
        CLIP_MAX = 0.5

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
        logger.info(y_test)   
    elif data_type == 'SVHN':
        # object = lenet5_SHVN.lenet5_SVHN()
        # (x_train, y_train), (x_test, y_test) = object.load_model()
        
        train_raw = loadmat(os.path.join(root,'..','..','datasets','SVHN', 'train_32x32.mat'))
        test_raw = loadmat(os.path.join(root,'..','..','datasets','SVHN', 'test_32x32.mat'))
        
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
        
        logger.info(y_test)
        

    elif data_type in ['amazon', 'office31_mix']:
        logger.info(f'read {data_type} dataset')
        train_raw = loadmat(os.path.join(root,'..','..','datasets',data_type, 'train.mat'))
        test_raw = loadmat(os.path.join(root,'..','..','datasets',data_type, 'diff_test.mat'))
        
        x_train = np.array(train_raw['X'])
        x_test = np.array(test_raw['X'])
        y_train = train_raw['y']
        y_test = test_raw['y']
        
        y_train = np.argmax(y_train,axis=1)
        y_test = np.argmax(y_test,axis=1)        
        
    elif data_type in ['caltech256']:
        
        logger.info(f'read {data_type} dataset')
        with open(os.path.join(root,'..','..','datasets',data_type,'train_test_indexes.pkl'),'rb') as f:
            dataset_indeces_dict = pickle.load(f)
        train_indeces = dataset_indeces_dict['train_indexes']
        all_train_ds, _, _, _, num_classes, _, _ = load_data(os.path.join(root, '..','..', '..','finetuning','data','256_ObjectCategories'), loaded_train_indeces=train_indeces, loaded_test_indeces=train_indeces) # train: 80 test 20 /  train: 50 test 50 / train: 20 test 80

        
        s = datetime.now()
        x_train, y_train = convert_to_numpy(all_train_ds)
        e = datetime.now()
        print('time to convert to numpy is:', e-s)
    
    elif data_type in ['caltech256_8020']:
        
        logger.info(f'read {data_type} dataset')
        
        with open(os.path.join(root,'..', '..', 'datasets', data_type, 'train_test_x_y.pkl'),'rb') as f:
                dataset_path_dict = pickle.load(f)

        train_x = dataset_path_dict['train_x']
        train_y = dataset_path_dict['train_y']
        test_x = dataset_path_dict['test_x']
        test_y = dataset_path_dict['test_y']
        num_classes = dataset_path_dict['num_classes']
        
        all_train_ds, test_ds= stratified_load_data_by_path_information(train_x_path=train_x, train_y_encoded=train_y, test_x_path=test_x, test_y_encoded=test_y) # train: 80 test 20 /  train: 50 test 50 / train: 20 test 80

        print('loading caltech256_8020 dataset')
        
        s = datetime.now()
        x_train, y_train = convert_to_numpy(all_train_ds)
        e = datetime.now()
        print('time to convert to numpy is:', e-s)
    
    
    elif data_type in ['office31']:
        
        logger.info(f'read {data_type} dataset')
        
        with open(os.path.join(root,'..', '..', 'datasets', data_type, 'train_test_x_y.pkl'),'rb') as f:
                dataset_path_dict = pickle.load(f)

        train_x = dataset_path_dict['train_x']
        train_y = dataset_path_dict['train_y']
        test_x = dataset_path_dict['test_x']
        test_y = dataset_path_dict['test_y']
        num_classes = dataset_path_dict['num_classes']
        
        all_train_ds, test_ds = stratified_load_data_by_path_information(train_x_path=train_x, train_y_encoded=train_y, test_x_path=test_x, test_y_encoded=test_y) # train: 80 test 20 /  train: 50 test 50 / train: 20 test 80

        print('loading office31 dataset')
        
        s = datetime.now()
        x_train, y_train = convert_to_numpy(all_train_ds)
        e = datetime.now()
        print('time to convert to numpy is:', e-s)
        
    # elif data_type in ['brightness_500', 'contrast_500', 'rotation_500', 'scale_500', 'shear_500', 'translation_500']:
    #     (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    #     x_train = x_train.astype('float32') / 255
    #     # x_test = x_test.astype('float32') / 255
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #     # x_test -= x_train_mean
        
        
    #     ### select random from target dataset
    #     transformer, selected_size = data_type.split('_')[0], data_type.split('_')[1]
    #     selected_target_data_path = os.path.join(root,'..', '..', 'datasets', 'cifar10_transformed', 'selected_target', transformer, f'size_{selected_size}', 'selected_train.npy')
        
    #     selected_target_dic = np.load(selected_target_data_path, allow_pickle=True).item()
    #     selected_target_train_x = selected_target_dic['x_train']
    #     selected_target_train_y = selected_target_dic['y_train']
    
        
        
    #     y_train = tf.keras.utils.to_categorical(y_train, 10)
        
    #     ## training set source + selected training set target
    #     x_train = np.concatenate([x_train, selected_target_train_x])
    #     y_train = np.concatenate([y_train, selected_target_train_y])
        
    #     y_train = np.argmax(y_train, axis=1)
        
    elif data_type in ['cifar10_brightness_5000', 'cifar10_contrast_5000', 'cifar10_rotation_5000', 'cifar10_scale_5000', 'cifar10_shear_5000', 'cifar10_translation_5000']:
        transformer, selected_size = data_type.split('_')[1], data_type.split('_')[2]
        selected_target_data_path = os.path.join(root,'..', '..', 'datasets', 'cifar10_transformed', 'selected_target', transformer, f'size_{selected_size}', 'selected_train.npy')
        
        selected_target_dic = np.load(selected_target_data_path, allow_pickle=True).item()
        selected_target_train_x = selected_target_dic['x_train']
        selected_target_train_y = selected_target_dic['y_train']
    
        x_train = selected_target_train_x
        y_train = selected_target_train_y
        
        y_train = np.argmax(y_train, axis=1)
        
    elif data_type in ['mnist_brightness_6000', 'mnist_contrast_6000', 'mnist_rotation_6000', 'mnist_scale_6000', 'mnist_shear_6000', 'mnist_translation_6000']:
        transformer, selected_size = data_type.split('_')[1], data_type.split('_')[2]
        selected_target_data_path = os.path.join(root,'..', '..', 'datasets', 'mnist_transformed', 'selected_target', transformer, f'size_{selected_size}', 'selected_train.npy')
        
        selected_target_dic = np.load(selected_target_data_path, allow_pickle=True).item()
        selected_target_train_x = selected_target_dic['x_train']
        selected_target_train_y = selected_target_dic['y_train']
    
        x_train = selected_target_train_x
        y_train = selected_target_train_y
        
        y_train = np.argmax(y_train, axis=1)

    
    ## after revised paper
    
    elif data_type == 'cifar100':
        if 'resnet' in subject:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            
            x_train = x_train.astype('float32')
            
            # data preprocessing (it is necessary based on pretrained model here is ResNet)
            x_train = tf.keras.applications.resnet.preprocess_input(x_train)
            # x_test = tf.keras.applications.resnet.preprocess_input(x_test)    
    
    
    elif data_type == 'imagenet':
        data_dir = '/home/kiba/scratch/datasets/imagenet/'
        write_dir = '/home/kiba/scratch/tf-imagenet-dirs'

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


        def preprocess_image(image, label, subject):
            '''
            This message will be only logged once.
            98/98 [==============================] - 530s 5s/step - loss: 1.0475 - accuracy: 0.7505
            validation Accuracy: 0.750540018081665
            '''
            # Resize the smaller side to 256 while maintaining aspect ratio
            initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
            ratio = 256.0 / tf.reduce_min(initial_shape)
            new_shape = tf.cast(initial_shape * ratio, tf.int32)
            image = tf.image.resize(image, new_shape)
            # Crop the central 224x224
            if 'resnet50' in subject:
                image = tf.image.resize_with_crop_or_pad(image, 224, 224)
                image = tf.keras.applications.resnet50.preprocess_input(image)
                
            else: ## inceptionV3
                image = tf.image.resize_with_crop_or_pad(image, 299, 299)
                image = tf.keras.applications.inception_v3.preprocess_input(image) ## 5 min with batch size 512
                
            return image, label

        
        train_data = train_data.map(preprocess_image).batch(512)
        x_train = train_data
        y_train = None
        
    data = {}
    data['root'] = os.path.dirname(os.path.abspath(__file__)) 
    data['subject'] = subject
    properties_handler.write_properties(data)

    TEASMA_root_path = os.path.join(root,'..','..',)
    
    
    if operator == -1: # all operators
        operators = list(range(8))
    else:
        operators = [operator]
    
    mutant_predictions_dir_path = os.path.join(TEASMA_root_path, 'mutants', 'mutant_prediction_outputs', subject, 'train_set')
    if not os.path.exists(mutant_predictions_dir_path):
        os.makedirs(mutant_predictions_dir_path)   

    model = load_model(model_path)
    logger.info(model_path)
    ori_acc, original_output_probability = model_predict(model, x_train, y_train)
    np.save(os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs'), original_output_probability) 
    logger.info(ori_acc)
    
    threshold = ori_acc * threshold
        
    mutant_predictions_dic = {}
    equivalent_mutants = []

    time_dictionary = {'mutant':[], 'generation_time':[], 'execution_time':[], 'generation_time_process (sec)':[], 'mutant_status':[]}
    
    total_start_time = datetime.now()
    
    for operator in operators:
        logger.info(f'operator: {operator} ')
        if operator in [5,6,7] and 'resnet' in model_path: ## these operators are not applicable for resnet structure
            logger.debug(f'The operator number {operator} is not applicable on a model include Add() layer')
            continue
        
        model = load_model(model_path)
        
       
        weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)

        print(colored("operator: %s" % cnn_operator_name(operator), 'blue'))
        print(colored("ori acc: %f" % ori_acc, 'blue'))
        print(colored("threshold acc: %f" % threshold, 'blue'))

    
        if operator == 0 or operator == 1:
            print("total weights: ", weight_count)
            print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)
        elif 2 <= operator <= 4:
            print("total neuron: ", neuron_count)
            print("process neuron num: ", int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1)

        # mutants generation
        i = 1
        # start_time = time.clock()
        start_time = time.process_time()
        count = 1
        while i <= num:
            if count > 100: # if number of attempts > 100 it is obviouse there is now mutant satisfy the thereshold accuracy
                break
            if i != 1: # In the first time it doesn't load original model and in the rest of the while iteration it loads original model
                model = load_model(model_path)
            # generate mutant
            stime = time.process_time()
            g_start_time = datetime.now()
            new_model = cnn_mutants_generation(model, operator, ratio, standard_deviation) # TODO: this function generate mutant based on operator's number
            g_end_time = datetime.now()
            
            etime = time.process_time()
            
            
           
            
            
            
            print(f'elapsed time for mutation operator {cnn_operator_name(operator)} number {count}: {(etime - stime)/60}')
            # test mutant
            # operator = 5 (Layer Romve(LR) or LD)  this operator maybe returns False because it could not find any layers which the input shape and output shape be same
            if type(new_model) == bool:
                time_dictionary['generation_time'].append(g_end_time-g_start_time)
                time_dictionary['generation_time_process (sec)'].append(etime - stime)
                time_dictionary['mutant'].append(cnn_operator_name(operator)+str(count))
                time_dictionary['mutant_status'].append(False)
                time_dictionary['execution_time'].append(0)
                break

            if type(new_model) == dict:
                for name_of_mutant, mutant_model in new_model.items():
                    time_dictionary['generation_time'].append(g_end_time-g_start_time)
                    time_dictionary['generation_time_process (sec)'].append(etime - stime)
                    time_dictionary['mutant'].append('total_'+cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5")
                    
                    ex_start_time = datetime.now()
                    new_acc, mutant_output_probability = model_predict(mutant_model, x_train, y_train)
                    ex_end_time = datetime.now()
                    time_dictionary['execution_time'].append(ex_end_time - ex_start_time)
                    
                    
                    
                    print(name_of_mutant)
                    print('acc of mutant:', new_acc)
                    if new_acc < threshold: # a generated mutant is acceptable as mutant if mutant_acc > ori_acc * threshold (I think it want to kill easily - trivial) [those mutants with accuracy higher than the threshold 0.9(-threshold)is stored in the foldermutants(-savepath).]
                        count += 1 # number of times that couldn't pass threshold 0.9
                        time_dictionary['mutant_status'].append(False)
                        
                        print(colored("ori acc: %f" % ori_acc, 'red'))
                        print(colored("threshold acc: %f" % threshold, 'red'))
                        print(colored("mutant acc: %f" % new_acc, 'red'))
                        K.clear_session()
                        del model
                        # del new_model
                        gc.collect()
                        model = load_model(model_path) 
                        continue
                    # save each mutant
                    are_equivalent = np.array_equal(np.argmax(mutant_output_probability, axis=1), np.argmax(original_output_probability, axis=1))
                    if are_equivalent:
                        equivalent_mutants.append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5")
                    
                    mutant_predictions_dic[cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5"] = mutant_output_probability
                    
                    
                    save_dir_path = os.path.join(save_path, str(ratio))
                    final_path = os.path.join(save_dir_path , cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5" )
                    mutant_model.save(final_path)
                    i = num+1
                    time_dictionary['mutant_status'].append(True)
                    

            else:
                time_dictionary['generation_time'].append(g_end_time-g_start_time)
                time_dictionary['generation_time_process (sec)'].append(etime - stime)
                ex_start_time = datetime.now()
                new_acc, mutant_output_probability = model_predict(new_model, x_train, y_train)
                ex_end_time = datetime.now()
                time_dictionary['execution_time'].append(ex_end_time - ex_start_time)
                time_dictionary['mutant'].append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5")
                
                
                print('acc of mutant:', new_acc)
                if new_acc < threshold: # a generated mutant is acceptable as mutant if mutant_acc > ori_acc * threshold (I think it want to kill easily) [those mutants with accuracy higher than the threshold0.9(-threshold)isstored in the foldermutants(-savepath).]
                    count += 1 # number of times that couldn't pass threshold 0.9
                    time_dictionary['mutant_status'].append(False)
                    print(colored("ori acc: %f" % ori_acc, 'red'))
                    print(colored("threshold acc: %f" % threshold, 'red'))
                    print(colored("mutant acc: %f" % new_acc, 'red'))
                    K.clear_session()
                    del model
                    del new_model
                    gc.collect()
                    model = load_model(model_path) 
                    continue
                
                are_equivalent = np.array_equal(np.argmax(mutant_output_probability, axis=1), np.argmax(original_output_probability, axis=1))
                if are_equivalent:
                    equivalent_mutants.append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5")
                        
                mutant_predictions_dic[cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5"] = mutant_output_probability
                # save each mutant
                save_dir_path = os.path.join(save_path, str(ratio))
                final_path = os.path.join(save_dir_path , cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5" )
                
                new_model.save(final_path)
                time_dictionary['mutant_status'].append(True)

            # p_bar.update(int((i / num) * 100))
            i += 1
            # count += 1
            K.clear_session()
            del model
            del new_model
            gc.collect()
        
        # p_bar.finish()
        elapsed = (time.process_time() - start_time)
        # elapsed = (time.clock() - start_time)
        print("running time: ", elapsed)
        print('number of mutants that filter due to tiriviality', count)
        
        try:
            if not os.path.exists('cost_of_mutant_generation'):
                os.makedirs('cost_of_mutant_generation')
        except:
            print('directory exists!')
            pass
        
        
        with open(os.path.join('cost_of_mutant_generation', f'cost_of_TEASMA_{subject}.txt'), 'a') as file:
            
            file.write(f"Time for generating mutant of operator number {operator} named {cnn_operator_name(operator)} for {subject} is = {elapsed}.\n")
            file.write(f"'number of mutants that filter due to tiriviality (acc < 90% of orginal accuracy)' = {elapsed}.\n")
            file.write(f"'number of tries' = {i}.\n")
        

        # extra_output_path = 'extra_outputs'
        # with open(os.path.join(extra_output_path,f'generate_mutants.csv'), 'a') as f1:
        #     writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        #     # writer.writerow([str(data_name), str(test_suite_name), str(basic_ms), str(dc_ms), str(FDR), str(time_taken)])
        #     writer.writerow([cnn_operator_name(operator), str(ratio), str(i-1)+'/'+str(count-1), str(elapsed)])

    total_end_time = datetime.now()
    
    equivalent_mutants_dir_path = os.path.join(TEASMA_root_path, 'mutants', 'equivalent_mutants', subject)
    try:
        if not os.path.exists(equivalent_mutants_dir_path):
            os.makedirs(equivalent_mutants_dir_path)
    except:
        print('file exist')
        
    equivalent_mutants_file = os.path.join(equivalent_mutants_dir_path,'equivalents.json')
    with open(equivalent_mutants_file,'w') as f:
        json.dump(equivalent_mutants, f) 


    if operator == -1:
        np.save(os.path.join(mutant_predictions_dir_path,'prediction_outputs'), mutant_predictions_dic) 
    else:
        np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{str(operator)}'), mutant_predictions_dic) 
    
    
    print(time_dictionary)
    df = pd.DataFrame(time_dictionary)
    cost_file = os.path.join(equivalent_mutants_dir_path,'time_cost.csv')
    df.to_csv(cost_file)
    # with open(cost_file,'w') as f:
    #     json.dump(time_dictionary, f) 
    

    print('elapsed time is :', total_end_time - total_start_time)

if __name__ == '__main__':
    generator()
    logger.info('End of mutation generator')

# python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2

# python generator.py --model_path models/cifar10.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2
