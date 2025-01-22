import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import gc #garbage collector
import keras.backend as K
from tensorflow.keras.backend import clear_session

from lenet5_mnist import lenet5_mnist # S1
from cifar10_conv import cifar10_conv # S2
from resnet20_cifar10 import resnet20_cifar10 # S3
from lenet4 import lenet4 # S4
from lenet5_SVHN import lenet5_SVHN # S5
from vgg16_SVHN import vgg16_SVHN # S6
from resnet152_cifar100 import resnet152_cifar100 # S7



 

from datetime import datetime
import subprocess
import re


root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..') # root of project


def get_gpu_memory():
    """Returns the current GPU memory usage by querying the nvidia-smi tool."""

    # Execute the command 'nvidia-smi' to get GPU details
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Use regex to parse the output for memory usage
    memory_usage = re.findall(r'\dMiB /  \dMiB', output)

    return memory_usage

def execute_mutants(subject, dataset_name='', dataset_part = 'test_set'):
    print(subject)
    ratio = 0.01
    
    num_classes = 10
    
    if subject == 'lenet4':
        model_object = lenet4()
        
    elif subject == 'lenet5_SVHN':
        model_object = lenet5_SVHN()
    
    elif subject == 'cifar10':
        model_object = cifar10_conv()
        
    elif subject == 'resnet20_cifar10':
        model_object = resnet20_cifar10()
    
    elif subject == 'vgg16_SVHN':
        model_object = vgg16_SVHN()
    
    elif subject == 'lenet5_mnist':
        model_object = lenet5_mnist()
    
    elif subject == 'resnet152_cifar100':
        model_object = resnet152_cifar100()
 
        
    model = model_object.load_original_model()
    
    if dataset_part == 'test_set':
        (_, _ ), (X, _) = model_object.load_dataset()
    
    else:
        (X, _ ), (_, _) = model_object.load_dataset()
    
    
    if hasattr(model_object, 'number_of_classes'):
        num_classes = model_object.number_of_classes
        print(f'number of classes is {num_classes}')
    else:
        print("the model object has not attribute number_of_classes so the default number of classes setted (default is 10).")
    
   
    
    mutant_predictions_dir_path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', subject, dataset_part)
    if not os.path.exists(mutant_predictions_dir_path):
        os.makedirs(mutant_predictions_dir_path)
    
    
    
    if model:
        predicted_y_by_original = model.predict(X, verbose=1)
        np.save(os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs'), predicted_y_by_original) 
        
    K.clear_session()
    del model
    clear_session()
    gc.collect()
        
    mutant_models_path = os.path.join(root,'mutants',subject,str(ratio))
    print('mutants path:', mutant_models_path)
    all_mutants_file_name = os.listdir(mutant_models_path)
    print('number of mutants:', len(all_mutants_file_name))

    
    mutant_predictions_dic = {}
    for mutant_file_name in all_mutants_file_name:
        mutant_model_path = os.path.join(mutant_models_path, mutant_file_name)
        print(mutant_model_path)
        
        model = load_model(mutant_model_path)
        
        predicted_y = model.predict(X, batch_size=32, verbose=0)
        print(predicted_y.shape)
        
        print('before clearing')
        gpu_memory = get_gpu_memory()
        print("Current GPU Memory Usage:")
        print(gpu_memory)
        
        K.clear_session()
        del model
        clear_session()
        gc.collect()
        
        print('after clearing')
        gpu_memory = get_gpu_memory()
        print("Current GPU Memory Usage:")
        print(gpu_memory)
        

        mutant_predictions_dic[mutant_file_name] = predicted_y
    
    
            
    np.save(os.path.join(mutant_predictions_dir_path,'prediction_outputs'), mutant_predictions_dic) 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        help="the name of subject [ 'cifar10', 'lenet4', 'lenet5_SVHN', ...]")

    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        default='',
                        help="the name of dataset [ 'cifar10', 'mnist', 'SVHN', ...]")
    
    parser.add_argument("--dataset_part", "-dataset_part",
                        type=str,
                        default='test_set',
                        help="train_set or test_set")
    
    args = parser.parse_args()
    subject = args.subject
    dataset = args.dataset
    dataset_part = args.dataset_part
    
    subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN', 'resnet152_cifar100'] # for inceptionV3_imagenet use mutant_testing_for_imagenet.py
    
    start = datetime.now()
    execute_mutants(subject, dataset, dataset_part)
    end = datetime.now()
    print('elapsed time for mutant executing:', end - start)
    
    
