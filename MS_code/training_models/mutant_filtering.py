import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import argparse


from lenet5_mnist import lenet5_mnist # S1
from cifar10_conv import cifar10_conv # S2
from resnet20_cifar10 import resnet20_cifar10 # S3
from lenet4 import lenet4 # S4
from lenet5_SVHN import lenet5_SVHN # S5
from vgg16_SVHN import vgg16_SVHN # S6
from resnet152_cifar100 import resnet152_cifar100 # S7




from datetime import datetime



root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..') # root of project


def find_equivalent_mutants(subject, dataset_name=''):
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
    (X_train, _ ), (_, _) = model_object.load_dataset()
    
    if hasattr(model_object, 'number_of_classes'):
        num_classes = model_object.number_of_classes
        print(f'number of classes is {num_classes}')
    else:
        print("the model object has not attribute number_of_classes so the default number of classes setted (default is 10).")
    
    mutant_predictions_dir_path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', subject, 'train_set')
    if not os.path.exists(mutant_predictions_dir_path):
        os.makedirs(mutant_predictions_dir_path)   
         
    if model:
        print('start predict input')
        predicted_y_by_original = model.predict(X_train, verbose=1)
        np.save(os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs'), predicted_y_by_original) 
        print('end predict input')
        
        print('start argmax')
        y_train_predicted = np.argmax(predicted_y_by_original, axis=1)
        print('end of argmax')
        
    mutant_models_path = os.path.join(root,'mutants',subject,str(ratio))
    print('mutants path:', mutant_models_path)
    all_mutants_file_name = os.listdir(mutant_models_path)
    print('number of mutants:', len(all_mutants_file_name))

    mutant_predictions_dic = {}
    equivalent_mutants_list = []
    for mutant_file_name in all_mutants_file_name:
        mutant_model_path = os.path.join(mutant_models_path, mutant_file_name)
        print(mutant_model_path)
        
        mispredicted_classes = [0]*num_classes
        mutant_model = load_model(mutant_model_path)
        
        predicted_y = mutant_model.predict(X_train, verbose=0)
        mutant_predictions_dic[mutant_file_name] = predicted_y
        
        for y_actual, y_predict in zip(y_train_predicted, np.argmax(predicted_y, axis=1)):
            if y_actual != y_predict:
                mispredicted_classes[y_actual] += 1

        if sum(mispredicted_classes) == 0:
            print('@@@ ',mutant_file_name)
            equivalent_mutants_list.append(mutant_file_name)
            
            
    equivalent_mutants_dir_path = os.path.join(root, 'mutants', 'equivalent_mutants', subject)
    if not os.path.exists(equivalent_mutants_dir_path):
        os.makedirs(equivalent_mutants_dir_path)
    
    equivalent_mutants_file = os.path.join(equivalent_mutants_dir_path,'equivalents.json')
    with open(equivalent_mutants_file,'w') as f:
        json.dump(equivalent_mutants_list, f) 
        print(f'equivalent mutants saved for subject: {subject}')
        print(f'the number of equivalents of subject {subject} is {len(equivalent_mutants_list)}')
        
    np.save(os.path.join(mutant_predictions_dir_path,'prediction_outputs'), mutant_predictions_dic) 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        help="the name of subject ['lenet5', 'cifar10', 'lenet4', 'lenet5_SVHN', ...]")

    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        default='',
                        help="the name of dataset ['cifar10', 'cifar100', 'mnist', 'SVHN', ...]")
    
    args = parser.parse_args()
    subject = args.subject
    dataset = args.dataset
    
    subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN', 'resnet152_cifar100']
    
    start = datetime.now()
    
    find_equivalent_mutants(subject, dataset)
    
    end = datetime.now()
    print('elapsed time for mutant filtering:', end - start)
    with open('cost_of_TEASMA.txt', 'a') as file:
        file.write(f"Time for mutant filtering of subject {subject} is = {end - start}.\n")
        
    
