import os
import json
import numpy as np
import argparse
from datetime import datetime



root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..') # root of project


### load predictions output of mutants and original model on total training set
def load_predictions_of_mutants_for_train_set(subject):
    path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'train_set', 'prediction_outputs.npy')
    mutants_predictions_dict = np.load(path, allow_pickle=True).item()
    
    # this is not dictionary
    path_for_original_model = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'train_set', 'original_model_prediction_outputs.npy')
    original_model_prediction_outputs = np.load(path_for_original_model)

    return original_model_prediction_outputs, mutants_predictions_dict

### load predictions output of mutants and original model on total test set
def load_predictions_of_mutants_for_test_set(subject):
    global mutants_predictions_dict
    path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', 'prediction_outputs.npy')
    mutants_predictions_dict = np.load(path, allow_pickle=True).item()
    
    # this is not dictionary
    global original_model_prediction_outputs
    path_for_original_model = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', 'original_model_prediction_outputs.npy')
    original_model_prediction_outputs = np.load(path_for_original_model)
    
    return original_model_prediction_outputs, mutants_predictions_dict






def find_equivalent_mutants(subject, dataset_name=''):
    original_model_prediction_outputs, mutants_predictions_dict = load_predictions_of_mutants_for_train_set(subject)
    
    num_classes = original_model_prediction_outputs.shape[1]
    print('num_of_classes is : ', num_classes)
    
    y_train_predicted = np.argmax(original_model_prediction_outputs, axis=1) # predicted by original model
    equivalent_mutants_list = []
    for mutant_file_name, predicted_y in mutants_predictions_dict.items():
        mispredicted_classes = [0]*num_classes

        for y_actual, y_predict in zip(y_train_predicted, np.argmax(predicted_y, axis=1)):
            if y_actual != y_predict:
                mispredicted_classes[y_actual] += 1

        if sum(mispredicted_classes) == 0:
            print('@@@ ',mutant_file_name)
            equivalent_mutants_list.append(mutant_file_name)
            
            
    mutant_predictions_dir_path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', subject, 'train_set')
    if not os.path.exists(mutant_predictions_dir_path):
        os.makedirs(mutant_predictions_dir_path)   
            
            
    equivalent_mutants_dir_path = os.path.join(root, 'mutants', 'equivalent_mutants', subject)
    if not os.path.exists(equivalent_mutants_dir_path):
        os.makedirs(equivalent_mutants_dir_path)
    
    equivalent_mutants_file = os.path.join(equivalent_mutants_dir_path,'equivalents.json')
    with open(equivalent_mutants_file,'w') as f:
        json.dump(equivalent_mutants_list, f) 
        print(f'equivalent mutants saved for subject: {subject}')
        print(f'the number of equivalents of subject {subject} is {len(equivalent_mutants_list)}')
        
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        help="the name of subject ['lenet5', 'cifar10', 'lenet4', 'lenet5_SVHN', ...]")

    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        default='',
                        help="the name of dataset ['amazon', 'webcam', 'deslar', 'merged_office31', ...]")
    
    args = parser.parse_args()
    subject = args.subject
    dataset = args.dataset
    
    subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10']
    
    start = datetime.now()
    
    find_equivalent_mutants(subject, dataset)
    
    end = datetime.now()
    print('elapsed time for mutant filtering:', end - start)
    with open('cost_of_TEASMA.txt', 'a') as file:
        file.write(f"Time for mutant filtering of subject {subject} is = {end - start}.\n")
        
    
