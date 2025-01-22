import os
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
import csv
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from keras.models import load_model
# import keras
# import tensorflow as tf
# import keras.backend as K
import constants as const
# from keras.datasets import mnist, cifar10
import copy


import logging

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


logger = MyLogger('runner.py') 



global base_path_of_outputs, num_classes, mutants_predictions_dict, original_model_prediction_outputs
base_path_of_outputs = ''
mutants_predictions_dict = {}




root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..') # root of project



### load predictions output of mutants and original model on total training set
def load_predictions_of_mutants_for_train_set(subject):
    global mutants_predictions_dict
    path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'train_set', 'prediction_outputs.npy')
    mutants_predictions_dict = np.load(path, allow_pickle=True).item()
    
    # this is not dictionary
    global original_model_prediction_outputs
    path_for_original_model = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'train_set', 'original_model_prediction_outputs.npy')
    original_model_prediction_outputs = np.load(path_for_original_model)




### load predictions output of mutants and original model on total test set
def load_predictions_of_mutants_for_test_set(subject):
    global mutants_predictions_dict
    path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', 'prediction_outputs.npy')
    mutants_predictions_dict = np.load(path, allow_pickle=True).item()
    
    # this is not dictionary
    global original_model_prediction_outputs
    path_for_original_model = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', 'original_model_prediction_outputs.npy')
    original_model_prediction_outputs = np.load(path_for_original_model)



### load predictions output of mutants and original model on shifted test set
def load_predictions_of_mutants_for_shifted_test_set(subject, shifted_dataset):
    global mutants_predictions_dict
    path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', shifted_dataset, 'prediction_outputs.npy')
    mutants_predictions_dict = np.load(path, allow_pickle=True).item()
    
    # this is not dictionary
    global original_model_prediction_outputs
    path_for_original_model = os.path.join(root, 'mutants', 'mutant_prediction_outputs', f'{subject}', 'test_set', shifted_dataset, 'original_model_prediction_outputs.npy')
    original_model_prediction_outputs = np.load(path_for_original_model)


def load_test_suite(path='', dataset = 'mnist'):
    CLIP_MAX = 0.5
    # load test suites with different sizes like 100,300,500,1000,...
    if path: 
        f = np.load(path, allow_pickle=True).item()
        x_test, y_test, indeces = f['x_test'], f['y_test'], f['indices']
    
    # # load strong dataset (for mnist)
    # else: #
    #     ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    #     indeces = list(range(len(x_train))) + list(range(len(x_test)))
    
    # if dataset == 'mnist':
    #     (img_rows, img_cols) = (28, 28)
    #     num_classes = 10
        
    #     if K.image_data_format() == 'channels_first':
    #         x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     else:
    #         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #     x_test = x_test.astype('float32')
    #     x_test = (x_test / 255) - (1 - CLIP_MAX)    

    # elif dataset == 'cifar10':
    #     x_test = x_test.astype("float32")
    #     x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
        
    # elif dataset == 'fashion_mnist':
    #     x_test = x_test.astype("float32")
    #     x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
    
    # elif dataset == 'SVHN':
    #     pass
 

    return (x_test, y_test, indeces)





#-------------#
#  E1 and E2  #
#-------------#
# mutation score using killed classes /// Grouping \\\
def mutation_score_of_E1_E2_by_grouping(test_suite_name, root_killed_classes_path, group, subject=None):
    test_suite_killed_classes_path = os.path.join(root_killed_classes_path, test_suite_name)
    killed_classes_df = pd.read_csv(test_suite_killed_classes_path)
    filtered_dfs = []
    
    # filter equivalent 
    list_of_equivalent_mutants = get_equivalent_mutants()
    killed_classes_df = killed_classes_df[~killed_classes_df['mutant'].isin(list_of_equivalent_mutants)]

    
    for op in group:
        df = killed_classes_df[killed_classes_df['mutant'].str.contains(op+'_')]
        filtered_dfs.append(df)
        
    if len(group) :
        killed_classes_df = pd.concat(filtered_dfs)
    
        

    mutants = killed_classes_df['mutant'].values

    killed_classes_df = killed_classes_df['c0 c1 c2 c3 c4 c5 c6 c7 c8 c9'.split()]
    
    # E1
    killed_classes_df_for_E1 = killed_classes_df.clip(0,1)
    number_of_killed_classes = killed_classes_df_for_E1.sum().sum()

    number_of_classes = 10
    if len(mutants):
        E1_ms = number_of_killed_classes / (len(mutants)*number_of_classes)
    else:
        E1_ms = 0

    
    # E2
    killed_classes_df_for_E1 = killed_classes_df.sum(axis=1) # sum columns of a row
    killed_classes_df_for_E1 = killed_classes_df_for_E1.clip(0,1)
    number_of_killed_classes = killed_classes_df_for_E1.sum()


    if len(mutants):
        E2_ms = number_of_killed_classes / len(mutants)
    else:
        E2_ms = 0
        

    return E1_ms, E2_ms



def mutation_score_of_E3_by_grouping(test_suite_name, root_path_killing_score_of_each_mutant, group, subject):
    test_suite_killing_score_path = os.path.join(root_path_killing_score_of_each_mutant, test_suite_name)
    killed_classes_df = pd.read_csv(test_suite_killing_score_path)
    filtered_dfs = []
    
    if len(group) == 0:
        group = const.MUTATION_OPERATORS_FOR_SUBJECT[subject]
    for op in group:
        df = killed_classes_df[killed_classes_df['mutant'].str.contains(op+'_')]
        filtered_dfs.append(df)
    mutation_killing_score_df = pd.concat(filtered_dfs)
    
    if len(mutation_killing_score_df):
        E3_ms = mutation_killing_score_df['killing_score'].sum() / len(mutation_killing_score_df)
    else:
        E3_ms = 0
    
    return E3_ms
    
    
    
    
def remove_mispredicteds_inputs(test_suite_path, model, dataset_name):
    x_test, y_test, indeces = load_test_suite(test_suite_path, dataset_name)
# with tf.device('/gpu:0'):
    y_test_predict = model.predict(x_test, verbose=0)
    y_test_predict = np.argmax(y_test_predict, axis=1)

    correctly_predicted_indeces = []
    
    for i in range(len(x_test)):
        if y_test_predict[i] == y_test[i]:
            correctly_predicted_indeces.append(i)
    
    x_test_correctly = x_test[correctly_predicted_indeces]
    y_test_correctly = y_test[correctly_predicted_indeces]
        

    return x_test_correctly, y_test_correctly


## Grouping 
# E2
def mutation_score_basic_using_group(mutant_models_path, test_suite_name, root_killed_classes_path, group):
    test_suite_killed_classes_path = os.path.join(root_killed_classes_path, test_suite_name)
    killed_classes_df = pd.read_csv(test_suite_killed_classes_path)

    filtered_dfs = []
    for op in group:
        df = killed_classes_df[killed_classes_df['mutant'].str.contains(op+'_')]
        filtered_dfs.append(df)
    killed_classes_df = pd.concat(filtered_dfs)

    mutants = killed_classes_df['mutant'].values

    killed_classes_df = killed_classes_df['c0 c1 c2 c3 c4 c5 c6 c7 c8 c9'.split()]
    killed_classes_df = killed_classes_df.sum(axis=1) # sum columns of a row
    killed_classes_df = killed_classes_df.clip(0,1)
    number_of_killed_classes = killed_classes_df.sum()


    if len(mutants):
        ms = number_of_killed_classes / len(mutants)
    else:
        ms = 0
    

    return ms






# #------#
# #  E3  #
# #------#
# # deepmutation++ (this is from scratch)
# def mutation_score_deepmutationPP(mutant_models_path, x_test_suite, y_test_suite):
#     ms_of_each_mutant = []
#     all_mutants_file_name = os.listdir(mutant_models_path)
#     killed_test_input_t = {'mutant':[],'killing_score':[]} # for example: T0 with size 100 has t0, t1, t2, ..., t99 if all correctly predicted by original model
#     for mutant_file_name in all_mutants_file_name:
#         mutant_model_path = os.path.join(mutant_models_path, mutant_file_name)
#         mutant_model = load_model(mutant_model_path)
        
#         # operators_that_need_to_compile_model = ['LA']
#         # for op in operators_that_need_to_compile_model:
#         #     if op in mutant_file_name:
#         #         mutant_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate = 1), metrics=['accuracy'])
#         #         print(f'{mutant_file_name} compiled!')
#         #         break
#         with tf.device('/gpu:0'):
#             predicted_y = mutant_model.predict(x_test_suite, verbose=0)
#             predicted_y = np.argmax(predicted_y, axis=1)
#             num_killing_inputs = 0
#             for y_actual, y_predict in zip(y_test_suite, predicted_y):
#                     if y_actual != y_predict:
#                         num_killing_inputs += 1
        
#         mutant_ms = num_killing_inputs/len(x_test_suite)
        
#         ms_of_each_mutant.append(mutant_ms)
#         # print(sum(ms_of_each_mutant)/len(all_mutants_file_name))
    
#     total_ms = sum(ms_of_each_mutant)/len(all_mutants_file_name)
#     killed_test_input_t = {'mutant':all_mutants_file_name, 'killing_score':ms_of_each_mutant}

#     return total_ms, pd.DataFrame(killed_test_input_t)
        


#-----------------#
#  E1, E2 and E3  # %%%
#-----------------#
def mutation_score_of_E1_E2_E3(mutant_models_path, x_test_suite, y_test_suite, subject = '', original_model = None, test_suite_indeces = []):
    # num_classes = 10
    # if subject in ['mnist', 'cifar10', 'fashion_mnist']:
    #     num_classes = 10
    
    global num_classes, mutants_predictions_dict
    # print('number of classes for this dataset is:', num_classes)
    
    # if dataset_name == 'cifar10':
    #     y_test_suite = y_test_suite.reshape(-1)

    # if len(y_test_suite.shape) == 1:
    #     y_test_suite_categorical = keras.utils.to_categorical(y_test_suite, num_classes)

    if original_model:
        # predicted_y_by_original = original_model.predict(x_test_suite, verbose=0)
        
        predicted_y_by_original = original_model_prediction_outputs[test_suite_indeces]
        # y_test_suite = np.argmax(predicted_y_by_original, axis=1)
        
    
    
    E3_ms_of_each_mutant = [] # for E3
    killed_test_input_t = {'mutant':[],'killing_score':[]} # for example: T0 with size 100 has t0, t1, t2, ..., t99 if all correctly predicted by original model
    
    failed_mutants_array = [] 
    sum_killed_classes = 0
    number_of_satisfied_mutants = 0
    
    try:
        all_mutants_file_name = os.listdir(mutant_models_path)
    except:
        all_mutants_file_name = list(mutants_predictions_dict.keys())
        

    
    # filter equivalent 
    list_of_equivalent_mutants = get_equivalent_mutants()
    all_mutants_file_name = [mut for mut in all_mutants_file_name if mut not in list_of_equivalent_mutants]
    

    # datafram to save killed classes
    name_of_columns = ['mutant']
    for n in range(num_classes):
        name_of_columns.append(f'c{n}')
        
    # killed_classes_df = pd.DataFrame(columns='mutant c0 c1 c2 c3 c4 c5 c6 c7 c8 c9'.split())
    killed_classes_df = pd.DataFrame(columns=name_of_columns)
    

    for mutant_file_name in all_mutants_file_name:
        
        
        
        mispredicted_classes = [0]*num_classes

        '''
        Old version (the two bellow lines take more time to execute nearly 5 to 10 second)
        '''
        # mutant_model_path = os.path.join(mutant_models_path, mutant_file_name)
        # mutant_model = load_model(mutant_model_path)
        # predicted_y = mutant_model.predict(x_test_suite, verbose=0)
        
        '''
        New version: using prediction file that achieved before
        '''
        predicted_y_by_mutant = mutants_predictions_dict[mutant_file_name][test_suite_indeces]
        
        
        
        ### new and optimum code
        num_killing_inputs = 0
        
        if subject == 'inceptionV3_imagenet':
            for y_actual, y_predict in zip(predicted_y_by_original, predicted_y_by_mutant):
                if y_actual != y_predict:
                    num_killing_inputs += 1
                    mispredicted_classes[y_actual] += 1
        else:
            for y_actual, y_predict in zip(np.argmax(predicted_y_by_original, axis=1), np.argmax(predicted_y_by_mutant, axis=1)):
                if y_actual != y_predict:
                    num_killing_inputs += 1
                    mispredicted_classes[y_actual] += 1
        
        
        # E3
        mutant_ms = num_killing_inputs/len(test_suite_indeces)
        E3_ms_of_each_mutant.append(mutant_ms)
        
        # E1
        error_rate = num_killing_inputs / len(test_suite_indeces)
        # print('error_rate:', error_rate)

        
        if error_rate > const.ERROR_RATE_THRESHOLD:
            print('error_rate > 0.20 ::', error_rate, 'remove mutant')
            # failed_mutants_array.append(1) # save those mutants that removed by error rate condition 
            pass
        # if sum(mispredicted_classes) == 0:
        #     print('find equivalent mutant based on current test suite')
        #     pass 
        else:
            # print('mutant satisfied since of error rate')
            number_of_satisfied_mutants += 1
            # failed_mutants_array.append(0)
            killed_classes_df.loc[len(killed_classes_df)] = [mutant_file_name] + mispredicted_classes
            num_of_killed_classes = num_classes - mispredicted_classes.count(0)  # count the classes that predicted correctly and the subtract from the number of classes
            sum_killed_classes += num_of_killed_classes
        
            
        # ## TODO: added for different distribution
        # number_of_satisfied_mutants += 1
        # # failed_mutants_array.append(0)
        # killed_classes_df.loc[len(killed_classes_df)] = [mutant_file_name] + mispredicted_classes
        # num_of_killed_classes = num_classes - mispredicted_classes.count(0)  # count the classes that predicted correctly and the subtract from the number of classes
        # sum_killed_classes += num_of_killed_classes

    
    
    # E1 mutation score
    killed_classes_df.set_index('mutant')
    # mutation score function of deepmutation
    print('number of satisfied mutlants:', number_of_satisfied_mutants)
    if number_of_satisfied_mutants:
        E1_ms = sum_killed_classes / (number_of_satisfied_mutants * num_classes)
        print('E1_ms = ', E1_ms)
    else:
        E1_ms = 0
        
    # E2 mutation score
    mutants = killed_classes_df['mutant'].values
    # temp_killed_classes_df = killed_classes_df['c0 c1 c2 c3 c4 c5 c6 c7 c8 c9'.split()]
    temp_killed_classes_df = killed_classes_df[name_of_columns[1:]]
    temp_killed_classes_df = temp_killed_classes_df.sum(axis=1) # sum columns of a row
    temp_killed_classes_df = temp_killed_classes_df.clip(0,1)
    number_of_killed_classes = temp_killed_classes_df.sum()

    if len(mutants):
        E2_ms = number_of_killed_classes / len(mutants)
    else:
        E2_ms = 0
    
    # E3 mutation score
    E3_ms = sum(E3_ms_of_each_mutant)/len(all_mutants_file_name)
    killed_test_input_t = {'mutant':all_mutants_file_name, 'killing_score':E3_ms_of_each_mutant}
    killed_test_input_t = pd.DataFrame(killed_test_input_t)
    
    
    return round(E1_ms,10), round(E2_ms,10), round(E3_ms,4), killed_classes_df, killed_test_input_t




 
 
 
# =====================================================
### equivalent mutants %%%
def get_equivalent_mutants():
    ratio = '0.01' # if you have another ratio please consider in addressing
    equivalent_mutant_path = os.path.join(root,'mutants','equivalent_mutants', subject, 'equivalents.json')
    
    if os.path.isfile(equivalent_mutant_path):
        with open(equivalent_mutant_path,'r') as f:
            return json.load(f)
        
    return []    
    

 
### grouping for all E1, E2, and E3
def run_grouping_for_E1_E2_E3_ms(test_suite_size, uniform_sampling = False, sampling_from = ''):
    # subject = 'cifar10'
    # dataset_name = 'cifar10'
    # original_model = 'original_model.h5'
    # original_model_path = os.path.join('models',subject, original_model)
    # original_model = load_model(original_model_path)

    ratio = '0.01'

    extra_output_path = 'extra_outputs'
    
    mutant_models_path = os.path.join('mutants',subject,ratio)
    all_mutants_file_name = os.listdir(mutant_models_path)
    
    group_op_mapping = {
        'all_operators':[], # the empty list considers all operators
        # 'GF':  ['GF'],
        # 'LA': ['LA'],
        # 'NAI': ['NAI'],
        # 'NEB': ['NEB'],
        # 'NS': ['NS'],
        # 'WS': ['WS'],
        
        # 'LD': ['LD'],
        # 'LR': ['LR'],

        # 'group1': ['NAI', 'WS', 'NS', 'NEB', 'GF', 'LD', 'LR'],
        # 'group2': ['NAI', 'NS', 'NEB', 'GF', 'WS', 'LA'],
        # 'group3': ['NAI', 'NS', 'NEB', 'GF', 'WS'],
        # 'group4': [ 'NS', 'NEB', 'GF', 'WS'],
        # 'group4': [ 'NS', 'NEB', 'NAI', 'WS'],
        # 'group5': [ 'NS', 'NEB', 'NAI', 'GF'],
        # 'group6': [ 'NS', 'NEB', 'NAI'],
        # 'group7': [ 'NS', 'NEB', 'GF'],
        # 'group8': [ 'NS', 'NEB', 'WS'],
        
        
        
        # 'group3': ['LA','NAI','NS','WS'],
        # 'group4': ['LA','WS','NS'],
        # 'group5': ['WS','NS'],
    }

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..') # root of project
    
    groups_path = os.path.join(root,'plots','correlation_outputs', 'training_set', subject,'grouping', 'sorted_groups.json')
    print(groups_path)
    
    if len(group_op_mapping) == 0:
        if os.path.isfile(groups_path):
            with open(groups_path,'r') as f:
                group_op_mapping = json.load(f)
    
    print(group_op_mapping)
        
    batch_sizes = [test_suite_size]
    for key, group in group_op_mapping.items():
        group_name = key
        for bs in batch_sizes:
            
            bs_start_time = datetime.now()
            
            all_results = {'test_suite_size':[], 'test_suite':[], 'E1_MS':[],'E2_MS':[],'E3_MS':[], 'time_taken':[]}
            
                
            
            # if not uniform_sampling:
            #     source_path_test_suites = os.path.join('data', dataset_name, 'sampled_test_suites', f'batch_size_{bs}')
            # else:
            #     source_path_test_suites = os.path.join('data', dataset_name, 'uniform_sampled_test_suites', f'batch_size_{bs}')
            
            
            if not uniform_sampling:
                # sampling from test set
                source_path_test_suites = os.path.join('data', dataset_name, 'sampled_test_suites', f'batch_size_{bs}')
                # sampling from training set
                if sampling_from == 'training_set':
                    source_path_test_suites = os.path.join('data', dataset_name, sampling_from, 'sampled_test_suites', f'batch_size_{bs}')
                    
            else:
                source_path_test_suites = os.path.join('data', dataset_name, 'uniform_sampled_test_suites', f'batch_size_{bs}')
                # sampling from training set
                if sampling_from == 'training_set':
                    # source_path_test_suites = os.path.join('data', dataset_name, sampling_from, 'uniform_sampled_test_suites', f'batch_size_{bs}')
                    source_path_test_suites = os.path.join('data', dataset_name, 'uniform_sampled_test_suites', sampling_from,'sampled_test_suites', f'batch_size_{bs}')
            
            
            
            
                
            sampled_test_suites_name = os.listdir(source_path_test_suites)
            sampled_test_suites_name.sort()

            
            # if uniform_sampling:
            #     save_path_of_killed_classes = os.path.join('killed_classes_output', subject, ratio, 'uniform_sampling', f'bs_{bs}')
            # else:
            #     save_path_of_killed_classes = os.path.join('killed_classes_output', subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
            # if not os.path.exists(save_path_of_killed_classes):
            #     raise TypeError("The path not exist:", save_path_of_killed_classes)
            
            if uniform_sampling:
                save_path_of_killed_classes = os.path.join('killed_classes_output', subject, ratio, 'uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_of_killed_classes = os.path.join('killed_classes_output', sampling_from, subject, ratio, 'uniform_sampling', f'bs_{bs}')
                    
                
            else:
                save_path_of_killed_classes = os.path.join('killed_classes_output', subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_of_killed_classes = os.path.join('killed_classes_output', sampling_from, subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
            
            
            
            
            
            
            # if uniform_sampling:
            #     save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', subject, ratio,'uniform_sampling', f'bs_{bs}')
            # else:
            #     save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', subject, ratio,'non_uniform_sampling', f'bs_{bs}')
            
            if uniform_sampling:
                save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', subject, ratio,'uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', sampling_from, subject, ratio,'uniform_sampling', f'bs_{bs}')
                    
            else:
                save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', subject, ratio,'non_uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_killing_score_of_each_mutant = os.path.join(extra_output_path,'killed_inputs', sampling_from, subject, ratio,'non_uniform_sampling', f'bs_{bs}')
                
                    
                
            # if not os.path.exists(save_path_killing_score_of_each_mutant):
            #     raise TypeError("The path not exist:", save_path_killing_score_of_each_mutant)

            counter = 0
            counter_for_ts = -1 # using for multithread 
            for test_suite_name in sampled_test_suites_name:
                if not test_suite_name.endswith('.npy'):
                    continue
                
                counter_for_ts += 1
                # if counter_for_ts < from_index or counter_for_ts >= to_index:
                #     print('this is for another job or thread:', counter_for_ts)
                #     continue
                    
                # if test_suite_name.replace('npy','csv') in os.listdir(save_path_of_killed_classes):
                #     print(f'{counter_for_ts} :', test_suite_name, 'done!')
                #     continue
                
                print(f'{counter_for_ts} :', test_suite_name, 'started!')
                
                # filter test suite by original model to get correctly predicteds inputs
                # test_suite_path = os.path.join(source_path_test_suites, f'{test_suite_name}')
                
                # x_test_suite, y_test_suite = remove_mispredicteds_inputs(test_suite_path, original_model, dataset_name) # return a test suite that correctly predicted by original model correctly
                # print('removed mispredicted inputs on original model')


                # filter mutants for test suite T' and calculate mutation score for test suite T' and all mutants
                start = datetime.now()
                
                    
                start = datetime.now()
                
                if not os.path.isfile(os.path.join(save_path_of_killed_classes, test_suite_name.replace('npy','csv'))):
                    continue
                
                
                E1_ms, E2_ms = mutation_score_of_E1_E2_by_grouping(test_suite_name.replace('npy','csv'), save_path_of_killed_classes, group)
                # E3_ms = mutation_score_of_E3_by_grouping(test_suite_name.replace('npy','csv'), save_path_killing_score_of_each_mutant, group)
                E3_ms = 0
                
                end = datetime.now()
                time_taken = end - start
               
                if group:
                    group_dir_name = '_'.join(group)
                else:
                    group_dir_name = ''
                
                # if uniform_sampling:
                #     save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results', subject, 'grouping', ratio,group_dir_name,'uniform_sampling', f'bs_{bs}')
                # else:
                #     save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results', subject,'grouping', ratio, group_dir_name, 'non_uniform_sampling', f'bs_{bs}')
                
                if uniform_sampling:
                    # pass # because I ran some experiment without using 'uniform_sampling' so it is better to hold same path
                    save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results',subject, 'grouping', ratio, group_dir_name,'uniform_sampling', f'bs_{bs}')
                    # save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results',subject, 'grouping', ratio, group_dir_name, 'uniform_sampling', f'bs_{bs}')
                    if sampling_from == 'training_set':
                        save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results',subject, 'grouping', ratio, group_dir_name, sampling_from,'uniform_sampling', f'bs_{bs}')
                                        
                else:
                    save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results',subject, 'grouping', ratio, group_dir_name, 'non_uniform_sampling', f'bs_{bs}')
                    
                    if sampling_from == 'training_set':
                        save_path_of_all_result_of_grouping = os.path.join(base_path_of_outputs,'all_results',subject, 'grouping', ratio, group_dir_name, sampling_from,'non_uniform_sampling', f'bs_{bs}')
                        
                
                if not os.path.exists(save_path_of_all_result_of_grouping):
                    os.makedirs(save_path_of_all_result_of_grouping)
                
                with open(os.path.join(save_path_of_all_result_of_grouping,f'bs{bs}_result_E1_E2_E3.csv'), 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n')
                    
                    if f1.tell() == 0:
                        header = all_results.keys()
                        writer.writerow(header)
                        
                    writer.writerow([ str(bs), str(test_suite_name), str(round(E1_ms,3)),str(round(E2_ms,3)),str(round(E3_ms,3)), str(time_taken)])
        
        
        
        
        
        
        
        
### run for all E1, E2 and E3 %%%
def run_for_E1_E2_E3_ms(test_suite_size, uniform_sampling = False, from_index = -1 , to_index = 300+1, sampling_from = '', shifted_test_input=0):
    print('start from index:',from_index,'   to:',to_index)
    global base_path_of_outputs
    
    
    # sampling_from = 'training_set' # this should be a parameter 
    use_actual_label_to_kill_mutant = False # if True means use acutal y else (False) use the predicted label by original model )()
    
    original_model = 'original_model.h5'
    original_model_path = os.path.join('models',subject, original_model)
    print('original model path:', original_model_path)
    # original_model = load_model(original_model_path) # need to import keras load_model

    ratio = '0.01'

    extra_output_path = 'extra_outputs'
    
    mutant_models_path = os.path.join('mutants',subject,ratio)
    # all_mutants_file_name = os.listdir(mutant_models_path)

    batch_sizes = [test_suite_size]
    for bs in batch_sizes:
        
        bs_start_time = datetime.now()
        
        all_results = {'test_suite_size':[], 'test_suite':[], 'E1_MS':[],'E2_MS':[],'E3_MS':[], 'time_taken':[]}
        
            
        
        if not uniform_sampling:
            # sampling from test set
            source_path_test_suites = os.path.join('data', dataset_name, 'sampled_test_suites', f'batch_size_{bs}')
            # sampling from training set
            if sampling_from == 'training_set':
                source_path_test_suites = os.path.join('data', dataset_name, sampling_from, 'sampled_test_suites', f'batch_size_{bs}')
                
        else:
            source_path_test_suites = os.path.join('data', dataset_name, 'uniform_sampled_test_suites', 'sampled_test_suites', f'batch_size_{bs}')
            # sampling from training set
            if sampling_from == 'training_set':
                # source_path_test_suites = os.path.join('data', dataset_name, sampling_from, 'uniform_sampled_test_suites', f'batch_size_{bs}')
                source_path_test_suites = os.path.join('data', dataset_name, 'uniform_sampled_test_suites', sampling_from, 'sampled_test_suites',  f'batch_size_{bs}')
            
        
            
        sampled_test_suites_name = os.listdir(source_path_test_suites) # if you are using dic get len
        
        use_one_dic_file = False
        if 'test_suites_all.npy' in sampled_test_suites_name:
            test_suites_all_dic = np.load(os.path.join(source_path_test_suites, 'test_suites_all.npy'), allow_pickle=True).item()
            sampled_test_suites_name = list(test_suites_all_dic.keys())
            use_one_dic_file = True
            
            
        
        ### new version of sampled_test_suites : for example of each batch size we have just one file (test_suites_all.npy) instead 300 file and the file is a dictionary include 300 keys with same name of 300 files
        
        
        sampled_test_suites_name.sort()

        # failed_mutants_testSuite_df = pd.DataFrame()
        # failed_mutants_testSuite_df['mutant'] = all_mutants_file_name
        # failed_mutants_testSuite_df.set_index('mutant')
        
        
        if uniform_sampling:
            save_path_of_killed_classes = os.path.join(base_path_of_outputs, 'killed_classes_output', subject, ratio, 'uniform_sampling', f'bs_{bs}')
            
            if sampling_from == 'training_set':
                save_path_of_killed_classes = os.path.join(base_path_of_outputs, 'killed_classes_output', sampling_from, subject, ratio, 'uniform_sampling', f'bs_{bs}')
                
            
        else:
            save_path_of_killed_classes = os.path.join(base_path_of_outputs, 'killed_classes_output', subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
            
            if sampling_from == 'training_set':
                save_path_of_killed_classes = os.path.join(base_path_of_outputs, 'killed_classes_output', sampling_from, subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
            
        if not os.path.exists(save_path_of_killed_classes):
            try:
                os.makedirs(save_path_of_killed_classes)
            except:
                print(f'The dir {save_path_of_killed_classes} has created before!')
        
        
        if uniform_sampling:
            save_path_killing_score_of_each_mutant = os.path.join(base_path_of_outputs, extra_output_path,'killed_inputs', subject, ratio,'uniform_sampling', f'bs_{bs}')
            
            if sampling_from == 'training_set':
                save_path_killing_score_of_each_mutant = os.path.join(base_path_of_outputs, extra_output_path,'killed_inputs', sampling_from, subject, ratio,'uniform_sampling', f'bs_{bs}')
                
        else:
            save_path_killing_score_of_each_mutant = os.path.join(base_path_of_outputs, extra_output_path,'killed_inputs', subject, ratio,'non_uniform_sampling', f'bs_{bs}')
            
            if sampling_from == 'training_set':
                save_path_killing_score_of_each_mutant = os.path.join(base_path_of_outputs, extra_output_path,'killed_inputs', sampling_from, subject, ratio,'non_uniform_sampling', f'bs_{bs}')
            
            
        # if not os.path.exists(save_path_killing_score_of_each_mutant):
        #     try:
        #         os.makedirs(save_path_killing_score_of_each_mutant)
        #     except:
        #         print(f'the dir {save_path_killing_score_of_each_mutant} has created before!')

        counter = 0
        counter_for_ts = -1 # using for multithread 
        for test_suite_name in sampled_test_suites_name:
            if not test_suite_name.endswith('.npy'):
                continue
            
            counter_for_ts += 1
            if counter_for_ts < from_index or counter_for_ts >= to_index:
                print('this is for another job or thread:', counter_for_ts)
                continue
                
            if test_suite_name.replace('npy','csv') in os.listdir(save_path_of_killed_classes):
                print(f'{counter_for_ts} :', test_suite_name, 'has been done before')
                continue
                
                start = datetime.now()
                
                E1_ms, E2_ms = mutation_score_of_E1_E2_by_grouping(test_suite_name.replace('npy','csv'), save_path_of_killed_classes, group=[], subject=subject)
                E3_ms = mutation_score_of_E3_by_grouping(test_suite_name.replace('npy','csv'), save_path_killing_score_of_each_mutant, group=[], subject=subject)
                
                end = datetime.now()
                time_taken = end - start
                
                ##########################################
                # continue
            
            else:
                print(f'{counter_for_ts} :', test_suite_name, 'started!')
                
                # filter test suite by original model to get correctly predicteds inputs
                test_suite_path = os.path.join(source_path_test_suites, f'{test_suite_name}')
                
                if use_actual_label_to_kill_mutant:
                    # filter mutants for test suite T' and calculate mutation score for test suite T' and all mutants
                    x_test_suite, y_test_suite = remove_mispredicteds_inputs(test_suite_path, original_model, dataset_name) # return a test suite that correctly predicted by original model correctly
                    print('removing mispredicted inputs on original model to get passed data')
                    temp_original_model = None
                    
                else:
                    if use_one_dic_file :
                        test_suite_x_y_index = test_suites_all_dic[test_suite_name]
                        x_test_suite, y_test_suite, indeces = test_suite_x_y_index['x_test'], test_suite_x_y_index['y_test'], test_suite_x_y_index['indices']
                    
                    else:   
                        # without filtering test suite with actual prediction (need original model to compare the predicted label with predicted label by mutant)
                        x_test_suite, y_test_suite, indeces = load_test_suite(test_suite_path, dataset_name)
                    # print('it does not remove mispredicted inputs, it use predicted label to kill mutants')
                    temp_original_model = original_model
                
                if uniform_sampling:
                    indeces = list(indeces)  

                # filter mutants for test suite T' and calculate mutation score for test suite T' and all mutants
                start = datetime.now()
                
                # calculate mutation score
                E1_ms, E2_ms, E3_ms, killed_classes_df, killing_score_df = mutation_score_of_E1_E2_E3(mutant_models_path, x_test_suite, y_test_suite, subject=subject, original_model= temp_original_model, test_suite_indeces = indeces)
                
                end = datetime.now()
                
                time_taken = end - start

                # save important information
                # killed_classes_df.to_csv(os.path.join(save_path_of_killed_classes, test_suite_name.replace('npy','csv'))) # needs for E1 and E2 grouping 
                # killing_score_df.to_csv(os.path.join(save_path_killing_score_of_each_mutant, test_suite_name.replace('npy','csv'))) # needs for E3 grouping


            if uniform_sampling:
                # pass # because I ran some experiment without using 'uniform_sampling' so it is better to hold same path
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio,'uniform_sampling', f'bs_{bs}')
                # save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio, 'uniform_sampling', f'bs_{bs}')
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio, sampling_from,'uniform_sampling', f'bs_{bs}')
                                    
            else:
                subject_postfix = subject
                if shifted_test_input:
                    subject_postfix = subject + '_' + dataset_name       
                    
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject_postfix, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject_postfix, ratio, sampling_from,'non_uniform_sampling', f'bs_{bs}')

                
            if not os.path.exists(save_path_of_all_result):
                try:
                    os.makedirs(save_path_of_all_result)
                except:
                    print(f'the dir {save_path_of_all_result} has created before!')
                
            
            
            prime = '_prime'
            
            if from_index == -1 and to_index == 301:
                save_file_name = f'bs{bs}_result_E1_E2_E3.csv'
            else:
                save_file_name = f'{from_index}_to_{to_index}_bs{bs}_result_temp_E1_E2_E3{prime}.csv'
            
            with open(os.path.join(save_path_of_all_result,save_file_name), 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n')
                
                if f1.tell() == 0:
                    header = all_results.keys()
                    writer.writerow(header)
                    
                writer.writerow([ str(bs), str(test_suite_name), str(round(E1_ms,4)),str(round(E2_ms,4)),str(round(E3_ms,4)), str(time_taken)])
                    
        bs_end_time = datetime.now()

        print(f'elapse time for bs {bs}:', bs_end_time-bs_start_time)

            


    

if __name__ == '__main__':
    '''
    execute like this command: python cnn_mutation/src/generator.py --model_path models/lenet4/original_model.h5 --subject_name lenet4 --data_type fashion_mnist --threshold 0.9 --operator -1 --ratio 0.01 --save_path mutants/lenet4 --num 50
    
    run from the root of project
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-experiment",
                        type=int,
                        default= 0,
                        help="experiment number ... [0 menas all experiments E1, E2, and E3]")
    
    parser.add_argument("--test_suite_size", "-test_suite_size",
                        type=int,
                        help="test suite size for example 1500 or 500 or ...")
    
    parser.add_argument("--uniform_sampling", "-uniform_sampling",
                        type=str,
                        choices=['False', 'True'],
                        default=False, # 0 means False
                        help="test suite sampling type choices=['False', 'True'] ")
        
    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        help="dataset name for example: mnist or cifar10")
    
    parser.add_argument("--model_name", "-model_name",
                    type=str,
                    help="model name for example: lenet5 or cifar10")
    
    parser.add_argument("--from_index", "-from_index",
                        type=int,
                        default= -1,
                        help="start index of test suite")
    
    parser.add_argument("--to_index", "-to_index",
                        type=int,
                        default= 300+1, # size of our test suites + 1
                        help="end index of test suite")
    
    parser.add_argument("--sampling_from", "-sampling_from",
                        type=str,
                        default= '',
                        help="training_set or test_set")
    
    parser.add_argument("--base_path", "-base_path",
                        type=str,
                        default= '',
                        help="base path for outputs")
    
    parser.add_argument("--shifted_test_input", "-shifted_test_input",
                        type=int,
                        default= 0,
                        help="if you want to test on different distribution of dataset please set up to 1")
    
    
    
    args = parser.parse_args()
    experiment = args.experiment
    test_suite_size = args.test_suite_size
    uniform_sampling = args.uniform_sampling
    dataset_name = args.dataset
    subject = args.model_name
    from_index = args.from_index
    to_index = args.to_index
    sampling_from = args.sampling_from
    shifted_test_input = args.shifted_test_input
    
    # global variable
    base_path_of_outputs = args.base_path
    
    # ------- TODO: IMPORTANT: set number of clasess based on your dataset -------
    num_classes = 10
    
    if dataset_name in ['cifar10', 'mnist', 'SVHN', 'fashion_mnist']: 
        num_classes = 10
    elif dataset_name in ['cifar100']:
        num_classes = 100
    elif dataset_name in ['imagenet']:
        num_classes = 1000
        
    
    if shifted_test_input == 0:
        if sampling_from == 'training_set':
            load_predictions_of_mutants_for_train_set(subject) # this function set some global variables
        else:
            load_predictions_of_mutants_for_test_set(subject) # this function set some global variables
            
    elif shifted_test_input:
        print('read from different distribution')
        load_predictions_of_mutants_for_shifted_test_set(subject, dataset_name) ## calculate MS for different distribution
    
    
    
    if uniform_sampling == 'True':
        uniform_sampling = True
    elif uniform_sampling == 'False':
        uniform_sampling = False
    
    
    if experiment == 0:
        run_for_E1_E2_E3_ms(test_suite_size, uniform_sampling, from_index, to_index, sampling_from, shifted_test_input)
        print('finished all experiments !!')
        
    elif experiment == 1:
        # run()
        # run_with_exsit_killed_classes_matrix()
        pass

    elif experiment == 2:
        # run_for_basic_mutation_score()
        # run_for_basic_mutation_score_if_acc_is_1()
        pass

    elif experiment == 3:
        pass
        # run_for_deepmutation_plus_plus_ms(test_suite_size)
    
    elif experiment == 4:
        run_grouping_for_E1_E2_E3_ms(test_suite_size, uniform_sampling, sampling_from)
    
    # run_for_grouping_basic_mutation_score()
    # run_for_grouping_mutation_score_deepmutation()
    # calculate_mutation_score_on_strong()

    