import os


import sys
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from test_suite_handler import load_test_suite
from subject_config import constants

from datetime import datetime

import argparse



# print(os.path.abspath())
root = os.path.dirname(os.path.abspath(__file__)) # 
print(root)


shifted_test_input = ''

    
## calculate FDR
def custom_FDR_calculator(test_suite_path,clusters, mis_index, number_of_training_faults=0, test_suite_dic = {}):
    maxT = len(set(clusters)) # number of total faults of test mispredicteds
    
    # maxT = (clusters[:len(mis_index)]).max()+1 # total faults (you should change this) <<<<<< check result when you use this total faults >>>>>>>>>>

    if -1 in clusters:
        maxT = maxT - 1 ## if there is noise cluster (-1) this igonres it
    
    faults = set()
    
    if len(test_suite_dic):
        x,y,test_suite_indexes = test_suite_dic['x_test'], test_suite_dic['y_test'], test_suite_dic['indices']

    else:
        x,y,test_suite_indexes = load_test_suite(test_suite_path)
    
    list_of_mispredicteds_included_test_suite = []
    noise_happend_count = 0
    for ts_index in test_suite_indexes:
        if ts_index in mis_index:
            list_of_mispredicteds_included_test_suite.append(ts_index)
            which_cluster = clusters[list(mis_index).index(ts_index)]
            # comment 3 below lines if you want use test suite include_noise_as_a_fault
            if which_cluster == -1:
                noise_happend_count += 1
                continue
            faults.add(which_cluster)
    
    # print('number of mispredicteds in test suite:', len(list_of_mispredicteds_included_test_suite))
    # print('number of found faults:', len(faults))
    # print('noise happend:', noise_happend_count)
    FDR = round(len(faults)/maxT,5)
    
    # if number_of_training_faults > 0:
    #     FDR = round(len(faults)/number_of_training_faults,2)
    
    return FDR




def get_clusters(model_name, key, source_for_faults, need_umap_features=False):
    # dir_path = os.path.join(root, '..', 'fault_extraction', 'output','fault_clusters', 'training_set', model_name)
    dir_path = os.path.join(root, '..', 'fault_extraction', 'output','fault_clusters', source_for_faults, model_name)
    print('read faults from:', dir_path)
    
    # if source_for_faults == 'test_set':
    #     output_of_clustering = np.load(os.path.join(dir_path, "output_of_clustering_dis.npy"), allow_pickle=True)
    # else:
    if shifted_test_input and source_for_faults == 'test_set': ### here we want to test shifted test set 
        output_of_clustering = np.load(os.path.join(dir_path, shifted_test_input, "output_of_clustering.npy"), allow_pickle=True)
    
    else: 
        output_of_clustering = np.load(os.path.join(dir_path, "output_of_clustering.npy"), allow_pickle=True)
        
    print('key -> ',key)

    clustering_info_dic = output_of_clustering.item().get(key)
    print(clustering_info_dic.keys())
    hdbscan_in_umap = clustering_info_dic['hdbscan_in_umap']
    clustering_label = clustering_info_dic['clustering_label']
    
    ### calculate silouhette score here by (hdbscan_in_umap, clustering_label)
    
    print(len(hdbscan_in_umap[0]))
    
    
    
    if source_for_faults == 'training_set':
        mis_tindex = clustering_info_dic['mis_train_index'] # index of mispredicted on train
        number_of_training_faults = 0
        if need_umap_features:
            return clustering_label, mis_tindex, number_of_training_faults, hdbscan_in_umap
        
        else:
            return clustering_label, mis_tindex, number_of_training_faults
    
    elif source_for_faults == 'test_set':
        mis_index = clustering_info_dic['mis_test_index'] # index of mispredicted on test
        number_of_training_faults = clustering_info_dic['number_of_training_faults']
        if need_umap_features:
            return clustering_label, mis_index, number_of_training_faults, hdbscan_in_umap
        
        else:
            return clustering_label, mis_index, number_of_training_faults
    
    return clustering_label



def run(model_name, key, type_of_test_suite_sampling, source_for_faults = ''):
    uniform_sampling = True
    
    dataset_name = constants[model_name]['dataset']
    
    if 'test' in source_for_faults:
        if shifted_test_input: # when we want to use of test set with different distributino of original dataset
            sampled_test_suites_path = os.path.join(root, '..' , "data", shifted_test_input, '',type_of_test_suite_sampling)   
             
        else:   
            sampled_test_suites_path = os.path.join(root, '..' , "data", dataset_name, '',type_of_test_suite_sampling)
    else:
        sampled_test_suites_path = os.path.join(root, '..' , "data", dataset_name, source_for_faults,type_of_test_suite_sampling)
        
        if uniform_sampling: 
            print('uniform subsets')
            sampled_test_suites_path = os.path.join(root, '..' , "data", dataset_name,'uniform_sampled_test_suites', source_for_faults,type_of_test_suite_sampling)

    
    

    print(sampled_test_suites_path)

    batch_sizes = [int(x.split('_')[-1]) for x in os.listdir(sampled_test_suites_path)]    
    
    exclude_batch_sizes = []
    
    batch_sizes = [i for i in batch_sizes if i not in exclude_batch_sizes]
    
    print(batch_sizes)
    
    clusters, mis_index, num_of_training_faults = get_clusters(model_name, key, source_for_faults)
    print(f'number of clusters (faults) of {source_for_faults}:', len(set(clusters)))
    print('number of noise', np.count_nonzero(clusters == -1))
    print(f'number of mispredicteds of {source_for_faults}:',len(mis_index))
    # print(f'number of training faults:',num_of_training_faults)
    s = 0

    
   
    
    for bs in batch_sizes:
        print(bs)
        print(type_of_test_suite_sampling)
        bs_dir = os.path.join(sampled_test_suites_path, f'batch_size_{bs}')
        result = {'test_suite':[], 'test_suite_size':[],'FDR':[]}
        
        try:
            output_FDR_path = os.path.join(root, "FDR_output", source_for_faults, model_name, type_of_test_suite_sampling, f'bs_{bs}')
            df = pd.read_csv(os.path.join(output_FDR_path,'result.csv'))
            loaded_result = df.to_dict(orient='list')
            result['FDR'] = loaded_result['FDR']
            result['test_suite'] = loaded_result['test_suite']
            result['test_suite_size'] = loaded_result['test_suite_size']
            
        except:
            pass     
           
           
        if not os.path.exists(bs_dir):
            continue
        
        counter = 0
        
        sampled_test_suites_name = os.listdir(bs_dir) 
        
        use_one_dic_file = False
        if 'test_suites_all.npy' in sampled_test_suites_name:
            test_suites_all_dic = np.load(os.path.join(bs_dir, 'test_suites_all.npy'), allow_pickle=True).item()
            sampled_test_suites_name = test_suites_all_dic.keys()
            use_one_dic_file = True
            
        
        ###### save path #######
        if shifted_test_input and source_for_faults == 'test_set':
            output_FDR_path = os.path.join(root, "FDR_output", 'shifted_test_set', model_name, shifted_test_input, type_of_test_suite_sampling, f'bs_{bs}')  
        elif uniform_sampling:
            output_FDR_path = os.path.join(root, "FDR_output", 'uniform_sampling', model_name, shifted_test_input, type_of_test_suite_sampling, f'bs_{bs}')  
        else:
            output_FDR_path = os.path.join(root, "FDR_output", source_for_faults, model_name, type_of_test_suite_sampling, f'bs_{bs}')

        if not os.path.exists(output_FDR_path):
            os.makedirs(output_FDR_path) 
        ### end save path
            
        sampled_test_suites_name.sort()
        
        for test_suite in sampled_test_suites_name:
            if test_suite in result['test_suite']:
                print(f'{test_suite} done before!')
                continue
            
            num_of_training_faults = 0
            
            if use_one_dic_file :
                test_suite_x_y_index_dic = test_suites_all_dic[test_suite]
            else:
                test_suite_x_y_index_dic = {}   
                     
            # s = datetime.now()
            FDR = custom_FDR_calculator(os.path.join(bs_dir, test_suite), clusters, mis_index, num_of_training_faults, test_suite_dic=test_suite_x_y_index_dic)
            # e = datetime.now()
            
            
            
            
            result['FDR'].append(FDR)
            result['test_suite'].append(test_suite)
            result['test_suite_size'].append(bs)
            
            # Append the current row of results to df_result

            if counter % 10 == 0:
                print(bs, FDR)
                
            counter += 1
            
            
            df_result = pd.DataFrame(result)
        
           
            df_result.to_csv(os.path.join(output_FDR_path,'result.csv'))


if __name__ == '__main__':

    type_of_test_suite_sampling = 'sampled_test_suites' # meanes this is random sampling

    models = [('lenet5_mnist',True),
              ('cifar10', True),
              ('lenet4',True), 
              ('lenet5_SVHN',True), 
              ('resnet20_cifar10',True),
              ('vgg16_SVHN',False),
              ('resnet152_cifar100',True)]
    

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        default='',
                        help="subject_dataset")
    
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='',
                        help="training set or test set")
    
    parser.add_argument("--shifted_test_input", "-shifted_test_input",
                        type=str,
                        default='',
                        help="if you want to use of shifted test set please set up the name of shifted dataset")
    
   
    
    
    args = parser.parse_args()
    subject = args.subject
    shifted_test_input = args.shifted_test_input
    data_type = args.data_type
    
    
    
    models = [(subject, True)]
    
    # models = [('lenet5_mnist',True)]
    
    
    for model_name, use_last_epoch in models:
        print('===============================================')
        print(model_name)
        # for source in ['training', 'test']:
        for source in [data_type]:
            print(f'calculate FDR for {source} set')
            source_data_for_clustering = source+'_set'
            key = 'last_epoch_'+source if use_last_epoch else 'all_epoch_'+source
            run(model_name, key, type_of_test_suite_sampling, source_data_for_clustering)
        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
        print('--------------------------------------------------------')
