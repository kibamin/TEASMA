# !pip install umap-learn
# !pip install tslearn
# !pip install hdbscan numpy==1.23
print('import libraries')

import os
import copy
import time
import argparse


import pickle

import logging
from datetime import datetime
# from mis_predicted_feature_extraction import get_model_object

logs_path = "app_logs"

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from subject_config import constants ### set before executing


shifted_test_input = 0
dataset_name = ''



job_id = os.getenv('SLURM_JOB_ID')

# Check if the environment variable exists
if job_id is not None:
    print(f"Running under Slurm with Job ID: {job_id}")
else:
    job_id = ''
    print("Not running under Slurm or Job ID not found.")

from sklearn.metrics import silhouette_score
import numpy as np
import hdbscan
import umap.umap_ as umap

# from scipy.spatial.distance import euclidean, cosine

class MyLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        log_file=f'{name}.log'
        file_handler = logging.FileHandler(os.path.join(logs_path, log_file))
        file_handler.setLevel(logging.DEBUG)  # Set the log level for the file handler
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.addHandler(file_handler)


logger = MyLogger(f'{job_id}_fault_clustering_estimation.py') 


root = os.path.dirname(os.path.abspath(__file__))

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def scale_one(X):
    nom = (X-X.min())*(1)
    denom = X.max() - X.min()
    return nom/denom

# constants = {
#     'lenet5': {'epochs':12, 'dataset':'mnist'},
#     'cifar10': {'epochs':20, 'dataset':'cifar10'},
#     'lenet4': {'epochs':20, 'dataset':'fashion_mnist'},
#     'lenet5_SVHN': {'epochs':20, 'dataset':'SVHN'},
#     'resnet20_cifar10': {'epochs':100, 'dataset':'cifar10'},
#     'vgg16_SVHN': {'epochs':10, 'dataset':'SVHN'},
    
#     'resnet50_amazon': {'epochs':5, 'dataset':'amazon'},
#     'resnet50_office31_mix': {'epochs':10, 'dataset':'office31_mix'},
    
#     'resnet50_cifar10': {'epochs':6, 'dataset':'cifar10'},
    
#     'lenet5_mnist': {'epochs':12, 'dataset':'mnist'},
    
#     'resnet50_caltech256': {'epochs':15, 'dataset':'caltech256'},
#     'resnet50_caltech256_8020': {'epochs':20, 'dataset':'caltech256_8020'},
#     'resnet50_office31': {'epochs':20, 'dataset':'office31'},
    
# }

def get_original_epoch(model_name):
    # model_obj = get_model_object(model_name)
    # constants[model_name]['epochs']
    # return model_obj.original_epochs
    return constants[model_name]['epochs']

def mix_unique_mispredicteds_of_all_epochs(model_name, from_epoch=1, diff_test_data_path=''):
    data_for_clustering = np.load(os.path.join(root, "output", "mispredicteds", model_name, diff_test_data_path, "data_for_clustering_for_epoch.npy"), allow_pickle=True)
    print(data_for_clustering)
    total_x_mis_of_all_epochs = []
    total_tt = []
    total_tst = []
    total_mis_index = []
    total_mis_tindex = []

    for e in range(from_epoch, get_original_epoch(model_name)+1):
        data_per_epoch = data_for_clustering.item().get(f'epoch_{e}')

        i = -1
        # test
        for index in data_per_epoch['mis_test_index']:
            i += 1
            if index not in total_mis_index: # check that be unique without any repetition
                total_mis_index.append(index)
                total_x_mis_of_all_epochs.append(data_per_epoch['x_mis'][i])
                total_tt.append(data_per_epoch['y_actual_test_and_train'][i])
                total_tst.append(data_per_epoch['y_predicted_test_and_train'][i])

        # train
        for index in data_per_epoch['mis_train_index']:
            i += 1
            if index not in total_mis_tindex: # check that be unique without any repetition
                total_mis_tindex.append(index)
                total_x_mis_of_all_epochs.append(data_per_epoch['x_mis'][i])
                total_tt.append(data_per_epoch['y_actual_test_and_train'][i])
                total_tst.append(data_per_epoch['y_predicted_test_and_train'][i])

    print('number of unique mispredicteds of training set:', len(total_mis_tindex))
    print('number of unique mispredicteds of test set:', len(total_mis_index))

    data_total_mix_for_clustering = {}

    data_total_mix_for_clustering[f'total_mix_unique_mispredicteds'] = {
        'x_mis': total_x_mis_of_all_epochs, 
        'y_actual_test_and_train': total_tt,
        'y_predicted_test_and_train': total_tst,
        'mis_test_index': total_mis_index,
        'mis_train_index': total_mis_tindex
    }

    # save mix of all epochs
    save_path_dir = os.path.join(root, "output", "mispredicteds", "training_set", model_name)
    try:
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
    except:
        pass
        
    np.save(os.path.join(save_path_dir, "data_for_clustering_of_mix_epochs.npy"), data_total_mix_for_clustering)

    return total_x_mis_of_all_epochs, total_tt, total_tst, total_mis_index, total_mis_tindex

def load_data_for_clustering_last_epoch(model_name):
    
    if shifted_test_input:
        data_for_clustering = np.load(os.path.join(root, "output", "mispredicteds", 'transformed', model_name, dataset_name, "data_for_clustering_for_epoch.npy"), allow_pickle=True)
    else:
        data_for_clustering = np.load(os.path.join(root, "output", "mispredicteds", model_name, "data_for_clustering_for_epoch.npy"), allow_pickle=True)
            
    
    # last_epoch = constants[model_name]['epochs']
    last_epoch = get_original_epoch(model_name)
    key = f'epoch_{last_epoch}'

    print(data_for_clustering)
    
    data_epoch = data_for_clustering.item().get(key)
        
    x_mis = data_epoch['x_mis']
    tt = data_epoch['y_actual_test_and_train']
    tst = data_epoch['y_predicted_test_and_train']
    mis_index = data_epoch['mis_test_index']
    mis_tindex = data_epoch['mis_train_index']

    return x_mis, tt, tst, mis_index, mis_tindex

#######################################################
### HDBSCAN Clustering

def run_clustering(x_mis, tt, tst, mis_index, mis_tindex, key, ii, jj, min_cluster_s = 20):
    print('start run clustering')
    
    # x_mis = np.load(os.path.join(root, "output", "mispredicteds", "inceptionV3_imagenet", "mispredicted_X_scf_train.npy"))

    # num_rows = len(x_mis)
    # sample_size = int(0.05 * num_rows)
    # indices = np.random.choice(num_rows, size=sample_size, replace=False)
    # x_mis = x_mis[indices]
    
    output_of_clustering = {}


    trace = []
    ss_tt = []
    Clustering_Label = []
    hdbscan_in_umap = []
    clusterer_object = []
    x_mis_test_u = []
    
    

    all_clustering_info = {'i_j':[],
                            'k_o':[],
                            'n_n':[],
                            'metric':[],
                            'first_umap_transformer':[],
                            'second_umap_transformer':[],
                            'min_cluster_size':[],
                            'clusterer_object':[],
                            'Silhouette_score':[],
                            'DBCV_score':[],
                            'clustering_label':[],
                            'hdbscan_in_umap':[]
                            }
   
    for metric in ['euclidean']:
        # for i,j in zip([250,200,100],[200,150,50]):  # double reduction 
        for i,j in zip([ii],[jj]):  # double reduction 
            # for k,o in zip([3,5,10,15],[3,5,10,15]): # k-neighbors
            for k,o in zip([10],[10]): # k-neighbors
                # for n_n in [0.001, 0.01, 0.1]:
                for n_n in [0.01]:
                    
                    
                    # dimension reduction x_mis (512 D) -> u1 (i D)
                    
                    # if len(x_mis) < 300:
                    #     k = k - 1
                    #     o = o - 1
                    try:
                        
                        # u is x_mis
                        u = None
                        # u = np.load("/TEASMA/MS_code/fault_extraction/output/mispredicteds/inceptionV3_imagenet/mispredicted_X_scf_train.npy")
                        u = np.load(os.path.join(root, "output", "mispredicteds", "inceptionV3_imagenet", "mispredicted_X_scf_train.npy"))
                        # num_rows = len(x_mis)
                        # sample_size = int(0.1 * num_rows)
                        # indices = np.random.choice(num_rows, size=sample_size, replace=False)
                        # sampled_rows = x_mis[indices]
                        
                        # len_x_mis = len(x_mis)
                        logger.info('start first umap')
                        # with open(f'{job_id}output_of_{ii}_{jj}.txt', 'a') as file:
                        #     file.write('start first umap \n')
                        #     file.write(f'------- i(first reduction)={i} j(second reduction)={j} k(k-neighbors for first)={k} o(k-neighbors for second)={o} n_n(min distance)={n_n} \n')
                            
    
                        logger.info(f'------- i(first reduction)={i} j(second reduction)={j} k(k-neighbors for first)={k} o(k-neighbors for second)={o} n_n(min distance)={n_n}')
                        
                        first_umap_reducer = umap.UMAP(min_dist=n_n, n_components=i, n_neighbors=k, metric=metric)
                        u = first_umap_reducer.fit_transform(u) # u1 ## it takes 1.5 hours
                        
                        
                        
                        # with open(f'{job_id}output_of_{ii}_{jj}.txt', 'a') as file:
                        #     file.write('===============1 first umap done=============== \n')
                        #     file.write('start second umap \n')
                            
                            
                        logger.info('===============1 first umap done===============')
                        
                        # dimension reduction u1 (i D) -> u (j D)
                        logger.info('start second umap')
                        
                        second_umap_reducer = umap.UMAP(min_dist=n_n, n_components=j, n_neighbors=o, metric=metric)
                        u = second_umap_reducer.fit_transform(u) # u ## it takes 1 hours
                        
                        with open(f'{job_id}output_of_{ii}_{jj}.txt', 'a') as file:
                            file.write('===============2 - second umap done=============== \n')
                    except:
                        logger.warning('--- exception in umap dimention reduction ---')
                        continue
                    
                    # u = np.c_[u, tt_scale]
                    # u = np.c_[u, tst_scale]
                    
                    # print('shape x_mis before split test mispredictions:', u.shape)
                    # test_umap = u[:len(mis_index)]
                    # print('u_map test:', test_umap.shape)
                    # u = u[len(mis_index):]
                    # print("u_map train", u.shape)
                    
                    
                    

                    for min_cluster_size in [min_cluster_s]: # minimum cluster size 
                        with open(f'{job_id}output_of_{ii}_{jj}.txt', 'a') as file:
                            file.write(f'i(first reduction)={i} j(second reduction)={j} k(k-neighbors for first)={k} o(k-neighbors for second)={o} n_n(min distance)={n_n} minimum cluster size(min_cluster_size):{min_cluster_size} \n')
                            
                        # print(f'i(first reduction)={i} j(second reduction)={j} k(k-neighbors for first)={k} o(k-neighbors for second)={o} n_n(min distance)={n_n} minimum cluster size(min_cluster_size):{min_cluster_size}')
                        logger.info(f'i(first reduction)={i} j(second reduction)={j} k(k-neighbors for first)={k} o(k-neighbors for second)={o} n_n(min distance)={n_n} minimum cluster size(min_cluster_size):{min_cluster_size}')
                        
                        logger.info('start clustering')
                        
                        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, gen_min_span_tree=True).fit(u)
                        
                        logger.info('end clustering')
                        
                        labels = hdbscan_clusterer.labels_
                        DBCV_score = hdbscan_clusterer.relative_validity_

                        # print("noisy:", list(labels).count(-1)) # noisy data save with value -1
                        logger.info(f"noisy:{list(labels).count(-1)}") # noisy data save with value -1
                        # print('number of clusters: ', len( set(list(labels))))
                        logger.info(f'number of clusters: {len(set(list(labels)))}')
                        # print('DBCV score:', DBCV_score)
                        logger.info(f'DBCV score: {DBCV_score}')
                        flag = 0
                        try:
                            clustering_score = silhouette_score(u, labels)
                        except:
                            flag = 1
                            
                        
                        print("Silhouette score", clustering_score)
                        logger.info(f"Silhouette score {clustering_score}")
                        
                        with open(f'{job_id}output_of_{ii}_{jj}.txt', 'a') as file:
                            file.write(f"noisy:{list(labels).count(-1)} \n") # noisy data save with value -1
                            file.write(f'number of clusters: {len(set(list(labels)))}  \n')
                            file.write(f'DBCV score: {DBCV_score}  \n')
                            file.write(f"Silhouette score {clustering_score}  \n")
                            
                        
                        if flag:
                            logger.warning(f'silhouette score exception')
                            continue  
                            
                        
                        
                        # all_clustering_info['i_j'].append((i,j))
                        # all_clustering_info['k_o'].append((k,o))
                        # all_clustering_info['n_n'].append(n_n)
                        # all_clustering_info['metric'].append(metric)
                        # all_clustering_info['first_umap_transformer'].append(first_umap_reducer)
                        # all_clustering_info['second_umap_transformer'].append(second_umap_reducer)
                        
                        # all_clustering_info['min_cluster_size'].append(min_cluster_size)
                        # all_clustering_info['clusterer_object'].append(hdbscan_clusterer)
                        # all_clustering_info['Silhouette_score'].append(clustering_score)
                        # all_clustering_info['DBCV_score'].append(DBCV_score)
                        # all_clustering_info['clustering_label'].append(labels)
                        # all_clustering_info['hdbscan_in_umap'].append(u)
                        
                        # save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
                        
                        # try:
                        #     if not os.path.exists(save_path_dir):
                        #         os.makedirs(save_path_dir)
                        # except:
                        #     pass
                        
                        postfix_name = f'{i}_{j}_{k}_{o}_{n_n}_{min_cluster_size}'
                        # with open(os.path.join(save_path_dir, f"all_clusters_info_{postfix_name}.pkl"), "wb") as file:
                        #     pickle.dump(all_clustering_info, file)
                        #     logger.info('saved to pkl file')

                        # all_clustering_info = {'i_j':[],
                        #     'k_o':[],
                        #     'n_n':[],
                        #     'metric':[],
                        #     'first_umap_transformer':[],
                        #     'second_umap_transformer':[],
                        #     'min_cluster_size':[],
                        #     'clusterer_object':[],
                        #     'Silhouette_score':[],
                        #     'DBCV_score':[],
                        #     'clustering_label':[],
                        #     'hdbscan_in_umap':[]
                        #     }
                        
                        
                        #########################################################################################
                        ########################## save clusters for training set ###############################
                        #########################################################################################
                        logger.info('saving clusster of faults for training set')
                        # train_labels = all_clusters_info['clustering_label'][selected_index]
                        train_labels = labels
                        # hdbscan_in_umap_train = all_clusters_info['hdbscan_in_umap'][selected_index]
                        saving_training_key = 'last_epoch_training'
                        train_clusters = {saving_training_key: {'clustering_label': np.array(train_labels),
                                                                'hdbscan_in_umap': [],
                                                                'mis_test_index': [], 
                                                                'mis_train_index':mis_tindex, 
                                                                'number_of_training_faults':0, 
                                                                'silhouette_score':clustering_score,
                                                                'DBCV_score':DBCV_score}}

                        save_path_dir = os.path.join(root, "output", "fault_clusters",'training_set', model_name)
                        
                        try:
                            if not os.path.exists(save_path_dir):
                                os.makedirs(save_path_dir)
                        except:
                            pass
                        
                        np.save(os.path.join(save_path_dir, f"output_of_clustering__{postfix_name}.npy"), train_clusters)
                        logger.info('saving clusster of faults for training set completed!!')
                        
                        
                        
                        
                        
                        
                        #########################################################################################
                        ########################## save clusters for test set ###############################
                        #########################################################################################
                        u = np.load(os.path.join(root, "output", "mispredicteds", "inceptionV3_imagenet", "mispredicted_X_scf_test.npy"))
                        
                        
                        u = first_umap_reducer.transform(u)
                        u = second_umap_reducer.transform(u)
                        
                        # hdbscan_clusterer = all_clusters_info['clusterer_object'][selected_index]
                        test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusterer, u)
                        
                        print('number of clusters of test faults:', len(set(list(test_labels))))
                        
                        saving_test_key = 'last_epoch_test'
                        test_clusters = {saving_test_key: {'clustering_label': np.array(test_labels), 'hdbscan_in_umap': [], 'mis_test_index': mis_index, 'mis_train_index':[], 'number_of_training_faults':0}}
                        
                        
                        save_path_dir = os.path.join(root, "output","fault_clusters",'test_set', model_name)    
                            
                        if not os.path.exists(save_path_dir):
                            os.makedirs(save_path_dir)
                            
                        np.save(os.path.join(save_path_dir, f"output_of_clustering__{postfix_name}.npy"), test_clusters)
                        logger.info('saving clusster of faults for test set completed!!')
                        
                        # if True:
                        # if (list(labels).count(-1) <= len_x_mis * 0.10): ## the number of noise should be less than 10% of mispredicteds  
                
                        #     # my_trace = [i, j, k, o, min_cluster_size, labels.max()+1, list(labels).count(-1)]
                        #     # trace.append(my_trace)
                            
                            
                        #     if len(set(labels)) >= 2: # if number of clusters >= 2
                        #         clustering_score = silhouette_score(x_mis, labels)
                        #         print("Silhouette score", clustering_score)
                                
                        #     if (clustering_score >= 0.60):
                        #         # print('---------------------------------')
                        #         # Clustering_Label.append(labels)
                        #         # # print("ll", ll)
                        #         # umap_min_cluster_size = [u, min_cluster_size]
                        #         # hdbscan_in_umap.append(umap_min_cluster_size)
                        #         ss_trace = [(i, j), k, n_n, min_cluster_size, labels.max() + 1, list(labels).count(-1), clustering_score]
                        #         # clusterer_object.append(copy.deepcopy(hdbscan_clusterer))
                        #         # # x_mis_test_u.append(test_umap) 
                        #         # ss_tt.append(ss_trace)
                        #         print(ss_trace)
                                
                        #         # test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusterer, test_umap)
                        #         # print('prediction of test mispredictions using fitted cluster')
                                
                        #         # print(test_labels)
                        #         # print(len(set(test_labels)))
                                
                        #         print('@@@@@@@@@@')
                            
        output_of_clustering[key] = {'clustering_label':Clustering_Label, 'hdbscan_in_umap':hdbscan_in_umap, 'ss_tt':ss_tt, 'clusterer_object': clusterer_object}

    return output_of_clustering, all_clustering_info




def select_high_score_cluster(clusters):
    best_cluster = {}
    max_score = 0
    best_clusterer_object = None
    for key,value in clusters.items():
        for i in range(len(value['ss_tt'])):
            ss_tt = value['ss_tt'][i]
            if max_score <= ss_tt[6]:
                max_score = ss_tt[6]
                print(max_score)
                best_cluster[key] = {'clustering_label':value['clustering_label'][i], 'hdbscan_in_umap':value['hdbscan_in_umap'][i], 'ss_tt':value['ss_tt'][i]}
                best_clusterer_object = value['clusterer_object'][i]
                

    return best_cluster, best_clusterer_object









def find_similar_clusters(x_mis, tt, tst, mis_index, model_name, training_key):
    '''
    x_mis: numpy array of shape (n_misclassified_samples, feature_dim) containing misclassified samples
    y_true: list of true labels of misclassified samples from both test and training set
    y_pred: list of predicted labels of misclassified samples from both test and training set
    mis_test_index: list of indices of misclassified samples from the test set
    mis_train_index: list of indices of misclassified samples from the training set
    '''
    dir_path = os.path.join(root, 'output','fault_clusters', 'training_set', model_name)
    output_of_clustering = np.load(os.path.join(dir_path, "output_of_clustering.npy"), allow_pickle=True)

    
    clustering_info_dic = output_of_clustering.item().get(training_key)
    print(clustering_info_dic)
    hdbscan_in_umap = clustering_info_dic['hdbscan_in_umap']
    clustering_label = clustering_info_dic['clustering_label']
    mis_index = clustering_info_dic['mis_test_index'] # index of mispredicted on test
    mis_tindex = clustering_info_dic['mis_train_index'] # index of mispredicted on train
    x_mis_test_u = clustering_info_dic['x_mis_test_umap'] # test u-map
    
    saving_key = training_key.replace('training','test')
    
    

    print(hdbscan_in_umap[0].shape)
    print(len(mis_tindex))
    
    # print(output_of_clustering)
    clusters_set = {}
    for cl, mis_x_train, mis_x_index in zip(clustering_label, hdbscan_in_umap[0], mis_tindex):
        if cl in clusters_set.keys():
            clusters_set[cl]['mis_x_train'].append(mis_x_train)
            clusters_set[cl]['mis_x_index'].append(mis_x_index)
        else:
            clusters_set[cl] = {'mis_x_train': [mis_x_train], 'mis_x_index': [mis_x_index]}
            
    
    print('NUMBER OF CLUSTERS BASED ON TRAINING MISPREDICTIONS IS:',len(clusters_set))
    
    print(len(x_mis))
    
    
    
    
    ## find cluster of test data using fitted hdbscan cluster
    # load clusterer 
    
    save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
    with open(os.path.join(save_path_dir, "hdbscan_clusterer_model.pkl"), "rb") as file:
        hdbscan_clusterer = pickle.load(file)
    
    test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusterer, x_mis_test_u)
    print('=+=+=+=')
    print(model_name)
    print(test_labels)
    print(strengths)
    print('=+=+=+=')
    ll, s = hdbscan.approximate_predict(hdbscan_clusterer, hdbscan_in_umap[0])
    print(ll)
    print('************************')
    
    

    test_clusters = {saving_key: {'clustering_label': np.array(test_labels), 'hdbscan_in_umap': x_mis_test_u, 'mis_test_index': mis_index, 'mis_train_index':[], 'number_of_training_faults':len(set(clustering_label))-1}}
    
    save_path_dir = os.path.join(root, "output","fault_clusters",'test_set', model_name)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
        
    np.save(os.path.join(save_path_dir, "output_of_clustering.npy"), test_clusters)
    

    
    
    # testing_clusters = {'clustering_label': np.array([200, 403,  -1, 88, 304, 129]), 'hdbscan_in_umap': [u,n_n], 'mis_test_index':[], 'mis_train_index':[]}
    
    
    
    
 
def select_and_save_best_training_fault_cluster(mis_index_train, model_name, saving_training_key, selected_index = None, do_saving=1):
    
    save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
    with open(os.path.join(save_path_dir, "all_clusters_info.pkl"), "rb") as file:
        all_clusters_info = pickle.load(file)

    if selected_index == None:
        silhouette_score_list = list(all_clusters_info['Silhouette_score'])
        max_silhouette_score = max(silhouette_score_list)
        selected_index = silhouette_score_list.index(max_silhouette_score)

    train_labels = all_clusters_info['clustering_label'][selected_index]
    hdbscan_in_umap_train = all_clusters_info['hdbscan_in_umap'][selected_index]
    train_clusters = {saving_training_key: {'clustering_label': np.array(train_labels), 'hdbscan_in_umap': hdbscan_in_umap_train, 'mis_test_index': [], 'mis_train_index':mis_index_train, 'number_of_training_faults':0}}

    save_path_dir = os.path.join(root, "output","fault_clusters",'training_set', model_name)
    
    try:
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
    except:
        pass
    
    
    if do_saving:
        np.save(os.path.join(save_path_dir, "output_of_clustering.npy"), train_clusters)
    
    print('number of clusters of train faults (highest silhoutte score):', len(set(list(train_labels))))
    
    return selected_index

def assign_cluster_to_test_input(x_mis, mis_index, model_name, saving_test_key, selected_index = None, diff_test_data_path=''):
    '''
    x_mis: numpy array of shape (n_misclassified_samples, feature_dim) containing misclassified samples
    y_true: list of true labels of misclassified samples from both test and training set
    y_pred: list of predicted labels of misclassified samples from both test and training set
    mis_test_index: list of indices of misclassified samples from the test set
    mis_train_index: list of indices of misclassified samples from the training set
    '''
        
    save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
    with open(os.path.join(save_path_dir, "all_clusters_info.pkl"), "rb") as file:
        all_clusters_info = pickle.load(file)

    if selected_index == None:
        silhouette_score_list = list(all_clusters_info['Silhouette_score'])
        max_silhouette_score = max(silhouette_score_list)
        selected_index = silhouette_score_list.index(max_silhouette_score)

    first_umap_transformer = all_clusters_info['first_umap_transformer'][selected_index]
    second_umap_transformer = all_clusters_info['second_umap_transformer'][selected_index]
    
    u1 = first_umap_transformer.transform(x_mis)
    u = second_umap_transformer.transform(u1)
    
    hdbscan_clusterer = all_clusters_info['clusterer_object'][selected_index]
    test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusterer, u)
    
    print('number of clusters of test faults:', len(set(list(test_labels))))

    
    
    test_clusters = {saving_test_key: {'clustering_label': np.array(test_labels), 'hdbscan_in_umap': u, 'mis_test_index': mis_index, 'mis_train_index':[], 'number_of_training_faults':0}}
    
    if shifted_test_input:
        save_path_dir = os.path.join(root, "output","fault_clusters",'test_set', model_name, dataset_name)
    else:   
        save_path_dir = os.path.join(root, "output","fault_clusters",'test_set', model_name)    
        
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
        
    np.save(os.path.join(save_path_dir, "output_of_clustering.npy"), test_clusters)
    
    print('test fault clusters saved!')
    
    
    

if __name__ == '__main__':
    
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-model_name",
                        type=str,
                        help="lenet5, cifar10, lenet4, lenet5_SVHN")

    parser.add_argument("--all_epoch", "-all_epoch",
                        type=int,
                        default=0,
                        help="if 1 so use all epochs instead of last epoch else use last epoch")

    parser.add_argument("--source_data_for_clustering", "-source_data_for_clustering",
                        type=str,
                        default='training',
                        help="choose from list ['training', 'test', 'mix']")
    
    parser.add_argument("--start_from_epoch_num", "-start_from_epoch_num",
                        type=int,
                        default=1,
                        help="start from epoch that you want to consider in all epoch mispredicteds")

    parser.add_argument("--diff_test_data_path", "-diff_test_data_path",
                        type=str,
                        default='', # other value: 'diff_test_data'
                        help="for using of diff test data acheived from finetuing differential testing")
    
    parser.add_argument("--shifted_test_input", "-shifted_test_input",
                        type=int,
                        default=0,
                        help="if you want to test the prediction model on shifted data just set up this to 1")
    
    parser.add_argument("--dataset_name", "-dataset_name",
                        type=str,
                        default='',
                        help="dataset name")
    
    parser.add_argument("--ii", "-ii", type=int)
    parser.add_argument("--jj", "-jj", type=int)
    parser.add_argument("--min_cluster_size", "-min_cluster_size", type=int)

    #


    args = parser.parse_args()
    model_name = args.model_name
    all_epoch = args.all_epoch
    source_data_for_clustering = args.source_data_for_clustering
    start_from_epoch_num = args.start_from_epoch_num
    diff_test_data_path = args.diff_test_data_path
    
    ## for adding test with different distribution of train
    shifted_test_input = args.shifted_test_input
    dataset_name = args.dataset_name
    
    
    ii = args.ii
    jj = args.jj
    min_cluster_size = args.min_cluster_size
    
    print(diff_test_data_path)
    

    key = ''
    if all_epoch:
        x_mis, tt, tst, mis_index, mis_tindex = mix_unique_mispredicteds_of_all_epochs(model_name,from_epoch=start_from_epoch_num, diff_test_data_path=diff_test_data_path)
        key = 'all_epoch'
        print(len(x_mis))
        print(key)
        
    else:
        x_mis, tt, tst, mis_index, mis_tindex = load_data_for_clustering_last_epoch(model_name)
        key = 'last_epoch'

    print('model name:',model_name)
    print('number of test mispredicteds:',len(mis_index))
    print('number of train mispredicteds:',len(mis_tindex))


    if source_data_for_clustering == 'training': # need clustering
        # x_mis_train = x_mis[len(mis_index):]
        x_mis_train = []
        # x_mis_train = np.load(os.path.join(root, "output", "mispredicteds", "inceptionV3_imagenet", "mispredicted_X_scf_train.npy"))
        
        # tt = tt[len(mis_index):]
        # tst = tst[len(mis_index):]
        training_key = key + '_training'
        prefix_path = 'training_set'
        
        
        
        
        do_clustering = True # set False if you have training clusters and just want to assign mispredicted test inputs
        
        if do_clustering:
        
            clusters, all_clusters_info = run_clustering(x_mis_train, tt, tst, mis_index, mis_tindex, key, ii, jj, min_cluster_s=min_cluster_size)
            # clusters = run_clustering_without_umap(x_mis, tt, tst, mis_index, mis_tindex, key)
                    
            
            # best_cluster, best_clusterer_object = select_high_score_cluster(clusters)
            # best_cluster[key]['mis_test_index'] = mis_index
            # best_cluster[key]['mis_train_index'] = mis_tindex
            
            
            # # save best cluster
            save_path_dir = os.path.join(root, "output","fault_clusters", prefix_path, model_name)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            # np.save(os.path.join(save_path_dir, "output_of_clustering.npy"), best_cluster)
            
            #save clusterer object
            save_path_dir = os.path.join(root, "output","clusters_object", prefix_path, model_name)
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
                
            # with open(os.path.join(save_path_dir, "hdbscan_clusterer_model.pkl"), "wb") as file:
            #     pickle.dump(best_clusterer_object, file)
                
            
            
            with open(os.path.join(save_path_dir, "all_clusters_info.pkl"), "wb") as file:
                pickle.dump(all_clusters_info, file)
            
            print('save all clusters info')

        print('============= change the code for selecting best for imagenet =========================')
        best_indeces_of_clusters = {'resnet50_office31_mix': 21}
        # best_indeces_of_clusters = {'resnet50_caltech256': 78}
        best_indeces_of_clusters = {'resnet50_caltech256': 3}
        # best_indeces_of_clusters = {'resnet50_caltech256': 146}
        

        best_index = None
        if model_name in best_indeces_of_clusters.keys():
            best_index = best_indeces_of_clusters[model_name] # see cluster_scores.csv file ### this is for resnet50_office31_mix - old best index was 124
            print(f'best index of {model_name} is:', best_index)
        
        do_saving = 1
        if shifted_test_input:
            do_saving = 0
            
        selected_index = select_and_save_best_training_fault_cluster(mis_index_train=mis_tindex, model_name=model_name, saving_training_key=training_key, selected_index= best_index, do_saving=do_saving)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('best selected index:',selected_index)

        ## now we can assign cluster to mispredictions input of test data
        # x_mis_test = x_mis[:len(mis_index)]
        x_mis_test = np.load(os.path.join(root, "output", "mispredicteds", "inceptionV3_imagenet", "mispredicted_X_scf_test.npy"))
        
        print('number of test mispredicted:', x_mis_test)
        # tt = tt[:len(mis_index)] # true label
        # tst = tst[:len(mis_index)] # predicted label
        test_key = key + '_test'
        prefix_path = 'test_set'
        
        assign_cluster_to_test_input(x_mis_test, mis_index, model_name, test_key, selected_index= selected_index, diff_test_data_path=diff_test_data_path)
        
    
    


    
    
    # elif source_data_for_clustering == 'test': # using the same clusters of training set
    #     x_mis = x_mis[:len(mis_index)]
    #     tt = tt[:len(mis_index)] # true label
    #     tst = tst[:len(mis_index)] # predicted label
    #     # key = key + '_test'
    #     prefix_path = 'test_set'
        
    #     # load clusters of training set
    #     key = key + '_training'
    #     find_similar_clusters(x_mis, tt, tst, mis_index, model_name, key)
        
        
    # elif source_data_for_clustering == 'mix':
    #     key = key + '_mix'
    #     prefix_path = 'mix'
        
    #     print('key:',key)
        
    #     clusters = run_clustering(x_mis, tt, tst, mis_index, mis_tindex, key)      
        
    #     best_cluster = select_high_score_cluster(clusters)
    #     best_cluster[key]['mis_test_index'] = mis_index
    #     best_cluster[key]['mis_train_index'] = mis_tindex
        
        
    #     save_path_dir = os.path.join(root, "output","fault_clusters", prefix_path, model_name)
    #     if not os.path.exists(save_path_dir):
    #         os.makedirs(save_path_dir)
            
    #     np.save(os.path.join(save_path_dir, "output_of_clustering.npy"), best_cluster)

    
    print('model name:',model_name)
    print('number of test mispredicteds:',len(mis_index))
    print('number of train mispredicteds:',len(mis_tindex))
    
    end_time = datetime.now()
    with open('cost_of_TEASMA.txt', 'a') as file:
        file.write(f"Time for fault extraction using clustering for subject {model_name} is = {end_time - start_time}.\n")    
    
    
