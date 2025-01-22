
import os
import pandas as pd


# import hdbscan
# import umap.umap_ as umap

import pickle

root = os.path.dirname(os.path.abspath(__file__))


subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN']
subjects = ['inceptionV3_imagenet']
for model_name in subjects:
    print(model_name)
    save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
    # save_path_dir = os.path.join(root, "output","clusters_object_old_10_oct", 'training_set', model_name)
    with open(os.path.join(save_path_dir, "all_clusters_info_100_50_10_10_0.01_5.pkl"), "rb") as file:
        all_clusters_info = pickle.load(file)

    print('load all cluster')
    num_of_clusters = []
    noise = []
    for labels in all_clusters_info['clustering_label']:
        num_of_clusters.append(len(set(list(labels))))
        noise.append(list(labels).count(-1))
        
    data = {'i_j':all_clusters_info['i_j'], 'k_o':all_clusters_info['k_o'], 'n_n':all_clusters_info['n_n'],
            'min_cluster_size':all_clusters_info['min_cluster_size'], 'Silhouette_score':all_clusters_info['Silhouette_score'],
            'DBCV_score':all_clusters_info['DBCV_score'], 'num_of_clusters':num_of_clusters, 'noise':noise, 'metric':all_clusters_info['metric']}

    df = pd.DataFrame(data)
    
    df.to_csv(os.path.join(save_path_dir,'cluster_scores.csv'))


# model_name = 'cifar10'
# save_path_dir = os.path.join(root, "output","clusters_object", 'training_set', model_name)
# with open(os.path.join(save_path_dir, "hdbscan_clusterer_model.pkl"), "rb") as file:
#     all_clusters_info = pickle.load(file)

#     print(all_clusters_info)
