import argparse
from datetime import datetime
import os
import numpy as np
import csv
from sa import get_sc

## import constant variable
from run import _LAYER_INDEX_NAME



_LAYER_INDEX_NAME = {'lenet5_mnist': ['dense_1'],
                     'cifar10': ['dense'],
                     'lenet4': ['dense'],
                     'resnet20_cifar10': ['activation_18'],
                     'lenet5_SVHN': ['dense_1'],
                     'vgg16_SVHN': ['fc2'],
                     'resnet152_cifar100': ['dense'],
                     'inceptionV3_imagenet':['activation_93']
                     } 

_LSA_MANUAL_LOWER_UPPER_BOUNDS_TEST = {
                    'lenet5_mnist': {'L':-70, 'U':500},
                    'cifar10': {'L':450, 'U':1000},
                    'lenet4': {'L':-70, 'U':500},
                    'resnet20_cifar10': {'L':-35, 'U':70},
                    'lenet5_SVHN': {'L':100, 'U':500},
                    'vgg16_SVHN': {'L':-600, 'U':1000},
                    'resnet152_cifar100': {'L':-180, 'U':3000},
                    'inceptionV3_imagenet': {'L':-55, 'U':200}
                    } 

# _LSA_MANUAL_LOWER_UPPER_BOUNDS_TRAIN = {
#                     'lenet5_mnist': {'L':-50, 'U':75000},
#                     'cifar10': {'L':450, 'U':25000},
#                     'lenet4': {'L':-30, 'U':2000},
#                     'resnet20_cifar10': {'L':-30, 'U':800},
#                     'lenet5_SVHN': {'L':100, 'U':5000},
#                     'vgg16_SVHN': {'L':-500, 'U':3000000},
#                     'resnet152_cifar100': {'L':150, 'U':5000}
#                     } 

_LSA_MANUAL_LOWER_UPPER_BOUNDS_TRAIN = {
                    'lenet5_mnist': {'L':-50, 'U':75000},
                    'cifar10': {'L':450, 'U':15000},
                    'lenet4': {'L':-30, 'U':2000},
                    'resnet20_cifar10': {'L':-30, 'U':800},
                    'lenet5_SVHN': {'L':100, 'U':4000},
                    'vgg16_SVHN': {'L':-500, 'U':3000000},
                    'resnet152_cifar100': {'L':150, 'U':5000},
                    'inceptionV3_imagenet': {'L':-55, 'U':200},
                    } 

root = os.path.join(os.path.dirname(os.path.abspath(__file__))) # root of project
base_path_of_outputs = root



def calculate_SC(subject, test_suite_indeces, sampling_from, args):
    
    layer_names = _LAYER_INDEX_NAME[subject]
    joined_layer_names = "_".join(layer_names)
    
    if args.lsa:
        
        lsa_load_path = os.path.join(args.sa_load_path, 'LSA')
        
        train_lsa_load_path = os.path.join(lsa_load_path, f'train_lsa_{joined_layer_names}_new.npy')
        
        if os.path.exists(train_lsa_load_path):
            train_lsa = np.load(train_lsa_load_path)
        #----------------------#
        #      LSA Train
        #----------------------#
        if sampling_from == 'training_set':
            
            train_lsa_load_path = os.path.join(lsa_load_path, f'train_lsa_{joined_layer_names}_new.npy')
            
            if os.path.exists(train_lsa_load_path):
                train_lsa = np.load(train_lsa_load_path)
                ### calculate lsa of subset
                test_suite_lsa = train_lsa[test_suite_indeces]
                test_suite_lsc = get_sc(_LSA_MANUAL_LOWER_UPPER_BOUNDS_TRAIN[subject]['L'], _LSA_MANUAL_LOWER_UPPER_BOUNDS_TRAIN[subject]['U'], args.n_bucket, test_suite_lsa)
                # test_suite_lsc = get_sc(np.amin(train_lsa), np.amax(train_lsa), args.n_bucket, test_suite_lsa)
                # test_suite_lsc = get_sc(np.amin(test_suite_lsa), np.amax(test_suite_lsa), args.n_bucket, test_suite_lsa)
            else:
                print(f'the train_lsa does not exist please execute it using run.py for this subject {subject}')  
                test_suite_lsc = -1
            
            
        
        #----------------------#
        #      LSA Test
        #----------------------#
        else:
            test_lsa_load_path = os.path.join(lsa_load_path, f'test_lsa_{joined_layer_names}.npy')   
            if os.path.exists(test_lsa_load_path):
                test_lsa = np.load(test_lsa_load_path)
                ### calculate lsa of subset
                test_suite_lsa = test_lsa[test_suite_indeces]
                # test_suite_lsc = get_sc(np.amin(test_lsa), np.amax(test_lsa), args.n_bucket, test_suite_lsa)
                # test_suite_lsc = get_sc(np.amin(train_lsa), np.amax(train_lsa), args.n_bucket, test_suite_lsa)
                test_suite_lsc = get_sc(_LSA_MANUAL_LOWER_UPPER_BOUNDS_TEST[subject]['L'], _LSA_MANUAL_LOWER_UPPER_BOUNDS_TEST[subject]['U'], args.n_bucket, test_suite_lsa)
                
                
                # test_suite_lsc = get_sc(np.amin(test_suite_lsa), np.amax(test_suite_lsa), args.n_bucket, test_suite_lsa)
                

            else:
                print(f'the test_lsa does not exist please execute it using run.py for this subject {subject}')  
                test_suite_lsc = -1
        
            
    if args.dsa:
        dsa_load_path = os.path.join(args.sa_load_path, 'DSA')
        
        train_dsa_load_path = os.path.join(dsa_load_path, f'train_dsa_{joined_layer_names}_new.npy')
        if os.path.exists(train_dsa_load_path):
            train_dsa = np.load(train_dsa_load_path)
        #----------------------#
        #      DSA Train
        #----------------------#
        if sampling_from == 'training_set':
            train_dsa_load_path = os.path.join(dsa_load_path, f'train_dsa_{joined_layer_names}_new.npy')
            if os.path.exists(train_dsa_load_path):
                train_dsa = np.load(train_dsa_load_path)
                ### calculate lsa of subset
                test_suite_dsa = train_dsa[test_suite_indeces]
                if subject == 'inceptionV3_imagenet':
                    lb_imagenet = 0.0
                    ub_imagenet = 2.5
                    test_suite_dsc = get_sc(lb_imagenet, ub_imagenet, args.n_bucket, test_suite_dsa)
                else:
                    test_suite_dsc = get_sc(np.amin(train_dsa), np.amax(train_dsa), args.n_bucket, test_suite_dsa)
                # test_suite_dsc = get_sc(np.amin(test_suite_dsa), np.amax(test_suite_dsa), args.n_bucket, test_suite_dsa)
            
            else:
                print(f'the train_dsa does not exist please execute it using run.py for this subject {subject}')    
                test_suite_dsc = -1
            
        
        
        #----------------------#
        #      DSA Test
        #----------------------#
        else:
            test_dsa_load_path = os.path.join(dsa_load_path, f'test_dsa_{joined_layer_names}.npy')   
            if os.path.exists(test_dsa_load_path):
                test_dsa = np.load(test_dsa_load_path)
                test_suite_dsa = test_dsa[test_suite_indeces]
                if subject == 'inceptionV3_imagenet':
                    lb_imagenet = 0.0
                    ub_imagenet = 2.5
                    test_suite_dsc = get_sc(lb_imagenet, ub_imagenet, args.n_bucket, test_suite_dsa)
                else:
                    test_suite_dsc = get_sc(np.amin(test_dsa), np.amax(test_dsa), args.n_bucket, test_suite_dsa)     
                # test_suite_dsc = get_sc(np.amin(train_dsa), np.amax(train_dsa), args.n_bucket, test_suite_dsa)
                
                # test_suite_dsc = get_sc(np.amin(test_suite_dsa), np.amax(test_suite_dsa), args.n_bucket, test_suite_dsa)
                
            
            else:
                print(f'the test_dsa does not exist please execute it using run.py for this subject {subject}')    
                test_suite_dsc = -1
        
    LSC = test_suite_lsc
    DSC = test_suite_dsc
    
    return LSC, DSC



def sc_runner(test_suite_size, dataset_name, subject, sampling_from, args):
    ratio = ''
    
    uniform_sampling = False ## if your subset samples are uniform please set flat to True
    
    batch_sizes = [test_suite_size]
    for bs in batch_sizes:
        
        bs_start_time = datetime.now()
        
        all_results = {'test_suite_size':[], 'test_suite':[], 'LSC':[], 'DSC':[], 'time_taken':[]}
        
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
                source_path_test_suites = os.path.join('data', dataset_name, sampling_from, 'uniform_sampled_test_suites', f'batch_size_{bs}')
            
        
            
        sampled_test_suites_name = os.listdir(source_path_test_suites) # if you are using dic get len
        
        use_one_dic_file = False
        if 'test_suites_all.npy' in sampled_test_suites_name:
            test_suites_all_dic = np.load(os.path.join(source_path_test_suites, 'test_suites_all.npy'), allow_pickle=True).item()
            sampled_test_suites_name = list(test_suites_all_dic.keys())
            use_one_dic_file = True
            
            
        
        ### new version of sampled_test_suites :  for each batch size we have just one file (test_suites_all.npy) instead 300 file and the file is a dictionary include 300 keys with same name of 300 files
        
        sampled_test_suites_name.sort()


        counter = 0
        counter_for_ts = -1 # using for multithread 
        for test_suite_name in sampled_test_suites_name:
            if not test_suite_name.endswith('.npy'):
                continue
            
            counter_for_ts += 1
            
                
            print(f'{counter_for_ts} :', test_suite_name, 'started!')
            
            # filter test suite by original model to get correctly predicteds inputs
            test_suite_path = os.path.join(source_path_test_suites, f'{test_suite_name}')
            
                
            if use_one_dic_file :
                test_suite_x_y_index = test_suites_all_dic[test_suite_name]
                x_test_suite, y_test_suite, test_suite_indeces = test_suite_x_y_index['x_test'], test_suite_x_y_index['y_test'], test_suite_x_y_index['indices']
            
            # else:   
            #     x_test_suite, y_test_suite, indeces = load_test_suite(test_suite_path, dataset_name)
                
                

            
            '''
            Calculating DSC and LSC
            '''
            start = datetime.now()
            print()
            lsc, dsc = calculate_SC(subject, test_suite_indeces, sampling_from, args)
            
            end = datetime.now()
            
            time_taken = end - start




            if uniform_sampling:
                # pass # because I ran some experiment without using 'uniform_sampling' so it is better to hold same path
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio,'uniform_sampling', f'bs_{bs}')
                # save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio, 'uniform_sampling', f'bs_{bs}')
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio, sampling_from,'uniform_sampling', f'bs_{bs}')
                                    
            else:  
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject, ratio, sampling_from,'non_uniform_sampling', f'bs_{bs}')

            
                
            if not os.path.exists(save_path_of_all_result):
                try:
                    os.makedirs(save_path_of_all_result)
                except:
                    print(f'the dir {save_path_of_all_result} has created before!')
                
            
            
            
            save_file_name = f'bs{bs}_result_SC.csv'
            
            
            with open(os.path.join(save_path_of_all_result,save_file_name), 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n')
                
                if f1.tell() == 0:
                    header = all_results.keys()
                    writer.writerow(header)
                    
                writer.writerow([ str(bs), str(test_suite_name), str(round(lsc,4)),str(round(dsc,4)), str(time_taken)])
                    
        bs_end_time = datetime.now()

        print(f'elapse time for bs {bs}:', bs_end_time-bs_start_time)

            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset",
                        help="Dataset",
                        type=str)
    
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    
    parser.add_argument(
        "--sa_load_path", "-sa_load_path", help="the path of LSA and DSA saved", type=str, default="SA_results"
    )
    
    
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    
    parser.add_argument(
        "--subject",
        "-subject",
        help="subject name most of case it is name of model plus dataset",
        type=str,
    )
    
    parser.add_argument("--test_suite_size", "-test_suite_size",
                        type=int,
                        help="test suite size for example 1500 or 500 or ...")
    
    parser.add_argument("--sampling_from", "-sampling_from",
                        type=str,
                        default= '',
                        help="training_set or test_set")
    
    
    args = parser.parse_args()
    
    print('-'*25)
    print(args)
    print('-'*25)
    
    dataset = args.dataset
    subject = args.subject
    test_suite_size = args.test_suite_size
    sampling_from = args.sampling_from
    
    args.sa_load_path = os.path.join(args.sa_load_path, subject)
    
    
    sc_runner(test_suite_size, dataset, subject, sampling_from, args)
    
    # python sc_runner.py --dataset mnist --subject lenet5_mnist -lsa -dsa --sampling_from training_set --test_suite_size 100 --n_bucket 1000 
    # python sc_runner.py --dataset mnist --subject inceptionV3_imagenet -lsa -dsa --sampling_from training_set --test_suite_size 100 --n_bucket 1000 
    