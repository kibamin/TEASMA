import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.models import load_model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from load_dataset import load_dataset
import os



_LAYER_INDEX_NAME = {'lenet5_mnist': ['dense_1'],
                     'cifar10': ['dense'],
                     'lenet4': ['dense'],
                     'resnet20_cifar10': ['activation_18'],
                     'lenet5_SVHN': ['dense_1'],
                     'vgg16_SVHN': ['fc2'],
                     'resnet152_cifar100': ['dense'],
                     'inceptionV3_imagenet':['activation_93']} 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="SA_results"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=float,
        default= 1e-5 ### for resnet152_cifar100 thr=0.02 , for other 1e-5 = 0.00001
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
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    
    parser.add_argument(
        "--subject",
        "-subject",
        help="subject name most of case it is name of model plus dataset",
        type=str,
    )
    
    args = parser.parse_args()
    
    print('-'*25)
    print(args)
    print('-'*25)
    
    dataset = args.d
    subject = args.subject
    
    args.save_path = os.path.join(args.save_path, subject)
    
    try:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    except:
        print(f'The {args.save_path} exists')    
    
    print(args.save_path)
    
    
    (x_train, y_train), (x_test, y_test), num_classes = load_dataset(subject=subject, data_type=dataset)
    args.num_classes = num_classes
       
    model = load_model(os.path.join("models", subject, "original_model.h5"))
    model.summary()
    # print_activation_layer_names(model)
    layer_names = _LAYER_INDEX_NAME[subject]
    joined_layer_names = "_".join(layer_names)
    
    assert layer_names, 'please select at least one layer'
    
    
    
    

    
    
    if args.lsa:
        lsa_save_path = os.path.join(args.save_path, 'LSA')
        try:
            os.makedirs(lsa_save_path)
        except:
            pass
        
        #----------------------#
        #      LSA Train
        #----------------------#
        
        train_lsa_save_path = os.path.join(lsa_save_path, f'train_lsa_{joined_layer_names}_new.npy')
        if os.path.exists(train_lsa_save_path):
            train_lsa = np.load(train_lsa_save_path)
        else:
            train_lsa = fetch_lsa(model, x_train, x_train, "train", layer_names, args)
            np.save(train_lsa_save_path, train_lsa)
            print('train_lsa saved!')  
                  
        train_cov = get_sc(np.amin(train_lsa), args.upper_bound, args.n_bucket, train_lsa)
        print('LSC of train:', train_cov)
        
        
        #----------------------#
        #      LSA Test
        #----------------------#
        
        
        
        test_lsa_save_path = os.path.join(lsa_save_path, f'test_lsa_{joined_layer_names}.npy')   
        if os.path.exists(test_lsa_save_path):
            test_lsa = np.load(test_lsa_save_path)
        else:
            test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
            np.save(test_lsa_save_path, test_lsa)
            print('test_lsa saved!') 
        
        test_cov = get_sc(np.amin(test_lsa), args.upper_bound, args.n_bucket, test_lsa)
        print('LSC of test:', test_cov)
        


    elif args.dsa:
        dsa_save_path = os.path.join(args.save_path, 'DSA')
        try:
            os.makedirs(dsa_save_path)
        except:
            pass
        
        #----------------------#
        #      DSA Train
        #----------------------#
        
        train_dsa_save_path = os.path.join(dsa_save_path, f'train_dsa_{joined_layer_names}_new.npy') ## _new is for when we removed target point from training set ATs because it leads to dsa = 0
        if os.path.exists(train_dsa_save_path):
            train_dsa = np.load(train_dsa_save_path)
        else:
            train_dsa = fetch_dsa(model, x_train, x_train, "train", layer_names, args)
            np.save(train_dsa_save_path, train_dsa)
            print('train_dsa saved!')   
                  
        train_cov = get_sc( np.amin(train_dsa), args.upper_bound, args.n_bucket, train_dsa)
        print('DSC of train:', train_cov)
        
        
        
        #----------------------#
        #      DSA Test
        #----------------------#
        
        test_dsa_save_path = os.path.join(dsa_save_path, f'test_dsa_{joined_layer_names}.npy')   
        if os.path.exists(test_dsa_save_path):
            test_dsa = np.load(test_dsa_save_path)
        else:
            test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
            np.save(test_dsa_save_path, test_dsa)
            print('test_dsa saved!')   
            
        
        test_cov = get_sc(np.amin(test_dsa), args.upper_bound, args.n_bucket, test_dsa)
        print('DSC of test:', test_cov)
        
    
        
        
