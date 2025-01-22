import os
import argparse
import numpy as np

dataset = 'imagenet'
subject = 'inceptionV3_imagenet'
from dnn_tip.surprise import LSA, DSA
from tqdm import tqdm
from datetime import datetime


def my_lsa(train_ATs, start_index, end_index, LSA_save_path):
    lsa_output = []
    counter = 0
    base_step = 1
    
    # for i in tqdm(range(start_index, end_index)):
    for i in tqdm(range(50000)):
        s = datetime.now()
        
        ii = start_index + (i * base_step)
        jj = start_index + ((i+1) * base_step)
        
        
        
        if i == 50000-1:
            jj = end_index
        
        test_ATs = train_ATs[ii:jj]
        sa = LSA(np.concatenate((train_ATs[0:ii], train_ATs[jj:]), axis=0))
        lsa_one_input = sa(test_ATs)
        lsa_output.append(lsa_one_input)
        
        # if (i+1) % 10 == 0:
        #     np.save(os.path.join(LSA_save_path,f'train_lsa_{start_index}_{end_index}_{i}.npy'), np.array(lsa_output))

        #     print(f'part {i} saved! ', datetime.now() - s)
        
        print(f'part {i} done {jj}! ', datetime.now() - s)
        
        if jj >= end_index:
            break
    
    return np.array(lsa_output)

def my_dsa(train_ATs, train_pred, start_index, end_index, DSA_save_path):
    dsa_output = []
    counter = 0
    base_step = 1 ## 25000/base_step(100) = 250 
    batch_size = 25000
    iters = batch_size//base_step 
    
    # for i in tqdm(range(start_index, end_index)):
    for i in tqdm(range(iters)):
        s = datetime.now()
        
        ii = start_index + (i * base_step)
        jj = start_index + ((i+1) * base_step)
        
        if i == iters-1:
            jj = end_index
        
        test_ATs = train_ATs[ii:jj]
        test_pred = train_pred[ii:jj]
        
        
        sa = DSA(np.concatenate((train_ATs[0:ii], train_ATs[jj:]), axis=0), np.concatenate((train_pred[0:ii], train_pred[jj:]), axis=0))
        
        
        dsa_one_input = sa(test_ATs, test_pred)
        dsa_output.append(dsa_one_input)
        
        # if (i+1) % 100 == 0:
        #     np.save(os.path.join(DSA_save_path,f'train_dsa_{start_index}_{end_index}_{i}.npy'), np.array(dsa_output))

        #     print(f'part {i} saved! ', datetime.now() - s)
        
        print(f'part {i} done! ', datetime.now() - s)
        
        if jj >= end_index:
            break
    
    return np.array(dsa_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    
    parser.add_argument(
        "--save_path", "-save_path", help="the path of LSA and DSA saved", type=str, default="SA_results"
    )
    
    
    parser.add_argument("--start_index", "-start_index",
                        type=int,
                        default=0,
                        help="start index of training set")
    
    
    parser.add_argument("--end_index", "-end_index",
                        type=int,
                        default=-1,
                        help="end index of training set")
    
    parser.add_argument("--sampling_from", "-sampling_from",
                        type=str,
                        default= '',
                        help="training_set or test_set")
    
    
    args = parser.parse_args()
    
    print('-'*25)
    print(args)
    print('-'*25)
    
    start_index = args.start_index
    end_index = args.end_index
    sampling_from = args.sampling_from
    save_path = os.path.join(args.save_path, subject)
    
    
    train_ATs = np.load("SA_results/inceptionV3_imagenet/ATS_predicteds/train_ats.npy") ## to have this file you have to run get_imagenet_activation_prediction.py
    test_ATs = np.load("SA_results/inceptionV3_imagenet/ATS_predicteds/test_ats.npy") ## to have this file you have to run get_imagenet_activation_prediction.py

    train_pred = np.load('SA_results/inceptionV3_imagenet/ATS_predicteds/all_actual_labels_train.npy')
    test_pred = np.load('SA_results/inceptionV3_imagenet/ATS_predicteds/all_actual_labels_validation.npy')

    if end_index < 0:
        end_index = len(train_pred)
    
    if start_index < 0:
        start_index = len(train_pred) + start_index

    try:
        DSA_save_path = os.path.join(save_path,'DSA', sampling_from)
        if not os.path.exists(DSA_save_path):
            os.makedirs(DSA_save_path)
    except:
        pass
    
    try:
        LSA_save_path = os.path.join(save_path,'LSA', sampling_from)
        if not os.path.exists(LSA_save_path):
            os.makedirs(LSA_save_path)
    except:
        pass
    
    if sampling_from == 'training_set':
        if args.lsa:
            lsa_output = my_lsa(train_ATs, start_index, end_index, LSA_save_path)
            np.save(os.path.join(LSA_save_path,f'train_lsa_{start_index}_{end_index}.npy'), lsa_output)   
            print('lsa saved!')         
        if args.dsa:
            dsa_output = my_dsa(train_ATs, train_pred, start_index, end_index, DSA_save_path)
            np.save(os.path.join(DSA_save_path,f'train_dsa_{start_index}_{end_index}.npy'), dsa_output)     
            
            print('dsa saved!')         
                     
    else: # for test
        if args.lsa:
            sa = LSA(train_ATs)
            lsa_output = sa(test_ATs)
            np.save(os.path.join(LSA_save_path,'all_test_lsa.npy'), lsa_output)
            
        if args.dsa:
            sa = DSA(train_ATs, train_pred) 
            dsa_output = sa(test_ATs, test_pred) 
            np.save(os.path.join(DSA_save_path,'all_test_dsa.npy'), dsa_output)
            
    
    
    