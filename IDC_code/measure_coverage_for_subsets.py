import argparse
import os
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import datasets, transforms
from disvae.utils.modelIO import load_model, load_metadata
from utils.helpers import get_device, set_seed, get_config_section, FormatterNoDuplicate
from utils.datasets import ImageNet
from helper import *
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as dset
from timeit import default_timer as timer
import imageio

from datetime import datetime
import csv


#Global variable declarations
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())

root = os.path.join(os.path.dirname(os.path.abspath(__file__))) # root of project
base_path_of_outputs = root

class MyDataset(Dataset):
    def __init__(self, data, targets, channels, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.channels = channels
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            if self.channels == 3:
                x = Image.fromarray(self.data[index].astype(np.uint8), 'RGB')
            else:
                x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)

#Calculate mean and log variance vectors output by the encoder for the test inputs
def evaluate(model, testloader, logger):
    initialize = True
    for data, _ in tqdm(testloader, leave=False, disable=not default_config['no_progress_bar']):
        data = data.to(device)
        recon_batch, latent_dist, latent_sample = model(data)
        if initialize:
            mu = latent_dist[0]
            sd = torch.exp(0.5 * latent_dist[1])
            initialize = False
        else:
            mu = torch.cat((mu, latent_dist[0]), 0)
            sd_temp = torch.exp(0.5 * latent_dist[1])
            sd = torch.cat((sd, sd_temp), 0)
      
    mu = mu.to('cpu')
    mu_np = mu.detach().numpy()
    
    sd = sd.to('cpu')
    sd_np = sd.detach().numpy()
    
    # logger.info("latent_dist mu_np shape for the mnist test dataset {}".format(mu_np.shape))
    return mu_np, sd_np

     
     
     
     
                     
default_config = get_config_section([CONFIG_FILE], "Custom")
description = 'Measure coverage over disentangled representations'
parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)
parser.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")  
parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')            
parser.add_argument('-b', '--no_bins', type=int, default=20,
                         help='no of bins.') 
parser.add_argument('-ways', '--ways', type=int, default=3,
                         help='ways')
parser.add_argument('-density', '--density', type=float, default=0.9999,
                         help='density')                         
parser.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')  
parser.add_argument('--batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')      
parser.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS) 
parser.add_argument('--dataset', type=str,
                            default="mnist", choices=["mnist", "cifar10", "fashion_mnist", "cifar100", "SVHN", "imagenet"],
                            help='dataset')
parser.add_argument('--path', type=str, default="None", help='path to the custom dataset in numpy file format')
                    

parser.add_argument("--test_suite_size", "-test_suite_size",
                    type=int,
                    help="test suite size for example 1500 or 500 or ...")

parser.add_argument("--sampling_from", "-sampling_from",
                    type=str,
                    default= '',
                    help="training_set or test_set")
                    

                         
args = parser.parse_args()

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(args.log_level.upper())
stream = logging.StreamHandler()
stream.setLevel(args.log_level.upper())
stream.setFormatter(formatter)
logger.addHandler(stream)
set_seed(args.seed)
device = get_device(is_gpu=not args.no_cuda)
exp_dir = os.path.join(RES_DIR, args.name)

test_suite_size = args.test_suite_size
sampling_from = args.sampling_from
load_train_split = True if sampling_from == 'training_set' else False


   
model = load_model(exp_dir, is_gpu=not args.no_cuda)
metadata = load_metadata(exp_dir)
                         
logger.info("Testing Device: {}".format(device))
print(f"VAE {args.name}")

if args.dataset == "mnist":
    print('load mnist dataset')
    testset = dset.MNIST(root="./data", train=load_train_split, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    
elif args.dataset == "cifar10":
    print('load cifar10 dataset') 
    testset = dset.CIFAR10(root="./data", download=True, train=load_train_split, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

elif args.dataset == "cifar100":
    print('load cifar100 dataset') 
    testset = dset.CIFAR100(root="./data", download=True, train=load_train_split, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

elif args.dataset == "SVHN":
    print('load SVHN dataset') 
    split_part = 'train' if load_train_split else 'test'
    testset = dset.SVHN(root="./data", download=True, split=split_part, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

elif args.dataset == "imagenet":
    print('load imagenet dataset') 
    split_part = 'train' if load_train_split else 'test'
    testset = ImageNet(split=split_part)
     
else:
    print('load fashion_mnist dataset')
    testset = dset.FashionMNIST(root="./data", train=load_train_split, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False)




def calculate_IDC(test_suite_indeces, test_suite_size, dataset):
    
    subset_data = Subset(testset, test_suite_indeces)
    test_loader = torch.utils.data.DataLoader(subset_data, batch_size=args.batchsize, shuffle=False)

    mu_test, sd_test = evaluate(model, test_loader, logger)

    #calculate the KL-divergence for the testdata set
    kl_div = 1 + np.log(np.square(sd_test)) - np.square(mu_test) - np.square(sd_test)
    kl_div *= -0.5
    kl_div = np.mean(kl_div, axis=0)

    #delete the dimensions with close to zero KL-divergence values
    noise = []
    for l in range(mu_test.shape[1]):
        if abs(kl_div[l]) <= 0.01:
            noise.append(l)

    #no of dimensions with information
    info_dims = mu_test.shape[1] - len(noise)


    mu_test = np.delete(mu_test, noise, 1)
    print(f"deleting columns {noise} with KL {kl_div[noise]} from the latent vectors")

    #create acts file for measuring total t-way coverage
    acts = create_acts(info_dims, args.no_bins)

    #generate feasible feature vectors
    feasible_vectors, valid_samples, _ = generate_array(mu_test, args.density, args.no_bins)
    print(f"#valid samples in the testset: {valid_samples}")
    
    timeout = 2
    i = 1
    while True:
        try:
            print('timeout:',timeout)
            coverage = measure_coverage(feasible_vectors, acts, ways=args.ways, timeout=timeout, suffix=f"{dataset}_{sampling_from}_{test_suite_size}")
            break
        except:
            print('the timeout was not enough, extended!')
            timeout = i * 2
            i += 1
        
        if i == 100:
            coverage = -1
            
    print(f"total {args.ways}-way coverage of the testset is {coverage}")
    
    return coverage



######################

def IDC_runner(test_suite_size, dataset_name, sampling_from):
    ratio = ''
    
    uniform_sampling = False ## if your subset samples are uniform please set flat to True
    
    batch_sizes = [test_suite_size]
    for bs in batch_sizes:
        
        bs_start_time = datetime.now()
        
        all_results = {'test_suite_size':[], 'test_suite':[], 'IDC':[], 'time_taken':[]}
        
        if not uniform_sampling:
            # sampling from test set
            source_path_test_suites = os.path.join('data_subsets', dataset_name, 'sampled_test_suites', f'batch_size_{bs}')
            # sampling from training set
            if sampling_from == 'training_set':
                source_path_test_suites = os.path.join('data_subsets', dataset_name, sampling_from, 'sampled_test_suites', f'batch_size_{bs}')
                
        else:
            source_path_test_suites = os.path.join('data_subsets', dataset_name, 'uniform_sampled_test_suites', f'batch_size_{bs}')
            # sampling from training set
            if sampling_from == 'training_set':
                source_path_test_suites = os.path.join('data_subsets', dataset_name, sampling_from, 'uniform_sampled_test_suites', f'batch_size_{bs}')
            
        
            
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
            idc_output = calculate_IDC(test_suite_indeces, test_suite_size, dataset_name)
            
            end = datetime.now()
            
            time_taken = end - start




            if uniform_sampling:
                # pass # because I ran some experiment without using 'uniform_sampling' so it is better to hold same path
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',dataset_name,ratio,'uniform_sampling', f'bs_{bs}')
                # save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',subject,ratio, 'uniform_sampling', f'bs_{bs}')
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',dataset_name,ratio, sampling_from,'uniform_sampling', f'bs_{bs}')
                                    
            else:  
                save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',dataset_name, ratio, 'non_uniform_sampling', f'bs_{bs}')
                
                if sampling_from == 'training_set':
                    save_path_of_all_result = os.path.join(base_path_of_outputs,'all_results',dataset_name, ratio, sampling_from,'non_uniform_sampling', f'bs_{bs}')

            
                
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
                    
                writer.writerow([ str(bs), str(test_suite_name), str(round(idc_output,4)), str(time_taken)])
                    
        bs_end_time = datetime.now()

        print(f'elapse time for bs {bs}:', bs_end_time-bs_start_time)


IDC_runner(test_suite_size, dataset_name=args.dataset, sampling_from=sampling_from)