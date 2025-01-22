from __future__ import print_function
import os
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import pickle
import argparse

root = os.path.dirname(os.path.abspath(__file__))

class TestSuiteHandler():
    def __init__(self, dataset_name, sampling_from) -> None:
        self.dataset_name = dataset_name
        self.sampling_from = sampling_from # 'test_set' or 'training_set'

    # non-uniform test suites
    def gnerate_test_suites_randomly(self, batch_size = 0, num_samples = 300):
        print('================================================')
        print('dataset:', self.dataset_name, 'batch_size:', batch_size, 'num_of_samples:', num_samples )
        print('================================================')
        
        
        if self.dataset_name == 'mnist':
            print('random sampling from mnist')
            ((x_train, y_train), (x_test, y_test)) = mnist.load_data()

        elif self.dataset_name == 'cifar10':
            print('random sampling from cifar10')
            ((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
        
        elif self.dataset_name == 'fashion_mnist':
            print('random sampling from fashion_mnist')
            ((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()
        
        elif self.dataset_name == 'SVHN':
            print('random sampling from SVHN')
            train_raw = loadmat(os.path.join(root,'datasets','SVHN', 'train_32x32.mat'))
            test_raw = loadmat(os.path.join(root,'datasets','SVHN', 'test_32x32.mat'))
            
            x_train = np.array(train_raw['X'])
            x_test = np.array(test_raw['X'])

            x_train = np.moveaxis(x_train, -1, 0)
            x_test = np.moveaxis(x_test, -1, 0)
            
            x_test= x_test.reshape (-1,32,32,3)
            x_train= x_train.reshape (-1,32,32,3)


        # --------------------------------------
        elif self.dataset_name in ['amazon', 'office31_mix']:
            print('random sampling from office31')
            train_raw = loadmat(os.path.join(root,'datasets',self.dataset_name, 'train.mat'))
            test_raw = loadmat(os.path.join(root,'datasets',self.dataset_name, 'test.mat')) ### differential testing output inputs
            
            x_train = np.array(train_raw['X'])
            x_test = np.array(test_raw['X'])
            print('size of training set:',len(x_train))
            print('size of test set:',len(x_test))
            y_train = train_raw['y']
            y_test = test_raw['y']
            
            y_train = np.argmax(y_train,axis=1)
            y_test = np.argmax(y_test,axis=1)
            
            
        elif self.dataset_name in ['caltech256']:
            with open(os.path.join(root, 'datasets', self.dataset_name, 'train_test_indexes.pkl'),'rb') as f:
                dataset_indeces_dict = pickle.load(f)
            x_train = dataset_indeces_dict['train_indexes']
            x_test = dataset_indeces_dict['test_indexes']
            
        elif self.dataset_name in ['caltech256_8020']:
            with open(os.path.join(root, 'datasets', self.dataset_name, 'train_test_x_y.pkl'),'rb') as f:
                dataset_path_dict = pickle.load(f)
            
            x_train = dataset_path_dict['train_x']
            x_test = dataset_path_dict['test_x']
            
            print(x_train.shape)
        
        elif self.dataset_name in ['office31']:
            with open(os.path.join(root, 'datasets', self.dataset_name, 'train_test_x_y.pkl'),'rb') as f:
                dataset_path_dict = pickle.load(f)
            
            x_train = dataset_path_dict['train_x']
            x_test = dataset_path_dict['test_x']
            
            print(x_train.shape)
                            
        elif self.dataset_name in [ 'cifar10_brightness_500', 'cifar10_contrast_500', 'cifar10_rotation_500', 'cifar10_scale_500', 'cifar10_shear_500', 'cifar10_translation_500']:
            ((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
            
            ### select random from target dataset
            selected_size =  self.dataset_name.split('_')[2]
            x_train =  np.arange(len(x_train)+int(selected_size)) ## 50500
            x_test = np.arange(len(x_test)) ## 10000
        
        elif self.dataset_name in [ 'mnist_brightness', 'mnist_contrast', 'mnist_rotation', 'mnist_scale', 'mnist_shear', 'mnist_translation', 'mnist_combined']: ## this is just for test we don't need train files because the original model is from mnist dataset
            x_test = np.arange(10000) ## 10000 size of test set of mnist
        
        
        elif self.dataset_name in 'cifar100':
            ((x_train, y_train), (x_test, y_test)) = cifar100.load_data()
        
        elif self.dataset_name in 'imagenet':
            x_train =  np.arange(1281167)
            x_test = np.arange(50000)
        
        
        
        all_test_suites = {}
        
        for i in range(num_samples):
            # if i % 20 == 0:
            #     print(i , batch_size)
            if self.sampling_from == 'test_set':
                # print("sampling from test set")
                sample_indices = np.random.choice(x_test.shape[0], batch_size, replace=False)
                # testX_sample = x_test[sample_indices]
                # testY_sample = y_test[sample_indices]
                
                testX_sample = np.array([])
                testY_sample = np.array([])
                test_sample = {'x_test':testX_sample, 'y_test':testY_sample, 'indices': sample_indices}

                # Save the data as numpy arrays in .npy format
                # dir_path = os.path.join(root,'data', self.dataset_name, 'sampled_test_suites', f'batch_size_{batch_size}')
                dir_path = os.path.join(root,'data', self.dataset_name, 'uniform_sampled_test_suites', 'sampled_test_suites', f'batch_size_{batch_size}')
            
            elif self.sampling_from == 'training_set':
                # print("sampling from training set")
                sample_indices = np.random.choice(x_train.shape[0], batch_size, replace=False) # sampling from training set
                # testX_sample = x_train[sample_indices] # sampling from training set
                # testY_sample = y_train[sample_indices] # sampling from training set
                
                testX_sample = np.array([])
                testY_sample = np.array([])
                test_sample = {'x_test':testX_sample, 'y_test':testY_sample, 'indices': sample_indices} # we don't change the key names because we use same code from test set in runner.py

                # Save the data as numpy arrays in .npy format
                dir_path = os.path.join(root,'data', self.dataset_name, 'training_set','sampled_test_suites',f'batch_size_{batch_size}')
            
            else:
                raise 'define sampling_from'
            
            
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

            path = os.path.join(dir_path, f'test_suite_{i}')
            
            all_test_suites[f'test_suite_{i}.npy'] = test_sample
            
            # np.save(path,test_sample)
        
        np.save(os.path.join(dir_path, 'test_suites_all.npy'), all_test_suites)
            

    # uniform test suites
    def gnerate_balanced_test_suites_randomly(self, batch_size = 0, num_samples = 300):

        print('dataset:', self.dataset_name, 'batch_size:', batch_size)
        
        if self.dataset_name == 'mnist':
            ((_, y_train), (_, y_test)) = mnist.load_data()
            x_train =  np.arange(60000)
            x_test = np.arange(10000)

        elif self.dataset_name == 'cifar10':
            ((_, y_train), (_, y_test)) = cifar10.load_data()
            x_train =  np.arange(50000)
            x_test = np.arange(10000)
        
        elif self.dataset_name == 'fashion_mnist':
            ((_, y_train), (_, y_test)) = fashion_mnist.load_data()
            x_train =  np.arange(60000)
            x_test = np.arange(10000)
        
        elif self.dataset_name == 'SVHN':
            x_train =  np.arange(73257)
            x_test = np.arange(26032)
            
            train_raw = loadmat(os.path.join(root,'datasets','SVHN', 'train_32x32.mat'))
            test_raw = loadmat(os.path.join(root,'datasets','SVHN', 'test_32x32.mat'))
            
            y_train = train_raw['y']
            y_test = test_raw['y']
            
            lb = LabelBinarizer()
            
            y_train = lb.fit_transform(y_train)
            y_test = lb.fit_transform(y_test)
            
            y_train = np.argmax(y_train,axis=1)
            y_test = np.argmax(y_test,axis=1)

        elif self.dataset_name == 'cifar100':
            ((_, y_train), (_, y_test)) = cifar100.load_data()
            x_train =  np.arange(50000)
            x_test = np.arange(10000)
        
        elif self.dataset_name in 'imagenet':
            x_train =  np.arange(1281167)
            x_test = np.arange(50000)
            y_train = np.load(os.path.join(root,'models','inceptionV3_imagenet','actual_labels', 'all_actual_labels_train.npy'))
            y_test = np.load(os.path.join(root,'models','inceptionV3_imagenet','actual_labels', 'all_actual_labels_validation.npy'))
            

            
          
            
        # Get the number of classes
        num_classes = len(np.unique(y_train))

        all_test_suites = {}
        print(num_samples)
        for i in range(num_samples):
            if self.sampling_from == 'test_set':
                print('=========')
                # Initialize the test dataset and count for each class
                sample_indices = []
                class_count = np.zeros(num_classes)

                # Loop through the MNIST test dataset
                k = 0
                while k < batch_size :

                    # Initialize the test dataset and count for each class
                    class_count = np.zeros(num_classes, dtype=int)
                    sample_indices = set()

                    while len(sample_indices) < batch_size:
                        remaining_slots = batch_size - len(sample_indices)
                        # Sample multiple indices to speed up the process
                        candidate_indices = np.random.choice(x_test.shape[0], remaining_slots * 2, replace=True)

                        for idx in candidate_indices:
                            if idx in sample_indices:
                                continue
                            label = y_test[idx]
                            # Check if the count for this class has reached its limit
                            if class_count[label] < int(batch_size / num_classes):
                                sample_indices.add(idx)
                                class_count[label] += 1
                                if len(sample_indices) >= batch_size:
                                    break
                
                
                
                testX_sample = np.array([])
                testY_sample = np.array([])
                test_sample = {'x_test':testX_sample, 'y_test':testY_sample, 'indices': sample_indices}

                # Save the data as numpy arrays in .npy format
                dir_path = os.path.join(root,'data', self.dataset_name, 'uniform_sampled_test_suites', 'sampled_test_suites' ,f'batch_size_{batch_size}')
            
            elif self.sampling_from == 'training_set':
                # Initialize the test dataset and count for each class
                class_count = np.zeros(num_classes, dtype=int)
                sample_indices = set()

                while len(sample_indices) < batch_size:
                    remaining_slots = batch_size - len(sample_indices)
                    # Sample multiple indices to speed up the process
                    candidate_indices = np.random.choice(x_train.shape[0], remaining_slots * 2, replace=True)

                    for idx in candidate_indices:
                        if idx in sample_indices:
                            continue
                        label = y_train[idx]
                        # Check if the count for this class has reached its limit
                        if class_count[label] < int(batch_size / num_classes):
                            sample_indices.add(idx)
                            class_count[label] += 1
                            if len(sample_indices) >= batch_size:
                                break
                


                testX_sample = np.array([])
                testY_sample = np.array([])
                test_sample = {'x_test':testX_sample, 'y_test':testY_sample, 'indices': sample_indices} # we don't change the key names because we use same code from test set in runner.py

                # Save the data as numpy arrays in .npy format
                dir_path = os.path.join(root,'data', self.dataset_name, 'uniform_sampled_test_suites', 'training_set','sampled_test_suites',f'batch_size_{batch_size}')
            
            
            
            else:
                raise 'define sampling_from'
            
            '''
            
            elif self.sampling_from == 'training_set':
                
                sample_indices = []

                class_count = np.zeros(num_classes)

                # Loop through the MNIST test dataset
                k = 0
                while k < batch_size :

                    sample_indix = np.random.choice(x_train.shape[0], 1, replace=True)
                    print(len(sample_indices))
                    # print(sample_indix)

                    if sample_indix in sample_indices:
                        continue
                    label = y_train[sample_indix[0]]
                    # Check if the count for this class has reached 10
                    if class_count[label] < int(batch_size/num_classes):
                        # Increment the count for this class
                        class_count[label] += 1

                        sample_indices.append(sample_indix[0])
                        k += 1
                    
                    if np.all(class_count >= int(batch_size/num_classes)):
                        break
                
                
                
                testX_sample = np.array([])
                testY_sample = np.array([])
                test_sample = {'x_test':testX_sample, 'y_test':testY_sample, 'indices': sample_indices} # we don't change the key names because we use same code from test set in runner.py

                # Save the data as numpy arrays in .npy format
                dir_path = os.path.join(root,'data', self.dataset_name, 'uniform_sampled_test_suites', 'training_set','sampled_test_suites',f'batch_size_{batch_size}')
            '''
            
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

            path = os.path.join(dir_path, f'test_suite_{i}')
            
            all_test_suites[f'test_suite_{i}.npy'] = test_sample
            
            # np.save(path,test_sample)
        
        np.save(os.path.join(dir_path, 'test_suites_all.npy'), all_test_suites)
        
        print(f'generate uniform test suite sample for size {batch_size} finished!')
        

    
def load_test_suite(path=''):
    # if not path:
    #     path = os.path.join(root,'datasets', self.dataset_name, 'sampled_test_suites',f'batch_size_{100}', f'test_suite_{0}.npy')
    f = np.load(path, allow_pickle=True).item()
    x_test, y_test, indeces = f['x_test'], f['y_test'], f['indices']      
    return (x_test, y_test, indeces)


dataset_subsets_sizes = {'cifar100': {'train_subset_sizes': [30000, 22000, 6000, 1500, 3000, 300, 4000, 12000, 40000, 8000, 20000, 10000, 35000, 16000, 18000, 24000, 700, 500, 1000, 100, 2000, 14000, 5000], 'test_subset_sizes': [9500, 1500, 3000, 300, 4000, 2500, 8000, 9900, 6500, 700, 500, 1000, 100, 2000, 9000, 5000]}, 
                        'imagenet': {'train_subset_sizes': [30000, 15000, 1500, 3000, 40000, 70000, 8000, 20000, 25000, 10000, 35000, 45000, 120000, 500, 1000, 90000, 100, 50000, 5000], 'test_subset_sizes': [30000, 15000, 1500, 3000, 12000, 40000, 8000, 20000, 25000, 10000, 35000, 45000, 500, 1000, 100, 5000]}, 
                        'SVHN': {'train_subset_sizes': [30000, 1500, 3000, 300, 12000, 40000, 8000, 20000, 35000, 16000, 24000, 500, 1000, 100, 5000], 'test_subset_sizes': [1500, 3000, 300, 12000, 8000, 20000, 16000, 500, 1000, 100, 5000]}, 
                        'fashion_mnist': {'train_subset_sizes': [28000, 1500, 3000, 300, 12000, 8000, 20000, 35000, 16000, 24000, 32000, 60000, 500, 1000, 100, 5000], 'test_subset_sizes': [1500, 3000, 300, 8000, 500, 1000, 100, 5000]}, 
                        'cifar10': {'train_subset_sizes': [28000, 1500, 3000, 300, 50, 12000, 8000, 20000, 35000, 16000, 24000, 32000, 500, 1000, 100, 50000, 5000], 'test_subset_sizes': [1500, 3000, 300, 50, 8000, 500, 1000, 100, 9000, 5000]}, 
                        'mnist': {'train_subset_sizes': [28000, 1500, 3000, 300, 50, 12000, 8000, 20000, 35000, 16000, 200, 24000, 32000, 700, 60000, 500, 1000, 400, 100, 5000], 'test_subset_sizes': [9500, 1500, 3000, 300, 8000, 7000, 9600, 500, 1000, 100, 9000, 5000]}
                        }


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        default= '')
    args = parser.parse_args()
    dataset = args.dataset
    
    datasets = [dataset]
    datasets = list(dataset_subsets_sizes.keys())
    
    for d in datasets:
        print(f'===> {d}')
        # for sampling in ['training_set', 'test_set']:
        for sampling in ['test_set']:
            tsh = TestSuiteHandler(d, sampling_from=sampling)

            
            if sampling == 'training_set':
                sizes = dataset_subsets_sizes[d]['train_subset_sizes']
            
            if sampling == 'test_set':
                sizes = dataset_subsets_sizes[d]['test_subset_sizes']


            for tss in sizes:
                ## Randomly sampling
                
                tsh.gnerate_test_suites_randomly(batch_size = tss, num_samples=300) 
                
                
                ## uniform sampling
                print('uniform sampling')
                # tsh.gnerate_balanced_test_suites_randomly(batch_size = tss, num_samples=300) 
        

