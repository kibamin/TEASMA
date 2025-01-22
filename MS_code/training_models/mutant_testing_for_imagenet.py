import os
import json
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import gc #garbage collector
import keras.backend as K
from tensorflow.keras.backend import clear_session

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


from datetime import datetime
import subprocess
import re


root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..') # root of project

K.clear_session()
clear_session()
gc.collect()

def model_predict(model, data, output_save_path=False):
    print('model_predict function started!')
    correct_prediction = 0
    
    predicted_labels_list = []
    
    file_exist = False
    if output_save_path and os.path.exists(output_save_path+'.npy'):
        print('The output prediction of mutant model has been done before.')
    
    else:
        print(' predicte labels of mutant model using evaluation!')

        for images, labels in data:
            predictions = model.predict(images, verbose=0)
            predicted_labels = tf.argmax(predictions, axis=1)
            predicted_labels_list.append(predicted_labels)
            
            correct_prediction += tf.reduce_sum(tf.cast(tf.equal(predicted_labels, labels), tf.int32)).numpy()

        all_predicted_labels = np.concatenate(predicted_labels_list, axis=0)
        
        accuracy = correct_prediction / len(all_predicted_labels)
   
    print(f'end of the prediction model and acc is {accuracy}')    
    
    
    return all_predicted_labels



def get_gpu_memory():
    """Returns the current GPU memory usage by querying the nvidia-smi tool."""

    # Execute the command 'nvidia-smi' to get GPU details
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Use regex to parse the output for memory usage
    memory_usage = re.findall(r'\dMiB /  \dMiB', output)

    return memory_usage

def execute_mutants(subject, dataset_name='', dataset_part = 'test_set', postfix_mutant = ''):
    print(subject)
    ratio = 0.01
    
    num_classes = 1000
    
    if subject != 'inceptionV3_imagenet':
        raise "subject is note is not inceptionV3_imagenet"
   
   
    if dataset_name == 'imagenet':
        data_dir = '../datasets/imagenet/'
        write_dir = '../datasets/tf-imagenet-dirs'

        # Construct a tf.data.Dataset
        download_config = tfds.download.DownloadConfig(
                            extract_dir=os.path.join(write_dir, 'extracted'),
                            manual_dir=data_dir
                        )
        download_and_prepare_kwargs = {
            'download_dir': os.path.join(write_dir, 'downloaded'),
            'download_config': download_config}
        
        train_data, validation_data = tfds.load('imagenet2012', 
               data_dir=os.path.join(write_dir, 'data'),         
               split=['train', 'validation'], 
            #    split='validation', 
               shuffle_files=False, 
               download=False, 
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)


        def preprocess_image(image, label):
            print(f'process_image function started for subject:{subject}')
            # Resize the smaller side to 256 while maintaining aspect ratio
            initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
            ratio = 256.0 / tf.reduce_min(initial_shape)
            new_shape = tf.cast(initial_shape * ratio, tf.int32)
            image = tf.image.resize(image, new_shape)
            
            if 'resnet50' in subject:
                image = tf.image.resize_with_crop_or_pad(image, 224, 224)
                image = tf.keras.applications.resnet50.preprocess_input(image)
                
            else: ## inceptionV3
                image = tf.image.resize_with_crop_or_pad(image, 299, 299)
                image = tf.keras.applications.inception_v3.preprocess_input(image) 
                
            return image, label

        validation_data = validation_data.map(preprocess_image).batch(256)
    
    
    
    mutant_predictions_dir_path = os.path.join(root, 'mutants', 'mutant_prediction_outputs', subject, dataset_part)
    if not os.path.exists(mutant_predictions_dir_path):
        os.makedirs(mutant_predictions_dir_path)
    
    model_path = os.path.join(root,"models/inceptionV3_imagenet/original_model.h5")
    
    model = load_model(model_path)
    if model:
        predicted_y_by_original = model_predict(model, validation_data)
        np.save(os.path.join(mutant_predictions_dir_path,f'{prefix_mo}_original_model_prediction_outputs'), predicted_y_by_original) 
        
    K.clear_session()
    del model
    clear_session()
    gc.collect()
        
    mutant_models_path = os.path.join(root,'mutants',subject,str(ratio))
    print('mutants path:', mutant_models_path)
    all_mutants_file_name = os.listdir(mutant_models_path)
    print('number of mutants:', len(all_mutants_file_name))

    
    mutant_predictions_dic = {}
    counter = 0
    for mutant_file_name in all_mutants_file_name:
        if postfix_mutant not in mutant_file_name:
            continue
        
        mutant_model_path = os.path.join(mutant_models_path, mutant_file_name)
        print(mutant_model_path)
        
        model = load_model(mutant_model_path)
        
        predicted_y = model_predict(model, validation_data)
        print(predicted_y.shape)
        
        print('before clearing')
        gpu_memory = get_gpu_memory()
        print("Current GPU Memory Usage:")
        print(gpu_memory)
        
        K.clear_session()
        del model
        clear_session()
        gc.collect()
        
        print('after clearing')
        gpu_memory = get_gpu_memory()
        print("Current GPU Memory Usage:")
        print(gpu_memory)
        

        mutant_predictions_dic[mutant_file_name] = predicted_y
        
        np.save(os.path.join(mutant_predictions_dir_path,f'{postfix_mutant}_prediction_outputs'), mutant_predictions_dic) 

        print(f'{counter} :: save until {mutant_file_name}')
        counter += 1
            
    np.save(os.path.join(mutant_predictions_dir_path,f'{postfix_mutant}_prediction_outputs'), mutant_predictions_dic) 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_mo", "-prefix_mo",
                        type=str,
                        help="mutation operator")
    

    args = parser.parse_args()
    prefix_mo = args.prefix_mo
    
    subject = 'inceptionV3_imagenet'
    dataset = 'imagenet'
    dataset_part = 'test_set'
    
    
    start = datetime.now()
    execute_mutants(subject, dataset, dataset_part, postfix_mutant=prefix_mo)
    end = datetime.now()
    print(f'elapsed time for mutant executing on validationset of imagenet for mutation operator {prefix_mo}:', end - start)
    
    
