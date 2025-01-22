import os
import tensorflow as tf
from cnn_operator import *
from keras.models import load_model
import argparse
from utils import summary_model, color_preprocessing
from termcolor import colored
import gc #garbage collector
import keras.backend as K
# from progressbar import *
import csv
# from keras.utils import np_utils
import properties_handler
import time
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import json
import pandas as pd

import logging

from datetime import datetime
import pickle

import tensorflow_datasets as tfds
from tqdm import tqdm

from tensorflow.keras.backend import clear_session





job_id = os.getenv('SLURM_JOB_ID')

# Check if the environment variable exists
if job_id is not None:
    print(f"Running under Slurm with Job ID: {job_id}")
else:
    job_id = ''
    print("Not running under Slurm or Job ID not found.")



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


logger = MyLogger(f'{job_id}_imagenet_mutant_generator.py') 


root = os.path.dirname(os.path.abspath(__file__)) 
import sys

sys.path.insert(1, os.path.join(root, '..','..'))
from preprocess_data import *
print(root)


def cnn_mutants_generation(ori_model, operator, ratio, standard_deviation=0.5):
    """

    :param ori_model:
    :param operator:
    :param ratio:
    :param standard_deviation:
    :return:
    """
    if operator < 5:
        cnn_operator(ori_model, operator, ratio, standard_deviation) 
    else:
        new_model = cnn_operator(ori_model, operator, ratio, standard_deviation)
        return new_model
    return ori_model

def model_predict(model, data, output_save_path=False):
    logger.info('model_predict function started!')
    correct_prediction = 0
    
    predicted_labels_list = []
    
    file_exist = False
    if output_save_path and os.path.exists(output_save_path+'.npy'):
        logger.info('read predicted labels of original model from file! and evaluate original model')
        
        # _, accuracy = model.evaluate(data, verbose=1)
        original_model_acc_inceptionV3 = 0.8935486078262329
        accuracy = original_model_acc_inceptionV3
        
        
        file_exist = True
        all_predicted_labels = np.load(output_save_path+'.npy')
        # all_actual_labels = []
        
        # for _, labels in tqdm(data):
        #     all_actual_labels.extend(labels)
        
        # with tf.device('/CPU:0'):
        #     all_actual_labels = tf.convert_to_tensor(all_actual_labels, dtype=tf.int32)
        #     correct_prediction += tf.reduce_sum(tf.cast(tf.equal(all_predicted_labels, all_actual_labels), tf.int32)).numpy()    
    
    else:
        logger.info(' predicte labels of mutant model using evaluation!')

        for images, labels in data:
            predictions = model.predict(images, verbose=0)
            predicted_labels = tf.argmax(predictions, axis=1)
            predicted_labels_list.append(predicted_labels)
            
            correct_prediction += tf.reduce_sum(tf.cast(tf.equal(predicted_labels, labels), tf.int32)).numpy()

        all_predicted_labels = np.concatenate(predicted_labels_list, axis=0)
        
        accuracy = correct_prediction / len(all_predicted_labels)
   
    logger.info(f'end of the prediction model and acc is {accuracy}')    
    
    
    return accuracy, all_predicted_labels, file_exist


def generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        help="original model path")
    
    parser.add_argument("--subject_name", "-subject_name",
                        type=str,
                        help="model name for example (lenet5, cifar10, ...)")
                        
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        help="mnist or cifar-10 or other dataset")
    parser.add_argument("--operator", "-operator",
                        type=int,
                        help="mutator (-1 means all operators)")
    parser.add_argument("--ratio", "-ratio",
                        type=float,
                        help="mutation ratio")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="mutants save path")
    parser.add_argument("--threshold", "-threshold",
                        type=float,
                        default=0.80,
                        help="ori acc * threshold must > mutants acc")
    parser.add_argument("--num", "-num",
                        type=int,
                        default=1,
                        help="mutants number")
    parser.add_argument("--standard_deviation", "-standard_deviation",
                        type=float,
                        default=0.5,
                        help="standard deviation for gaussian fuzzing")
    
    
    
    args = parser.parse_args()
    model_path = args.model_path
    subject = args.subject_name
    data_type = args.data_type
    operator = args.operator
    ratio = args.ratio
    save_path = args.save_path
    threshold = args.threshold
    num = args.num
    standard_deviation = args.standard_deviation


    if data_type == 'imagenet':
        data_dir = '/home/kiba/scratch/datasets/imagenet/'
        write_dir = '/home/kiba/scratch/tf-imagenet-dirs'

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
            logger.info(f'process_image function started for subject:{subject}')
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
                image = tf.keras.applications.inception_v3.preprocess_input(image) ## 5 min with batch size 512
                
            return image, label

        # # Function to extract labels
        # def extract_labels(dataset):
        #     all_labels = []
        #     for _, label in tqdm(tfds.as_numpy(dataset)):  # Convert dataset to NumPy format
        #         all_labels.append(label)
        #     return all_labels

        # # Extract labels from training data
        # train_labels = extract_labels(train_data)
        
        # print(len(train_labels))
        # print(train_labels[:10])
        # print('=====================================')
        train_data = train_data.map(preprocess_image).batch(256)

        
    data = {}
    data['root'] = os.path.dirname(os.path.abspath(__file__)) 
    data['subject'] = subject
    properties_handler.write_properties(data)

    TEASMA_root_path = os.path.join(root,'..','..',)
    
    
    if operator == -1: # all operators
        operators = list(range(8))
    else:
        operators = [operator]
    
    mutant_predictions_dir_path = os.path.join(TEASMA_root_path, 'mutants', 'mutant_prediction_outputs', subject, 'train_set')
    try:
        if not os.path.exists(mutant_predictions_dir_path):
            os.makedirs(mutant_predictions_dir_path)   
    except:
        pass
    
    model = load_model(model_path)
    logger.info(f'model path: {model_path}')
    logger.info(os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs.npy'))
    
    ori_acc, original_output_probability, file_exist = model_predict(model, train_data, output_save_path = os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs'))
    if not file_exist:
        np.save(os.path.join(mutant_predictions_dir_path,'original_model_prediction_outputs'), original_output_probability) 
    logger.info(ori_acc)
    
    threshold = ori_acc * threshold
        
    mutant_predictions_dic = {}
    equivalent_mutants = []

    time_dictionary = {'mutant':[], 'generation_time':[], 'execution_time':[], 'generation_time_process (sec)':[], 'mutant_status':[]}
    
    total_start_time = datetime.now()
    new_model = None
    for operator in operators:
        logger.info(f'operator: {operator}')
        if operator in [5,6,7]: ## these operators are not applicable for resnet structure
            logger.debug(f'The operator number {operator} is not applicable on a model include Add() or concatenation() layer')
            continue
        
        
        K.clear_session()
        del model
        del new_model
        gc.collect()
        clear_session()  # Clear models from previous iteration
        
        
        model = load_model(model_path)
        
       
        weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)

        print(colored("operator: %s" % cnn_operator_name(operator), 'blue'))
        print(colored("ori acc: %f" % ori_acc, 'blue'))
        print(colored("threshold acc: %f" % threshold, 'blue'))

    
        if operator == 0 or operator == 1:
            logger.info(f"total weights: {weight_count}")
            # print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)
        elif 2 <= operator <= 4:
            logger.info("total neuron: {neuron_count}")
            # print("process neuron num: ", int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1)

        # mutants generation
        i = 1
        # start_time = time.clock()
        start_time = time.process_time()
        count = 1
        while i <= num:
            if count > 100: # if number of attempts > 100 it is obviouse there is now mutant satisfy the thereshold accuracy
                break
            if i != 1: # In the first time it doesn't load original model and in the rest of the while iteration it loads original model
                model = load_model(model_path)
            # generate mutant
            stime = time.process_time()
            g_start_time = datetime.now()
            logger.warning(f'start mutant generation for opt: {operator}')
            
            #----------------------#
            #    Generate mutant   #
            #----------------------#
            new_model = cnn_mutants_generation(model, operator, ratio, standard_deviation) # TODO: this function generate mutant based on operator's number
            
            logger.warning(f'end mutant generation for opt: {operator}')

            g_end_time = datetime.now()
            
            etime = time.process_time()
            
            
           
            
            
            
            logger.info(f'elapsed time for mutation operator {cnn_operator_name(operator)} number {count}: {(etime - stime)/60}')
            # test mutant
            # operator = 5 (Layer Romve(LR) or LD)  this operator maybe returns False because it could not find any layers which the input shape and output shape be same
            if type(new_model) == bool:
                logger.warning('bool type so break from the loop')
                time_dictionary['generation_time'].append(g_end_time-g_start_time)
                time_dictionary['generation_time_process (sec)'].append(etime - stime)
                time_dictionary['mutant'].append(cnn_operator_name(operator)+str(count))
                time_dictionary['mutant_status'].append(False)
                time_dictionary['execution_time'].append(0)
                break

            if type(new_model) == dict:
                logger.info('dictonary of mutants')
                for name_of_mutant, mutant_model in new_model.items():
                    time_dictionary['generation_time'].append(g_end_time-g_start_time)
                    time_dictionary['generation_time_process (sec)'].append(etime - stime)
                    time_dictionary['mutant'].append('total_'+cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5")
                    
                    ex_start_time = datetime.now()
                    new_acc, mutant_output_probability,_ = model_predict(mutant_model, train_data)
                    ex_end_time = datetime.now()
                    time_dictionary['execution_time'].append(ex_end_time - ex_start_time)
                    
                    logger.info(f'execution time for evaluation of mutant {ex_end_time - ex_start_time}')
                    
                    
                    
                    logger.info(name_of_mutant)
                    if new_acc < threshold: # a generated mutant is acceptable as mutant if mutant_acc > ori_acc * threshold (I think it want to kill easily - trivial) [those mutants with accuracy higher than the threshold 0.9(-threshold)is stored in the foldermutants(-savepath).]
                        logger.warning('mutant did not satisfied the threshold. filtered')
                        count += 1 # number of times that couldn't pass threshold 0.9
                        time_dictionary['mutant_status'].append(False)
                        
                        print(colored("ori acc: %f" % ori_acc, 'red'))
                        print(colored("threshold acc: %f" % threshold, 'red'))
                        print(colored("mutant acc: %f" % new_acc, 'red'))
                        K.clear_session()
                        del model
                        # del new_model
                        gc.collect()
                        model = load_model(model_path) 
                        continue
                    # save each mutant
                    logger.info('check equivalents labels')
                    are_equivalent = np.array_equal(mutant_output_probability, original_output_probability)
                    
                    if are_equivalent:
                        equivalent_mutants.append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5")
                    
                    logger.info('append to dictionary')
                    mutant_predictions_dic[cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5"] = mutant_output_probability
                    logger.info('successfully appended')
                    
                    
                    if operator == -1:
                        np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{job_id}'), mutant_predictions_dic) 
                    else:
                        np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{str(operator)}_{job_id}'), mutant_predictions_dic) 
                    
                    logger.info('saved mutant_prediction_dic')
                    save_dir_path = os.path.join(save_path, str(ratio), f"{str(operator)}_{cnn_operator_name(operator)}_{str(job_id)}")
                    final_path = os.path.join(save_dir_path , cnn_operator_name(operator) + "_" + str(ratio) + "_" + name_of_mutant + ".h5" )
                    mutant_model.save(final_path)
                    i = num+1
                    time_dictionary['mutant_status'].append(True)
                    

            else:
                logger.info('process of mutants')
                
                time_dictionary['generation_time'].append(g_end_time-g_start_time)
                time_dictionary['generation_time_process (sec)'].append(etime - stime)
                ex_start_time = datetime.now()
                new_acc, mutant_output_probability,_ = model_predict(new_model, train_data)
                ex_end_time = datetime.now()
                time_dictionary['execution_time'].append(ex_end_time - ex_start_time)
                time_dictionary['mutant'].append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5")
                
                
                if new_acc < threshold: # a generated mutant is acceptable as mutant if mutant_acc > ori_acc * threshold (I think it want to kill easily) [those mutants with accuracy higher than the threshold0.9(-threshold)isstored in the foldermutants(-savepath).]
                    logger.error('mutant faild since of accuracy threshold')
                    count += 1 # number of times that couldn't pass threshold 0.9
                    time_dictionary['mutant_status'].append(False)
                    print(colored("ori acc: %f" % ori_acc, 'red'))
                    print(colored("threshold acc: %f" % threshold, 'red'))
                    print(colored("mutant acc: %f" % new_acc, 'red'))
                    K.clear_session()
                    del model
                    del new_model
                    gc.collect()
                    model = load_model(model_path) 
                    continue
                
                logger.error('mutant accepted, threshold satisfied !!!')
                
                are_equivalent = np.array_equal(mutant_output_probability, original_output_probability)
                if are_equivalent:
                    equivalent_mutants.append(cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5")
                
                logger.info('append to dictionary number B')
                mutant_predictions_dic[cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5"] = mutant_output_probability
                logger.info('successfully appended number B')
                
                if operator == -1:
                    np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{job_id}'), mutant_predictions_dic) 
                else:
                    np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{str(operator)}_{job_id}'), mutant_predictions_dic) 
                
                logger.info('mutant_predictions_dic saved successfully!')
                
                # save each mutant
                
                save_dir_path = os.path.join(save_path, str(ratio), f"{str(operator)}_{cnn_operator_name(operator)}_{str(job_id)}")
                final_path = os.path.join(save_dir_path , cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5" )
                
                new_model.save(final_path)
                time_dictionary['mutant_status'].append(True)
                
                logger.error('mutant saved!')

            # p_bar.update(int((i / num) * 100))
            i += 1
            # count += 1
            K.clear_session()
            del model
            del new_model
            gc.collect()
        
        # p_bar.finish()
        elapsed = (time.process_time() - start_time)
        # elapsed = (time.clock() - start_time)
        logger.info(f"running time: {elapsed}")
        logger.info(f'number of mutants that filter due to tiriviality {count}')
        
        try:
            if not os.path.exists('cost_of_mutant_generation'):
                os.makedirs('cost_of_mutant_generation')
        except:
            print('directory exists!')
            pass
        
        
        with open(os.path.join('cost_of_mutant_generation', f'cost_of_TEASMA_{subject}.txt'), 'a') as file:
            
            file.write(f"Time for generating mutant of operator number {operator} named {cnn_operator_name(operator)} for {subject} is = {elapsed}.\n")
            file.write(f"'number of mutants that filter due to tiriviality (acc < 90% of orginal accuracy)' = {len(equivalent_mutants)}.\n")
            file.write(f"'number of tries' = {i}.\n")
        

        # extra_output_path = 'extra_outputs'
        # with open(os.path.join(extra_output_path,f'generate_mutants.csv'), 'a') as f1:
        #     writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        #     # writer.writerow([str(data_name), str(test_suite_name), str(basic_ms), str(dc_ms), str(FDR), str(time_taken)])
        #     writer.writerow([cnn_operator_name(operator), str(ratio), str(i-1)+'/'+str(count-1), str(elapsed)])

    total_end_time = datetime.now()
    
    equivalent_mutants_dir_path = os.path.join(TEASMA_root_path, 'mutants', 'equivalent_mutants', subject)
    try:
        if not os.path.exists(equivalent_mutants_dir_path):
            os.makedirs(equivalent_mutants_dir_path)
    except:
        print('file exist')
        
    equivalent_mutants_file = os.path.join(equivalent_mutants_dir_path,'equivalents.json')
    with open(equivalent_mutants_file,'w') as f:
        json.dump(equivalent_mutants, f) 


    # if operator == -1:
    #     np.save(os.path.join(mutant_predictions_dir_path,'prediction_outputs'), mutant_predictions_dic) 
    # else:
    #     np.save(os.path.join(mutant_predictions_dir_path,f'prediction_outputs_{str(operator)}'), mutant_predictions_dic) 
    
    
    df = pd.DataFrame(time_dictionary)
    cost_file = os.path.join(equivalent_mutants_dir_path,'time_cost.csv')
    df.to_csv(cost_file)
    # with open(cost_file,'w') as f:
    #     json.dump(time_dictionary, f) 
    

    logger.info('elapsed time is : {total_end_time - total_start_time}')
    logger.info(time_dictionary)
    

if __name__ == '__main__':
    generator()
    logger.info('End of mutation generator')

# python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2

# python generator.py --model_path models/cifar10.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2
