import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image
import json


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split







def get_excluded_classes(source, target):
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__))) # root of project
    file_path = os.path.join(root_path,'datasets','exclude_classes.json')
    with open(file_path,'r') as json_file:
        excluded_classes_dic = json.load(json_file)

    target_ex_classes = excluded_classes_dic[source][target]
    list_of_not_found_classes_in_source = []
    for object_name, labels in target_ex_classes.items():
        if -1 in labels:
            list_of_not_found_classes_in_source.append(object_name)
    
    return list_of_not_found_classes_in_source

def get_path_of_images_and_labels(path, excluded_classes = []):
    
    X_path = []
    y = []

    # fig, axes = plt.subplots(1, num_of_classes, figsize=(16, 5))
    i = 0
    for r, d, f in os.walk(path):
        for direct in d:
            if direct in excluded_classes:
                continue
            if not ".ipynb_checkpoints" in direct:
                for r, d, f in os.walk(os.path.join(path , direct)):
                    for file in f:
                        path_to_image = os.path.join(r, file)
                        if not ".ipynb_checkpoints" in path_to_image:
                            X_path.append(path_to_image)
                            y.append(direct)
                i+=1
    
    one = OneHotEncoder(sparse=False)
    
    y_lab = one.fit_transform(np.array(y).reshape(-1, 1))
    num_of_classes = len(y_lab[0])
    
    y_lab = list(np.argmax(y_lab,axis=1))

    
    
    mapping_dict = {}
    for i, category in enumerate(one.categories_[0]):
        mapping_dict[category] = i
        
    # original_categorical_data = one.inverse_transform(y_lab)
    
    return  X_path, y_lab, num_of_classes, mapping_dict            

def load_data(path, test_split_size = 0.5, excluded_classes=[], loaded_train_indeces = [], loaded_test_indeces = [], source='', target=''):
    if source and target:
        excluded_classes = get_excluded_classes(source, target)
    
    X_path, y_lab, num_of_classes, mapping_dict = get_path_of_images_and_labels(path, excluded_classes)

    np.random.seed(0)
    
    if len(loaded_train_indeces) and len(loaded_test_indeces):
        train_indexes = loaded_train_indeces
        all_train_indexes = train_indexes
        test_indexes = loaded_test_indeces
        
    else: 
        train_indexes, test_indexes = train_test_split(np.arange(len(X_path)), test_size=test_split_size, shuffle=True)
        all_train_indexes = train_indexes
        
        
   
    train_indexes, validation_indexes = train_test_split(train_indexes, test_size=0.1, shuffle=True)
    print('====================================================================')
    print("all_train_size: %i, Train size: %i, Test size: %i"%(len(all_train_indexes), len(train_indexes), len(test_indexes)))
    print('====================================================================')
    
    
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # Use decode_png for PNG images
        # image = Image.open(path)
        
        return image, label

    # Preprocessing function
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    
    # Creating TensorFlow datasets
    ds_train = tf.data.Dataset.from_tensor_slices((np.array(X_path)[train_indexes], np.array(y_lab)[train_indexes]))
    ds_all_train = tf.data.Dataset.from_tensor_slices((np.array(X_path)[all_train_indexes], np.array(y_lab)[all_train_indexes]))
    ds_validation = tf.data.Dataset.from_tensor_slices((np.array(X_path)[validation_indexes], np.array(y_lab)[validation_indexes]))
    ds_test = tf.data.Dataset.from_tensor_slices((np.array(X_path)[test_indexes], np.array(y_lab)[test_indexes]))

    # Load and preprocess the datasets
    train_ds = ds_train.map(load_image).map(preprocess).batch(32)
    all_train_ds = ds_all_train.map(load_image).map(preprocess).batch(32)
    val_ds = ds_validation.map(load_image).map(preprocess).batch(32)
    test_ds = ds_test.map(load_image).map(preprocess).batch(32)
    
    return all_train_ds, train_ds, test_ds, val_ds, num_of_classes, all_train_indexes, test_indexes


########

def stratified_load_data_by_path_information(train_x_path, train_y_encoded, test_x_path, test_y_encoded):
        
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # Use decode_png for PNG images
        # image = Image.open(path)
        
        return image, label

    # Preprocessing function
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    
    # Creating TensorFlow datasets
    ds_all_train = tf.data.Dataset.from_tensor_slices((np.array(train_x_path), np.array(train_y_encoded)))
    ds_test = tf.data.Dataset.from_tensor_slices((np.array(test_x_path), np.array(test_y_encoded)))
    

    # Load and preprocess the datasets
    all_train_ds = ds_all_train.map(load_image).map(preprocess).batch(32)
    test_ds = ds_test.map(load_image).map(preprocess).batch(32)
    
    return all_train_ds, test_ds
    



def convert_to_numpy(prefetch_dataset):
    """
    Convert a TensorFlow PrefetchDataset to NumPy arrays.

    Args:
    prefetch_dataset (tf.data.Dataset): A TensorFlow PrefetchDataset object.

    Returns:
    (np.ndarray, np.ndarray): Two NumPy arrays representing X_train and Y_train.
    """
    X_list, Y_list = [], []

    for X_batch, Y_batch in prefetch_dataset:
        # Convert the batches to NumPy and append them to the lists
        X_list.append(X_batch.numpy())
        Y_list.append(Y_batch.numpy())

    # Concatenate all batches
    X_train = np.concatenate(X_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)

    return (X_train, Y_train)





