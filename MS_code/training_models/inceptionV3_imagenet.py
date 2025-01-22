

'''
Before executing this file, make sure to download the imagenet dataset from the official website and put them into the datasets directory in the ../datasets/imagenet path and then set the data_dir to the directory where the .tar files are stored.
see the imagenet_evaluation.py file for more information.
'''

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import os
# import numpy as np

import time
import numpy as np

root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')

class inceptionV3_imagenet():
    def __init__(self) -> None:
        self.model_name = 'inceptionV3'
        self.dataset = 'imagenet'
        self.original_epochs = 0
        

data_dir = os.path.join(root, 'datasets', 'imagenet')
write_dir = os.path.join(root, 'datasets', 'tf-imagenet-dirs')

# Construct a tf.data.Dataset
download_config = tfds.download.DownloadConfig(
                      extract_dir=os.path.join(write_dir, 'extracted'),
                      manual_dir=data_dir
                  )
download_and_prepare_kwargs = {
    'download_dir': os.path.join(write_dir, 'downloaded'),
    'download_config': download_config,
}

# imagenet_data, info = tfds.load('imagenet2012:5.1.0', with_info=True, split=['train', 'validation', 'test'])

# train_data, validation_data = tfds.load('imagenet2012', 
#                data_dir=os.path.join(write_dir, 'data'),         
#                split=['train', 'validation'], 
#             #    split='validation', 
#                shuffle_files=False, 
#                download=False, 
#                as_supervised=True,
#                download_and_prepare_kwargs=download_and_prepare_kwargs)




def preprocess_image1(image, label):
    '''
    executed with this function and achieved 70% acc on validation set
    '''
    # Resize and preprocess as required by your specific model
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label



def preprocess_image(image, label):
    '''
    This message will be only logged once.
    98/98 [==============================] - 530s 5s/step - loss: 1.0475 - accuracy: 0.7505
    validation Accuracy: 0.750540018081665
    '''
    # Resize the smaller side to 256 while maintaining aspect ratio
    initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    ratio = 256.0 / tf.reduce_min(initial_shape)
    new_shape = tf.cast(initial_shape * ratio, tf.int32)
    image = tf.image.resize(image, new_shape)
    # Crop the central 224x224
    # image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    # Apply the specific preprocess input function for ResNet50
    # image = tf.keras.applications.resnet50.preprocess_input(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image) ## 5 min with batch size 512
    return image, label





# Load the trained ResNet50 model from TensorFlow
# model = tf.keras.applications.ResNet50(weights='imagenet')

# Load InceptionV3 model pretrained on ImageNet
model = tf.keras.applications.InceptionV3(weights='imagenet')


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

save_path = os.path.join(root, 'models', 'inceptionV3_imagenet')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
model.save(os.path.join(save_path, 'original_model.h5'))

