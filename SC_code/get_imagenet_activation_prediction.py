import os
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import numpy as np
import sys




job_id = os.getenv('SLURM_JOB_ID')

# Check if the environment variable exists
if job_id is not None:
    print(f"Running under Slurm with Job ID: {job_id}")
else:
    print("Not running under Slurm or Job ID not found.")

import logging

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


logger = MyLogger(f'{job_id}_imagenet_activation_trace.py') 



gpus = tf.config.experimental.list_physical_devices('GPU')

root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')

data_dir = os.path.join(root, 'MS_code','datasets', 'imagenet') # this is the path to the imagnet tar files in ../MS_code/datasets/imagenet (check the MS_code/README.md for more info)
write_dir = os.path.join(root, 'MS_code', 'datasets', 'tf-imagenet-dirs') # this is the path to the extracted files in ../MS_code/datasets/tf-imagenet-dirs (check the MS_code/README.md for more info)


download_config = tfds.download.DownloadConfig(
                      extract_dir=os.path.join(write_dir, 'extracted'),
                      manual_dir=data_dir
                  )
download_and_prepare_kwargs = {
    'download_dir': os.path.join(write_dir, 'downloaded'),
    'download_config': download_config,
}

train_data, validation_data = tfds.load('imagenet2012', 
               data_dir=os.path.join(write_dir, 'data'),         
               split=['train', 'validation'], 
               shuffle_files=False, 
               download=False, 
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)



def preprocess_image(image, label):
    initial_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    ratio = 256.0 / tf.reduce_min(initial_shape)
    new_shape = tf.cast(initial_shape * ratio, tf.int32)
    image = tf.image.resize(image, new_shape)
    # Crop the central 224x224
    # image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

# Apply preprocessing to the dataset
train_data = train_data.map(preprocess_image).batch(512)
validation_data = validation_data.map(preprocess_image).batch(512)






def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


model = InceptionV3(weights='imagenet')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# len_train = len(np.load('/home/amin/IDC/SC/SA_results/inceptionV3_imagenet/imagenet_train_pred.npy'))
# len_test = len(np.load('/home/amin/IDC/SC/SA_results/inceptionV3_imagenet/imagenet_test_pred.npy'))
# print(len_train)
# print(len_test)




layer_names = ['activation_93']
temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )


ATS_save_path = 'SA_results/inceptionV3_imagenet/ATS_predicteds'

if not os.path.exists(ATS_save_path):
    os.makedirs(ATS_save_path)

ats = []
batch_counter = 1

for batch in train_data:
    logger.info(f'train:{batch_counter}')
    
    batch_data = batch[0].numpy()
    predictions = temp_model.predict(batch_data, verbose=1) # shape => (512, 8, 8, 192)
    predictions = np.array([_aggr_output(i) for i in predictions])
    ats.append(predictions)
    
    # if batch_counter % 20 == 0:
    #     np.save(os.path.join(ATS_save_path, f'train_ats_{batch_counter}.npy'), np.vstack(ats))
    #     logger.info(f'save train_ats batch {batch_counter}')
    
    batch_counter += 1
    
ats = np.vstack(ats)
np.save(os.path.join(ATS_save_path, f'train_ats.npy'), ats)
logger.info('saved final ats of train')





ats = []
batch_counter = 1

for batch in validation_data:
    logger.info(f'test:{batch_counter}')
    batch_data = batch[0].numpy()
    predictions = temp_model.predict(batch_data, verbose=0) # shape => (512, 8, 8, 192)
    predictions = np.array([_aggr_output(i) for i in predictions])
    ats.append(predictions)
    
    if batch_counter % 20 == 0:
        np.save(os.path.join(ATS_save_path, f'test_ats_{batch_counter}.npy'), np.vstack(ats))
        logger.info(f'save test_ats batch {batch_counter}')
    
    batch_counter += 1
    
ats = np.vstack(ats)
np.save(os.path.join(ATS_save_path, f'test_ats.npy'), ats)
logger.info('saved final ats of test')

