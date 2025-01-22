'''
you have to do download the dataset manually and put them into datasets directories in the ../datasets path and then set the data_dir to the directory where the .tar files are stored.
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

'''

import tensorflow as tf
import tensorflow_datasets as tfds

from keras.models import load_model


# Helper libraries
import os
# import numpy as np

import time
import numpy as np

from tqdm import tqdm


# Get imagenet labels
# labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())

# Set data_dir to a read-only storage of .tar files
# Set write_dir to a w/r storage

root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')

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

train_data, validation_data = tfds.load('imagenet2012', 
               data_dir=os.path.join(write_dir, 'data'),         
               split=['train', 'validation'], 
            #    split='validation', 
               shuffle_files=False, 
               download=False, 
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)




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


# Apply preprocessing to the dataset
train_data = train_data.map(preprocess_image).batch(512)
validation_data = validation_data.map(preprocess_image).batch(512)


# # Set the random seed for reproducibility
# seed = 42

# # Subsample 10% of the training data
# total_samples = 1200000
# ten_percent_samples = int(total_samples * 0.1)
# train_data_subsampled = train_data.take(ten_percent_samples).shuffle(buffer_size=ten_percent_samples, seed=seed)

# # Apply preprocessing and batching
# train_data_subsampled = train_data_subsampled.map(preprocess_image).batch(512)


# Load the trained ResNet50 model from TensorFlow
# model = tf.keras.applications.ResNet50(weights='imagenet')

# Load InceptionV3 model pretrained on ImageNet
model = tf.keras.applications.InceptionV3(weights='imagenet')


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.save('inceptionV3.h5')




s = time.time()
# Evaluate the model on the test data
_, accuracy = model.evaluate(validation_data, verbose=2)


def extract_labels(dataset):
    all_labels = []
    for _, labels in tqdm(tfds.as_numpy(dataset)):  # Convert dataset to NumPy format
        all_labels.extend(labels)
        np.save('all_actual_labels_validation', all_labels)
        
    return all_labels

# Extract labels from training data
train_labels = extract_labels(train_data)
np.save('all_actual_labels_train_final', train_labels)


# # train_labels = np.load('all_actual_labels_train_final.npy')
# # print(len(train_labels))

# # Extract labels from training data
test_labels = extract_labels(validation_data)
np.save('all_actual_labels_validation_final', test_labels)


# test_labels = np.load('all_actual_labels_validation_final.npy')
# print(len(test_labels))





















'''
Hello,

Thanks for replying, I actually fixed the error in the following way:

Download both ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar (remember to check that names are the same, otherwise manually change them). You do not need to unzip those files manually, otherwise it'd be a waste of memory and time. tfds will do it later through download_and_prepare.

Set a folder for your downloaded data, and another one for tfds dataset, in my case those are BASEDIR and DOWNLOADIR respectively.

I used the following code:

def load_ImageNet(ds_type, BASEDIR, batch_size):
    [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train', 'validation'],
                                             data_dir=BASEDIR, download=True, shuffle_files=True,
                                             as_supervised=True, with_info=True, 
                                             download_and_prepare_kwargs= {'download_dir':DOWNLOADIR})
    
    
    ds_train = prepare_training(ds_train, batch_size)
    ds_test = prepare_test(ds_test, batch_size)
    return [ds_train, ds_test], ds_info
It will take a while for tfds to prepare the dataset. Note that for above step, download = True, even if the dataset is already downloaded.
After that, tfds will create a folder in the following path : BASEDIR\imagenet2012\5.. , in my case as I am using tfds 3.2.1 -> BASEDIR\imagenet2012\5.0.0. For tfds4.0.0 -> BASEDIR\imagenet2012\5.1.0. If you check that folder, you will find the tf records files generated by tfds.
For using the generated dataset, set download = False, do the necessary pre-processing, and your dataset will be ready !
Note: I indeed update to the latest tfds version, but also tf must be updated, I was using previously tf2.0, had to update it.

If there is another way of doing it let me know, otherwise i hope this works for someone else ..
'''