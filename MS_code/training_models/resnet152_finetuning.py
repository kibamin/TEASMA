import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, Dropout,BatchNormalization,GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, ResNet152
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import ResNet152, ResNet101, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping





import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-epochs",
                        type=int,
                        default=10,
                        help="number of epochs")

parser.add_argument("--model_name", "-model_name",
                        type=str,
                        default='resnet152',
                        help="name of pretrained model")

args = parser.parse_args()
num_of_epochs = args.epochs
model_name = args.model_name





def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet.preprocess_input(input_images) ## for efficientNet use efficientnet
  return output_ims


'''
load dataset
'''
def load_preprocessed_dataset(dataset):
    if dataset == 'cifar10':
        (x_train, y_train) , (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_of_classes = 10
        
    if dataset == 'cifar100':
        (x_train, y_train) , (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        num_of_classes = 100
        
    
    # data preprocessing (it is necessary based on pretrained model here is ResNet50)
    x_train = preprocess_image_input(x_train)
    x_test = preprocess_image_input(x_test)
    y_train = tf.keras.utils.to_categorical(y_train, num_of_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_of_classes)
    
    return (x_train, y_train) , (x_test, y_test), num_of_classes


(X_train, Y_train) , (x_test, y_test), num_of_classes = load_preprocessed_dataset('cifar100')

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)
print('Stratify')

# y_train = tf.keras.utils.to_categorical(y_train, num_of_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_of_classes)
# y_val = tf.keras.utils.to_categorical(y_val, num_of_classes)



if model_name == 'resnet152':
    base_model = ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

elif model_name == 'resnet50':
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
elif model_name == 'resnet101':
    base_model = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))


model = Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(.25),
    # Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(num_of_classes, activation='softmax')
])

# Freeze the base model's layers
base_model.trainable = False

# for layer in base_model.layers:
#     layer.trainable = False
    
    

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, restore_best_weights=True)

#reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)


model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train,validation_data = (x_val, y_val), epochs=num_of_epochs, batch_size=64, verbose=2,
          callbacks=[rlrop])




# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {accuracy*100:.2f}%")

loss, accuracy = model.evaluate(X_train, Y_train, verbose=2)
print(f"Train accuracy: {accuracy*100:.2f}%")



if not os.path.exists('../models/resnet152_cifar100'):
    os.makedirs('../models/resnet152_cifar100')
    
model.save('../models/resnet152_cifar100/original_model.h5')
