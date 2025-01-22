from __future__ import print_function
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.models import load_model
import numpy as np
import os


root = os.path.dirname(os.path.abspath(__file__)) 


class resnet20_cifar10():
    def __init__(self, dataset_name = 'cifar10') -> None:
        self.model_name = 'resnet20_cifar10'
        self.dataset = dataset_name
        self.original_epochs = 100
        self.load_dataset()
        self.batch_size = 128
    
    def load_dataset(self):
        # Load Fashion-MNIST dataset
        
        
            
        if self.dataset in ['cifar10_brightness','cifar10_contrast', 'cifar10_rotation', 'cifar10_scale', 'cifar10_shear','cifar10_translation']:
            CLIP_MIN = -0.5
            CLIP_MAX = 0.5

            (x_train, y_train), (_, y_test) = cifar10.load_data() ## because we want to use shifted test input to test in different distribution of train inputs

            x_train = x_train.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)

            transformed_test_set_path = os.path.join(root,'..','datasets', 'transformed', 'cifar10', f'{self.dataset}_test.npy')
            target_test_x = np.load(transformed_test_set_path)
            target_test_x = target_test_x.astype('float32')
            target_test_x = (target_test_x / 255.0) - (1.0 - CLIP_MAX)
            
            y_train = tf.keras.utils.to_categorical(y_train, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)
            
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (target_test_x, y_test)
            
        
        else: ## normal cifar10 dataset without shift dataset
            CLIP_MIN = -0.5
            CLIP_MAX = 0.5

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

            y_train = tf.keras.utils.to_categorical(y_train, 10)
            y_test = tf.keras.utils.to_categorical(y_test, 10)

            
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (x_train, y_train), (x_test, y_test)
            
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
        
        
        
    def fit_model(self, epochs = 1):
        
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = self.batch_size

        data_augmentation = True
        num_classes = 10
        input_shape = x_train.shape[1:]


        # Subtracting pixel mean improves accuracy
        subtract_pixel_mean = True

        n = 3

        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
        version = 1

        # Computed depth from supplied model parameter n
        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2

        model_type = 'ResNet%dv%d' % (depth, version)

        def lr_schedule(epoch):
            """Learning Rate Schedule

            Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
            Called automatically every epoch as part of callbacks during training.

            # Arguments
                epoch (int): The number of epochs

            # Returns
                lr (float32): learning rate
            """
            lr = 1e-3
            if epoch > 130:
                lr *= 0.5e-3
            elif epoch > 110:
                lr *= 1e-3
            elif epoch > 80:
                lr *= 1e-2
            elif epoch > 50:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr

        def resnet_layer(inputs,
                        num_filters=16,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)

            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        def resnet_v1(input_shape, depth, num_classes=10):
            """ResNet Version 1 Model builder [a]

            Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
            Last ReLU is after the shortcut connection.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filters is
            doubled. Within each stage, the layers have the same number filters and the
            same number of filters.
            Features maps sizes:
            stage 0: 32x32, 16
            stage 1: 16x16, 32
            stage 2:  8x8,  64
            The Number of parameters is approx the same as Table 6 of [a]:
            ResNet20 0.27M
            ResNet32 0.46M
            ResNet44 0.66M
            ResNet56 0.85M
            ResNet110 1.7M

            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)

            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 6 != 0:
                raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
            # Start model definition.
            num_filters = 16
            num_res_blocks = int((depth - 2) / 6)

            inputs = Input(shape=input_shape)
            x = resnet_layer(inputs=inputs)
            # Instantiate the stack of residual units
            for stack in range(3):
                for res_block in range(num_res_blocks):
                    strides = 1
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        strides = 2  # downsample
                    y = resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    strides=strides)
                    y = resnet_layer(inputs=y,
                                    num_filters=num_filters,
                                    activation=None)
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                        num_filters=num_filters,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
                    x = keras.layers.add([x, y])
                    x = Activation('relu')(x)
                num_filters *= 2

            # Add classifier on top.
            # v1 does not use BN after last shortcut connection-ReLU
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

        def resnet_v2(input_shape, depth, num_classes=10):
            """ResNet Version 2 Model builder [b]

            Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
            bottleneck layer
            First shortcut connection per layer is 1 x 1 Conv2D.
            Second and onwards shortcut connection is identity.
            At the beginning of each stage, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filter maps is
            doubled. Within each stage, the layers have the same number filters and the
            same filter map sizes.
            Features maps sizes:
            conv1  : 32x32,  16
            stage 0: 32x32,  64
            stage 1: 16x16, 128
            stage 2:  8x8,  256

            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)

            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 9 != 0:
                raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
            # Start model definition.
            num_filters_in = 16
            num_res_blocks = int((depth - 2) / 9)

            inputs = Input(shape=input_shape)
            # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
            x = resnet_layer(inputs=inputs,
                            num_filters=num_filters_in,
                            conv_first=True)

            # Instantiate the stack of residual units
            for stage in range(3):
                for res_block in range(num_res_blocks):
                    activation = 'relu'
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        num_filters_out = num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        num_filters_out = num_filters_in * 2
                        if res_block == 0:  # first layer but not first stage
                            strides = 2    # downsample

                    # bottleneck residual unit
                    y = resnet_layer(inputs=x,
                                    num_filters=num_filters_in,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=activation,
                                    batch_normalization=batch_normalization,
                                    conv_first=False)
                    y = resnet_layer(inputs=y,
                                    num_filters=num_filters_in,
                                    conv_first=False)
                    y = resnet_layer(inputs=y,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    conv_first=False)
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                        num_filters=num_filters_out,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
                    x = keras.layers.add([x, y])

                num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

        if version == 2:
            model = resnet_v2(input_shape=input_shape, depth=depth)
        else:
            model = resnet_v1(input_shape=input_shape, depth=depth)


        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=lr_schedule(0)),
                    metrics=['accuracy'])
        model.summary()
        print(model_type)        

                
                
                
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=5,
                                    min_lr=0.5e-6)

        callbacks = [lr_reducer, lr_scheduler]

        # Run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                epochs=epochs, verbose=0, workers=4,
                                callbacks=callbacks,
                                steps_per_epoch= x_train.shape[0] // batch_size,)

        
        
        
        
        # optimizer = keras.optimizers.Adam()
        
        # # lr_scheduler = LearningRateScheduler(lr_schedule)

        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        return model
    
    def train_an_epoch_more(self,model):
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        batch_size = self.batch_size
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
        return model


    def load_original_model(self):
        original_model_path = os.path.join(root,'..', 'models', self.model_name,'original_model.h5')
        print(original_model_path)

        if os.path.exists(original_model_path):
            model = load_model(original_model_path,compile=False)
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        
        else:
            print('start training')
            model = self.fit_model(epochs = self.original_epochs)
            model.save(original_model_path)
            return model
        
    def execute_model(self, model):
        score = model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])

        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        print(model.summary())


        
    
    