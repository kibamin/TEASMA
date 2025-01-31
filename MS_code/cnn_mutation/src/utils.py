import numpy as np
from keras.utils import np_utils
import random
import keras


def color_preprocessing(x_train, x_test, mean, std):
    """
    process the input data, scaling, adding bias...
    :param x_train: training data
    :param x_test: testing data
    :param mean: scale
    :param std: bias
    :return: training and testing data after pre-processing
    """
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if len(x_train.shape) == 4:
        for i in range(x_train.shape[3]):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        for i in range(x_test.shape[3]):
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    elif len(x_train.shape) == 3:
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    return x_train, x_test


def model_predict(model, x, y):
    """

    :param model:
    :param x:
    :param y:
    :return:
    """
    y_p = model.predict(x, verbose=2)
    y_p_class = np.argmax(y_p, axis=1)
    correct = np.sum(y.flatten() == y_p_class.flatten())
    acc = float(correct) / len(x)
    return acc, y_p

# didn't use
def load_preprocessed_data(data, mean, std):
    """
    load data
    :param data: input data
    :param mean: scale
    :param std: bias
    :return: training and testing data, format: (data, label)
    """
    f = np.load(data)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train, x_test = color_preprocessing(x_train, x_test, mean, std)
    return (x_train, y_train), (x_test, y_test)


def get_type_layers(model):
    """
    get each type of layer name
    :param model:
    :return: dense layer list, convolution layer list
    """
    dense_layer_list = []
    convolution_layer_list = []
    dense_con_layer_list = []
    flatten_layer_list = []
    for layer in model.layers:
        # choose dense, convolution and flatten layer
        if isinstance(layer, keras.layers.core.Dense):
            dense_layer_list.append(layer.name)
            dense_con_layer_list.append(layer.name)
        elif isinstance(layer, keras.layers.convolutional.Conv2D): # keras.layers.convolutional._Conv ---changeTo--> keras.layers.convolutional.Conv2D
            convolution_layer_list.append(layer.name)
            dense_con_layer_list.append(layer.name)
        elif isinstance(layer, keras.layers.core.Flatten):
            flatten_layer_list.append(layer.name)
    return dense_layer_list, convolution_layer_list, dense_con_layer_list, flatten_layer_list


def find_sameshape_layer(model):
    """
    find the layers which can be deleted or duplicated
    :param model: model used
    :return: layer list
    """
    candidate_layer_list = []
    layer_num = len(model.layers)
    # has hidden layers?
    if layer_num > 2:
        for layer_index in range(layer_num):
            # pass input and output layers
            if layer_index == 0 or layer_index == (layer_num - 1):
                continue
            # last layer's output shape = next layer's input shape
            # print(f'{model.layers[layer_index].output_shape} <--> {model.layers[layer_index].input_shape}')
            if model.layers[layer_index].output_shape == model.layers[layer_index].input_shape:
                candidate_layer_list.append(model.layers[layer_index].name)
    return candidate_layer_list


# didn't use
def generate_correct_input(model, x_train, y_train, x_test, y_test, classlabel):
    """
    select input which is correct prediction
    :param model:
    :param x_train: training data
    :param y_train: training label
    :param x_test: test data
    :param y_test: test label
    :param classlabel:  class label list eg. [0,1,2,3,4,5,6,7,8,9]
    :return: correct training data, testing data
    """
    classnum = len(classlabel)
    # the data, shuffled and split between train and test sets
    y_train_cp = y_train.copy()
    y_test_cp = y_test.copy()

    # init correct prediction index
    correct_train_index = []
    correct_test_index = []
    for i in range(classnum):
        correct_train_index.append([])
        correct_test_index.append([])
    # load model
    model = model
    # predict and save the correct index
    y_pred_model_train = model.predict(x_train)
    y_prediction_classes_model = np.argmax(y_pred_model_train, axis=1)
    y_train = np_utils.to_categorical(y_train, classnum)
    y_train_classes = np.argmax(y_train, axis=1)
    correct_train = np.where(y_prediction_classes_model == y_train_classes)[0]

    y_pred_model_test = model.predict(x_test)
    y_prediction_classes_model_test = np.argmax(y_pred_model_test, axis=1)
    y_test = np_utils.to_categorical(y_test, classnum)
    y_test_classes = np.argmax(y_test, axis=1)
    correct_test = np.where(y_prediction_classes_model_test == y_test_classes)[0]
    print(len(correct_train))
    print(len(correct_test))
    for i in range(len(correct_train)):
        for j in range(classnum):
            if y_train_cp[correct_train[i]] == classlabel[j]:
                correct_train_index[j].append(correct_train[i])
                break

    # print(len(correct_train_index[0]))
    for i in range(len(correct_test)):
        for j in range(classnum):
            if y_test_cp[correct_test[i]] == classlabel[j]:
                correct_test_index[j].append(correct_test[i])
                break
    # save correct index
    correct_train = []
    correct_test = []

    for i in range(classnum): ## select for each class
        # training data uniform random sampling
        correct_train.extend(random.sample(list(correct_train_index[i]), 500))

        # test data uniform random sampling
        correct_test.extend(random.sample(list(correct_test_index[i]), 100))

    print("data preparation finish")
    return correct_train, correct_test


def summary_model(model):
    """

    :param model:
    :return:
    """
    weights_dict = {}
    neuron_dict = {}
    weight_count = 0
    neuron_count = 0
    for layer in model.layers:
        # we only calculate dense later and conv layer
        '''
        for error: AttributeError: module 'keras.layers.convolutional' has no attribute '_Conv'
        Try this: pip install keras==2.2.2 & pip install tensorflow==1.10.0
        
        _Conv method not available with later version keras 2.4, the way to solve this is use Conv2D API
        '''
        if isinstance(layer, keras.layers.core.Dense) or isinstance(layer, keras.layers.convolutional.Conv2D):
            w_n = layer.get_weights()[0].size
            n_n = layer.output_shape[-1]
            weight_count += w_n
            neuron_count += n_n
            weights_dict[layer.name] = weight_count
            neuron_dict[layer.name] = neuron_count
    return weight_count, neuron_count, weights_dict, neuron_dict


