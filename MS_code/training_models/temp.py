from lenet5_mnist import lenet5_mnist
from keras.models import load_model
import numpy as np



datasets = ['mnist_c_brightness','mnist_c_contrast', 'mnist_c_rotation', 'mnist_c_scale', 'mnist_c_shear','mnist_c_translation']

for ds in datasets:
    print(ds)
    model_object = lenet5_mnist(dataset_name = ds)
    (X_train, _ ), (x_test, y_test) = model_object.load_dataset()
    model = model_object.load_original_model()
    
    if model:
        score = model.evaluate(x_test, y_test, verbose=2)
        print(f'test set {ds} accuracy:', score[1])
        # print('test accuracy of target model: %.4f' % score[1])
        
        
        
        
