from lenet4 import lenet4 # fashion_mnist
from lenet5_SVHN import lenet5_SVHN
from lenet5 import lenet5 # mnist
from cifar10_conv import cifar10_conv # cifar10 conv-8
from resnet20_cifar10 import resnet20_cifar10 
from vgg16_SVHN import vgg16_SVHN 
from lenet5_mnist import lenet5_mnist 



subjects = ['lenet5_mnist', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN']
use_last_epoch = [True, True, True, True, True, True]

for model_name, dont_need_save_pervious_epoch in zip(subjects, use_last_epoch):
    if model_name == 'lenet4':
        raw_model = lenet4()
        
    elif model_name == 'lenet5_SVHN':
        raw_model = lenet5_SVHN()
        
    elif model_name == 'lenet5_mnist':
        raw_model = lenet5()
    
    elif model_name == 'cifar10':
        raw_model = cifar10_conv()
    
    elif model_name == 'resnet20_cifar10':
        raw_model = resnet20_cifar10()
        
    elif model_name == 'vgg16_SVHN':
        raw_model = vgg16_SVHN()

    elif model_name == 'resnet152_cifar100' or model_name == 'inceptionV3_imagenet':
        raise ValueError('This model is not supported by this script. Please use the inceptionV3_imagenet.py or resnet152_finetuning.py script.')
    
    # train the original model and save it
    model = raw_model.load_original_model()
    
    if not dont_need_save_pervious_epoch:
        for i in range(1,raw_model.original_epochs):
            print(f'epoch {i} is training')
            model = raw_model.train_an_epoch_more(model, i)
            raw_model.execute_model(model)
    
    raw_model.execute_model(model)


# raw_model = lenet5()
# start_time = time.time()
# model = raw_model.fit_model(12)
# end_time = time.time()

# execution_time = end_time - start_time
# print("Execution Time lenet5:", execution_time, "seconds")

# # for i in range(2,13):
# #     print(i)
# #     model = raw_model.train_an_epoch_more(model, i)
# #     raw_model.execute_model(model)
    
# # model = raw_model.load_original_model()
# # raw_model.execute_model(model)


# raw_model = resnet20_cifar10()
# start_time = time.time()
# model = raw_model.fit_model(raw_model.original_epochs)
# end_time = time.time()

# execution_time = end_time - start_time
# print("Execution Time resnet:", execution_time, "seconds")
