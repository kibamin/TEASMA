# Here we difined constants

ERROR_RATE_THRESHOLD = 0.20


VALID_ATIVATION_FUNCTIONS = {
    'lenet5': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model
    
    'lenet5_SVHN': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model

        
    'cifar10': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model
    
    'lenet4': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model
    
    'resnet20_cifar10': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model
    'vgg16_SVHN': ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "sigmoid", "hard_sigmoid", "linear", "exponential"], #  "relu" is default for this model
    

}

MUTATION_OPERATORS_FOR_SUBJECT = {
    'lenet5' : ['GF','LA', 'NAI' ,'NEB', 'NS', 'WS'], # S1
    'cifar10' : ['GF','LA', 'NAI' ,'NEB', 'NS', 'WS', 'LD','LR'], # S2
    'lenet4' : ['GF','LA', 'NAI' ,'NEB', 'NS', 'WS'], # S3
    'lenet5_SVHN' : ['GF','LA', 'NAI' ,'NEB', 'NS', 'WS'], # S4
    'resnet20_cifar10' : ['GF', 'NAI' ,'NEB', 'NS', 'WS'], # S5
    
    
}
