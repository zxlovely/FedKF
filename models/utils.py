from models.lenet5 import LeNet5
from models.resnet import resnet8
from models.generator import Generator, CGenerator
from data.utils import CONFIGS


def get_model(dataset='EMNIST'):
    assert dataset in CONFIGS
    if dataset == 'EMNIST':
        model = LeNet5(img_channel=CONFIGS[dataset]['img_channel'], num_classes=CONFIGS[dataset]['num_classes'])
    elif dataset == 'CIFAR10':
        model = resnet8(num_classes=CONFIGS[dataset]['num_classes'], norm='gn')
    elif dataset == 'CIFAR100':
        model = resnet8(num_classes=CONFIGS[dataset]['num_classes'], norm='gn')
    else:
        raise KeyError

    return model


def get_generator(dataset='EMNIST'):
    assert dataset in CONFIGS

    return Generator(dataset=dataset)


def get_cgenerator(dataset='EMNIST'):
    assert dataset in CONFIGS

    return CGenerator(dataset=dataset)
