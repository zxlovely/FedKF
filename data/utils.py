from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

CONFIGS = {
    'EMNIST': {
        'split': 'letters',
        'proportion': 0.1,
        'num_classes': 26,
        'img_size': 32,
        'img_channel': 1,
        'noise_dim': 100,
        'mean': [0.1739],
        'std': [0.3170],
        'root': '/path/to/EMNIST',
        'lr': 0.01,
        'momentum': 0,
        'weight_decay': 0,
        'g_use_kl': False
    },
    'CIFAR10': {
        'proportion': 1,
        'num_classes': 10,
        'img_size': 32,
        'img_channel': 3,
        'noise_dim': 1000,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'root': '/path/to/CIFAR10',
        'lr': 0.1,
        'momentum': 0,
        'weight_decay': 0,
        'g_use_kl': True
    },
    'CIFAR100': {
        'proportion': 1,
        'num_classes': 100,
        'img_size': 32,
        'img_channel': 3,
        'noise_dim': 1000,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'root': '/path/to/CIFAR100',
        'lr': 0.1,
        'momentum': 0,
        'weight_decay': 0,
        'g_use_kl': True
    }
}


class MyEMNIST(Dataset):
    def __init__(self, raw_data, split, transform=None):
        self.raw_data = raw_data
        self.transform = transform
        self.split = split

    def __getitem__(self, item):
        img, cls = self.raw_data[item]
        if self.transform is not None:
            img = self.transform(img)
            img = img.permute(0, 2, 1)

        if self.split == 'letters':
            cls -= 1

        return img, cls

    def __len__(self):
        return len(self.raw_data)


def load_dataset(dataset='EMNIST'):
    assert dataset in CONFIGS
    if dataset == 'EMNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIGS[dataset]['mean'], std=CONFIGS[dataset]['std'])
        ])
        data_train = datasets.EMNIST(root=CONFIGS[dataset]['root'], split=CONFIGS[dataset]['split'], train=True)
        raw_data = MyEMNIST(raw_data=data_train, split=CONFIGS[dataset]['split'], transform=transform)
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIGS[dataset]['mean'], std=CONFIGS[dataset]['std'])
        ])
        raw_data = datasets.CIFAR10(root=CONFIGS[dataset]['root'], train=True, transform=transform)
    elif dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIGS[dataset]['mean'], std=CONFIGS[dataset]['std'])
        ])
        raw_data = datasets.CIFAR100(root=CONFIGS[dataset]['root'], train=True, transform=transform)
    else:
        raise KeyError

    return raw_data


def classify_indices_by_class(data):
    class2indices = {}
    for idx, (x, y) in enumerate(data):
        if y not in class2indices:
            class2indices[y] = []
        class2indices[y].append(idx)

    return class2indices


class Indices2Dataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __getitem__(self, item):
        idx = self.indices[item]
        image, label = self.data[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


def show_clients_data_distribution(raw_data, id2indices: dict, num_classes: int):
    print(f'Clients\\Classes', end=' ')
    for cla in range(num_classes):
        print(f'%5d' % cla, end=' ')
    print()
    class2num = [0 for _ in range(num_classes)]
    ids = sorted(list(id2indices.keys()))
    for client_id in ids:
        class2num_client = [0 for _ in range(num_classes)]
        for idx in id2indices[client_id]:
            cla = raw_data[idx][1]
            class2num_client[cla] += 1
            class2num[cla] += 1
        print(f'       %2d      ' % client_id, end=' ')
        for num in class2num_client:
            print(f'%5d' % num, end=' ')
        print(f'  %5d' % sum(class2num_client))
    print('               ', end=' ')
    for num in class2num:
        print(f'%5d' % num, end=' ')
    print(f'  %5d' % sum(class2num))
