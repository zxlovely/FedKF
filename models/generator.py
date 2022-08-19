import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as f

from data.utils import CONFIGS

GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'EMNIST': (256, 120, 1, 26, 32),
    'CIFAR10': (512, 256, 3, 10, 64),
    'CIFAR100': (512, 256, 3, 100, 64)

}


class Normalize(nn.Module):
    def __init__(self, dataset='MNIST'):
        super(Normalize, self).__init__()
        assert dataset in CONFIGS
        self.mean = CONFIGS[dataset]['mean']
        self.std = CONFIGS[dataset]['std']

    def forward(self, x):
        out = f.normalize(x, self.mean, self.std)

        return out


class Generator(nn.Module):
    def __init__(self, dataset='CIFAR10'):
        super(Generator, self).__init__()
        assert dataset in CONFIGS
        self.num_classes = CONFIGS[dataset]['num_classes']
        self.noise_dim = CONFIGS[dataset]['noise_dim']
        img_size = CONFIGS[dataset]['img_size']
        img_channel = CONFIGS[dataset]['img_channel']
        input_dim = self.noise_dim
        self.init_size = img_size // 4
        self.fc = nn.Linear(input_dim, 128 * self.init_size * self.init_size)
        self.bn = nn.BatchNorm2d(128)
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.normalize = Normalize(dataset)
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, noise, out_image=False):
        out = self.fc(noise)
        img = self.bn(out.view(-1, 128, self.init_size, self.init_size))
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        out = self.normalize(img)
        if out_image:
            return out, img

        return out


class CGenerator(nn.Module):
    def __init__(self, dataset='CIFAR10', latent_layer_idx=-1):
        super(CGenerator, self).__init__()
        print("Dataset {}".format(dataset))
        self.dataset = dataset
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduction='none')  # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU(inplace=True)
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Sequential(
            nn.Linear(self.fc_configs[-1], self.latent_dim),
            nn.ReLU(inplace=True)
        )
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.size(0)
        noise = torch.rand(batch_size, self.noise_dim).to(labels.device)  # sampling from Gaussian ?
        result['noise'] = noise
        # one-hot (sparse) vector
        y_input = torch.FloatTensor(batch_size, self.n_class).to(labels.device)
        y_input.zero_()
        # labels = labels.view
        y_input.scatter_(1, labels.view(-1, 1), 1)
        z = torch.cat((noise, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
