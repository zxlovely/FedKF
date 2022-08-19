import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from models.utils import get_model
from data.utils import CONFIGS


class ClientBase(object):
    def __init__(self, args, client_id, data):
        assert args.dataset in CONFIGS
        self.args = args
        self.client_id = client_id
        self.data_train = copy.copy(data)
        self.data_test = copy.copy(data)
        self.num_classes = CONFIGS[args.dataset]['num_classes']
        self.random_state = np.random.RandomState(client_id)
        self.random_state.shuffle(data.indices)
        self.data_train.indices = data.indices[:int(len(data.indices) * args.train_proportion)]
        self.data_test.indices = data.indices[int(len(data.indices) * args.train_proportion):]
        self.model = get_model(dataset=args.dataset)
        self.model.to(args.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=CONFIGS[args.dataset]['lr'], momentum=CONFIGS[args.dataset]['momentum'],
            weight_decay=CONFIGS[args.dataset]['weight_decay']
        )
        self.sl1 = nn.SmoothL1Loss()
        self.nll = nn.NLLLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.train_loader = DataLoader(dataset=self.data_train, batch_size=args.batch_size_local_training, shuffle=True,
                                       drop_last=True)
        self.test_loader = DataLoader(dataset=self.data_test, batch_size=256, shuffle=False)
        self.train_iter = iter(self.train_loader)

    def get_batch_data(self):
        try:
            images, labels = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            images, labels = next(self.train_iter)
        return images, labels

    def test(self, weight):
        self.model.load_state_dict(weight)
        self.model.eval()
        num_corrects = 0
        batch_loss = []
        for data_batch in self.test_loader:
            images, labels = data_batch
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            with torch.no_grad():
                outputs = self.model(images)
            predicts = outputs['logits'].argmax(dim=-1)
            num_corrects += torch.eq(predicts, labels).sum().cpu().item()
            loss = self.ce(outputs['logits'], labels).cpu().item()
            batch_loss.append(loss)
        loss = sum(batch_loss) / len(batch_loss)

        return loss, num_corrects

    '''same batch size, different number of batches'''
    def base_train1(self, weight):
        self.model.load_state_dict(weight)
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss = []
        for epoch in range(self.args.num_epochs_local_training):
            for images, classes in self.train_loader:
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss = self.ce(outputs['logits'], classes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.cpu().item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f' ce_loss: %.5f' % avg_loss)
        return copy.deepcopy(self.model.state_dict())

    '''same batch size, same number of steps'''
    def base_train2(self, weight, num_batches_per_epoch):
        self.model.load_state_dict(weight)
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss = []
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss = self.ce(outputs['logits'], classes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.cpu().item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f' ce_loss: %.5f' % avg_loss)
        return copy.deepcopy(self.model.state_dict())

    '''same batch size, same number of steps, but different learning rate'''
    def base_train3(self, weight, eta, num_batches_per_epoch):
        self.model.load_state_dict(weight)
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss = []
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss = self.ce(outputs['logits'], classes)
                loss = loss * eta
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.cpu().item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f' ce_loss: %.5f' % avg_loss)
        return copy.deepcopy(self.model.state_dict())

    '''different batch size, same number of steps'''
    def base_train4(self, weight, batch_size):
        self.model.load_state_dict(weight)
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss = []
        train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True, drop_last=True)
        for epoch in range(self.args.num_epochs_local_training):
            for batch_data in train_loader:
                images, classes = batch_data
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss = self.ce(outputs['logits'], classes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.cpu().item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f' ce_loss: %.5f' % avg_loss)
        return copy.deepcopy(self.model.state_dict())
