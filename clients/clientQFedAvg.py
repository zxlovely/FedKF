import copy
import torch

from clients.clientBase import ClientBase
from data.utils import CONFIGS


class ClientQFedAvg(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientQFedAvg, self).__init__(args, client_id, data)

    def train_with_fairness(self, weight, q=0):
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
        print(f' ce_loss=%.5f' % avg_loss)
        # Compute delta weight.
        new_weight = copy.deepcopy(self.model.state_dict())
        delta_weight = copy.deepcopy(weight)
        for key in delta_weight:
            delta_weight[key] = (weight[key] - new_weight[key]) / CONFIGS[self.args.dataset]['lr']
        # Compute loss on the whole training data, with respect to the starting point (the global model).
        self.model.load_state_dict(weight)
        self.model.eval()
        losses = []
        for images, classes in self.train_loader:
            images, classes = images.to(self.args.device), classes.to(self.args.device)
            with torch.no_grad():
                outputs = self.model(images)
            losses.append(self.ce(outputs['logits'], classes).cpu().item())
        loss = sum(losses) / len(losses)
        # Compute delta.
        delta = copy.deepcopy(weight)
        for key in delta:
            delta[key] = ((loss + 1e-10) ** q) * delta_weight[key]
        # Compute h.
        conf = 0
        for key in delta_weight:
            conf += delta_weight[key].pow(2).sum()
        h = q * ((loss + 1e-10) ** (q - 1)) * conf + ((loss + 1e-10) ** q) / CONFIGS[self.args.dataset]['lr']

        return delta, h
