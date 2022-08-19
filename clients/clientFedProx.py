import copy
import torch

from clients.clientBase import ClientBase


class ClientFedProx(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientFedProx, self).__init__(args, client_id, data)

    def train_with_regularize(self, weight, num_batches_per_epoch=31, mu=0.01):
        self.model.load_state_dict(weight)
        self.model.train()
        init_weight = torch.cat([torch.flatten(param) for param in copy.deepcopy(self.model).parameters()])
        print(f'[Client %2d]:' % self.client_id, end='')
        batch_loss_ce = []
        batch_loss_reg = []
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss_ce = self.ce(outputs['logits'], classes)
                curr_weight = torch.cat([torch.flatten(param) for param in self.model.parameters()])
                loss_reg = torch.pow(curr_weight - init_weight.detach(), 2).sum()
                loss = loss_ce + loss_reg * mu / 2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss_ce.append(loss_ce.cpu().item())
                batch_loss_reg.append(loss_reg.cpu().item())
        avg_loss_ce = sum(batch_loss_ce) / len(batch_loss_ce)
        avg_loss_reg = sum(batch_loss_reg) / len(batch_loss_reg)
        print(' [loss_ce=%.5f, loss_reg=%.5f]' % (avg_loss_ce, avg_loss_reg))
        return copy.deepcopy(self.model.state_dict())
