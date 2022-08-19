import copy
import torch

from clients.clientBase import ClientBase
from models.utils import get_cgenerator
from data.utils import CONFIGS


class ClientFedGen(ClientBase):
    def __init__(self, args, client_id, data):
        super(ClientFedGen, self).__init__(args, client_id, data)
        assert args.dataset in CONFIGS
        self.generator = get_cgenerator(dataset=args.dataset)
        self.generator.to(args.device)

    def base_train_with_count_label(self, weight, num_batches_per_epoch=31):
        self.model.load_state_dict(weight)
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss = []
        label_counter = [0 for _ in range(self.num_classes)]
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                for cls in classes.tolist():
                    label_counter[cls] += 1
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss = self.ce(outputs['logits'], classes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.cpu().item())
        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f' ce_loss: %.5f' % avg_loss)
        return copy.deepcopy(self.model.state_dict()), label_counter

    def train_with_regularization(self, generator_weight, global_weight, global_p, num_batches_per_epoch=31):
        self.generator.load_state_dict(generator_weight)
        self.model.load_state_dict(global_weight)
        self.generator.eval()
        self.model.train()
        print(f'[Client %2d]' % self.client_id, end='')
        batch_loss_local = []
        batch_loss_teacher = []
        label_counter = [0 for _ in range(self.num_classes)]
        for epoch in range(self.args.num_epochs_local_training):
            for batch_idx in range(num_batches_per_epoch):
                images, classes = self.get_batch_data()
                for cls in classes.tolist():
                    label_counter[cls] += 1
                images, classes = images.to(self.args.device), classes.to(self.args.device)
                outputs = self.model(images)
                loss_local = self.ce(outputs['logits'], classes)
                sampled_classes = self.random_state.choice(self.num_classes, self.args.batch_size_local_training,
                                                           p=global_p)
                sampled_classes = torch.from_numpy(sampled_classes).long().to(self.args.device)
                with torch.no_grad():
                    output_g = self.generator(sampled_classes)['output']
                outputs3 = self.model.classifier(output_g.detach())
                loss_teacher = self.ce(outputs3, sampled_classes)

                loss = loss_local + loss_teacher
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss_local.append(loss_local.cpu().item())
                batch_loss_teacher.append(loss_teacher.cpu().item())
        avg_loss_local = sum(batch_loss_local) / len(batch_loss_local)
        avg_loss_teacher = sum(batch_loss_teacher) / len(batch_loss_teacher)
        print(f' [loss_local=%.5f,' % avg_loss_local, end='')
        print(f' loss_teacher=%.5f]' % avg_loss_teacher, end='')
        print()

        return copy.deepcopy(self.model.state_dict()), label_counter
