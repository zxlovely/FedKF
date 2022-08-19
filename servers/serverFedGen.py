import copy
import numpy as np
import torch
from torch import optim

from servers.serverBase import ServerBase
from clients.clientFedGen import ClientFedGen
from models.utils import get_cgenerator


class ServerFedGen(ServerBase):
    def __init__(self, args):
        super(ServerFedGen, self).__init__(args)
        self.clients = self._init_clients()
        self.generator = get_cgenerator(dataset=args.dataset)
        self.generator.to(args.device)
        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=1e-4)
        self.all_train_data_num = sum([len(self.clients[client_id].data_train) for client_id in self.clients])
        self.num_batches_per_epoch = self.all_train_data_num // (
                self.args.num_clients * self.args.batch_size_local_training)
        print(f'number of total training data: {self.all_train_data_num}')
        print(f'number of batches per epoch: {self.num_batches_per_epoch}')

    def _init_clients(self):
        clients = {}
        for client_id in self.id2data:
            data = self.id2data[client_id]
            clients[client_id] = ClientFedGen(self.args, client_id, data)

        return clients

    def run(self):
        global_p = None
        for r in range(self.args.num_rounds):
            print('*' * 52, end=' ')
            print(f'Round: %3d/%3d' % (r + 1, self.args.num_rounds), end=' ')
            print('*' * 52)
            active_clients = self.random_state.choice(
                a=self.args.num_clients,
                size=self.args.num_active_clients,
                replace=False
            )
            id2counter = {}
            for client_id in active_clients:
                client = self.clients[client_id]
                if global_p is None:
                    local_weight, local_counter = client.base_train_with_count_label(self.latest_weight,
                                                                                     self.num_batches_per_epoch)
                else:
                    local_weight, local_counter = client.train_with_regularization(
                        generator_weight=self.generator.state_dict(),
                        global_weight=self.latest_weight,
                        global_p=global_p,
                        num_batches_per_epoch=self.num_batches_per_epoch
                    )
                self.id2weight[client_id] = local_weight
                self.id2amount[client_id] = len(client.data_train)
                id2counter[client_id] = local_counter
            # Compute global distribution
            collector = []
            for client_id in active_clients:
                num_samples = self.id2amount[client_id]
                counter = id2counter[client_id]
                collector.append(np.array(counter) * num_samples)
            global_counter = sum(collector)
            global_p = global_counter / global_counter.sum()
            for pk in global_p:
                print('%.3f' % pk, end=' ')
            # Train generator
            print()
            self.train_generator(active_clients=active_clients, global_p=global_p, num_steps=50, batch_size=64,
                                 global_r=r)
            # Server update
            self.aggregation(active_clients)
            # Aggregate local results
            self.aggregate_local_results('latest')
            self.aggregate_local_results('global')
        self._print_results('latest')
        self._print_results('global')

    def train_generator(self, active_clients, global_p, num_steps, batch_size, global_r):
        random_state = np.random.RandomState(seed=global_r)
        teachers = []
        samples = []
        for client_id in active_clients:
            samples.append(self.id2amount[client_id])
            teacher_weight = self.id2weight[client_id]
            teacher = copy.deepcopy(self.model)
            teacher.load_state_dict(teacher_weight)
            teacher.eval()
            teachers.append(teacher)
        teacher_losses = []
        diversity_losses = []
        print('train generator:', end='')
        self.generator.train()
        for step in range(num_steps):
            sampled_classes = random_state.choice(self.num_classes, batch_size, p=global_p)
            sampled_classes = torch.LongTensor(sampled_classes).to(self.args.device)
            result = self.generator(sampled_classes)
            diversity_loss = self.generator.diversity_loss(result['noise'], result['output'])
            ######### get teacher loss ############
            teacher_logits_collector = []
            for teacher, sample in zip(teachers, samples):
                teacher_logits = teacher.classifier(result['output'])
                teacher_logits_collector.append(teacher_logits * sample)
            avg_teacher_logits = sum(teacher_logits_collector) / sum(samples)
            teacher_loss = self.ce(avg_teacher_logits, sampled_classes)
            loss = teacher_loss + diversity_loss
            self.optimizer_g.zero_grad()
            loss.backward()
            self.optimizer_g.step()
            teacher_losses.append(teacher_loss.cpu().item())
            diversity_losses.append(diversity_loss.cpu().item())
        avg_teacher_loss = sum(teacher_losses) / len(teacher_losses)
        avg_diversity_loss = sum(diversity_losses) / len(diversity_losses)
        print(' [teacher_loss=%.5f,' % avg_teacher_loss, end='')
        print(' diversity_loss=%.5f]' % avg_diversity_loss, end='')
        print()
