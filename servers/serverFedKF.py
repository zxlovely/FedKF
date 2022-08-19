import copy

from servers.serverBase import ServerBase
from clients.clientFedKF import ClientFedKF
from models.utils import get_generator


class ServerFedKF(ServerBase):
    def __init__(self, args):
        super(ServerFedKF, self).__init__(args)
        self.generator = get_generator(dataset=args.dataset)
        self.generator.to(args.device)
        self.clients = self._init_clients()
        self.all_train_data_num = sum([len(self.clients[client_id].data_train) for client_id in self.clients])
        self.num_batches_per_epoch = self.all_train_data_num // (
                self.args.num_clients * self.args.batch_size_local_training)
        print(f'number of total training data: {self.all_train_data_num}')
        print(f'number of batches per epoch: {self.num_batches_per_epoch}')

    def _init_clients(self):
        clients = {}
        for client_id in self.id2data:
            data = self.id2data[client_id]
            clients[client_id] = ClientFedKF(self.args, client_id, data)
            clients[client_id].generator = copy.deepcopy(self.generator)

        return clients

    def run(self):
        for global_round in range(self.args.num_rounds):
            print('*' * 52, end=' ')
            print(f'Round: %3d/%3d' % (global_round + 1, self.args.num_rounds), end=' ')
            print('*' * 52)
            # Clients local training
            active_clients = self.random_state.choice(
                a=self.args.num_clients,
                size=self.args.num_active_clients,
                replace=False
            )
            for client_id in active_clients:
                client = self.clients[client_id]
                if global_round == 0:
                    self.id2weight[client_id] = client.base_train2(weight=self.latest_weight,
                                                                   num_batches_per_epoch=self.num_batches_per_epoch)
                else:
                    if self.args.algorithm == 'FedKF':
                        teacher_weight = self.global_weight
                    else:
                        teacher_weight = self.latest_weight
                    self.id2weight[client_id] = client.train_with_distill(
                        teacher_weight=teacher_weight,
                        student_weight=self.latest_weight,
                        oh=self.args.kf_oh,
                        a=self.args.kf_a,
                        gamma=self.args.kf_gamma,
                        num_batches_per_epoch=self.num_batches_per_epoch
                    )

                self.id2amount[client_id] = len(client.data_train)
            self.aggregation(active_clients)
            # Aggregate local results
            self.aggregate_local_results('latest')
            self.aggregate_local_results('global')
        self._print_results('latest')
        self._print_results('global')
