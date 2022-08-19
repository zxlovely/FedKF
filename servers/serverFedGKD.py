from servers.serverBase import ServerBase
from clients.clientFedGKD import ClientFedGKD


class ServerFedGKD(ServerBase):
    def __init__(self, args):
        super(ServerFedGKD, self).__init__(args)
        self.clients = self._init_clients()
        self.all_train_data_num = sum([len(self.clients[client_id].data_train) for client_id in self.clients])
        self.num_batches_per_epoch = self.all_train_data_num // (
                self.args.num_clients * self.args.batch_size_local_training)
        print(f'number of total training data: {self.all_train_data_num}')
        print(f'number of batches per epoch: {self.num_batches_per_epoch}')
        self.teacher_weights = []

    def _init_clients(self):
        clients = {}
        for client_id in self.id2data:
            data = self.id2data[client_id]
            clients[client_id] = ClientFedGKD(self.args, client_id, data)

        return clients

    def run(self):
        buffer_size = 5
        teacher_weight = None
        for r in range(self.args.num_rounds):
            print('*' * 52, end=' ')
            print(f'Round: %3d/%3d' % (r + 1, self.args.num_rounds), end=' ')
            print('*' * 52)
            active_clients = self.random_state.choice(
                a=self.args.num_clients,
                size=self.args.num_active_clients,
                replace=False
            )
            if len(self.teacher_weights) >= buffer_size:
                teacher_weight = self._avg_weights(self.teacher_weights[-buffer_size:], [1 for _ in range(buffer_size)])
            for client_id in active_clients:
                client = self.clients[client_id]
                if teacher_weight is None:
                    self.id2weight[client_id] = client.base_train2(weight=self.latest_weight,
                                                                   num_batches_per_epoch=self.num_batches_per_epoch)
                else:
                    self.id2weight[client_id] = client.train_with_distill(teacher_weight=teacher_weight,
                                                                          student_weight=self.latest_weight,
                                                                          num_batches_per_epoch=self.num_batches_per_epoch,
                                                                          gamma=self.args.gkd_gamma)
                self.id2amount[client_id] = len(client.data_train)
            # Server update
            self.aggregation(active_clients)
            # Aggregate local results
            self.aggregate_local_results('latest')
            # self.aggregate_local_results('global')
            self.teacher_weights.append(self.latest_weight)
        self._print_results()