from servers.serverBase import ServerBase
from clients.clientQFedAvg import ClientQFedAvg


class ServerQFedAvg(ServerBase):
    def __init__(self, args):
        super(ServerQFedAvg, self).__init__(args)
        self.clients = self._init_clients()
        self.all_train_data_num = sum([len(self.clients[client_id].data_train) for client_id in self.clients])
        self.num_batches_per_epoch = self.all_train_data_num // (
                self.args.num_clients * self.args.batch_size_local_training)
        print(f'number of total training data: {self.all_train_data_num}')
        print(f'number of batches per epoch: {self.num_batches_per_epoch}')
        self.id2d = {}
        self.id2h = {}

    def _init_clients(self):
        clients = {}
        for client_id in self.id2data:
            data = self.id2data[client_id]
            clients[client_id] = ClientQFedAvg(self.args, client_id, data)

        return clients

    def run(self):
        for r in range(self.args.num_rounds):
            print('*' * 52, end=' ')
            print(f'Round: %3d/%3d' % (r + 1, self.args.num_rounds), end=' ')
            print('*' * 52)
            active_clients = self.random_state.choice(
                a=self.args.num_clients,
                size=self.args.num_active_clients,
                replace=False
            )
            for client_id in active_clients:
                client = self.clients[client_id]
                delta, h = client.train_with_fairness(weight=self.latest_weight, q=self.args.q)
                self.id2d[client_id] = delta
                self.id2h[client_id] = h
            # Server update
            self.aggregation(active_clients)
            # Aggregate local results
            self.aggregate_local_results('latest')
        self._print_results()

    def aggregation(self, active_clients):
        # Aggregate latest weight.
        sum_h = 0
        for client_id in active_clients:
            h = self.id2h[client_id]
            sum_h += h
        for key in self.latest_weight:
            sum_d = 0
            for client_id in active_clients:
                d = self.id2d[client_id][key]
                sum_d += d
            self.latest_weight[key] -= sum_d / sum_h
