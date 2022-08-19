import copy
import numpy as np
from torch import nn

from models.utils import get_model
from data.utils import load_dataset, classify_indices_by_class, Indices2Dataset, show_clients_data_distribution
from data.dirichlet import dirichlet_partition_class
from data.utils import CONFIGS


class ServerBase(object):
    def __init__(self, args):
        assert args.dataset in CONFIGS
        self.args = args
        self.num_classes = CONFIGS[args.dataset]['num_classes']
        self.random_state = np.random.RandomState(args.seed)
        self.model = get_model(dataset=args.dataset)
        self.model.to(args.device)
        self._print_hp()
        self.id2data = self._prepare_data()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.latest_weight = copy.deepcopy(self.model.state_dict())
        self.global_weight = copy.deepcopy(self.model.state_dict())
        self.id2weight = {}
        self.id2amount = {}
        self.clients = {}
        self.results_latest = []
        self.results_global = []

    def _print_hp(self):
        print(f'dataset: {self.args.dataset}')
        print(f'model: {self.model.name}')
        print(f'algorithm: {self.args.algorithm}')
        print(f'total clients: {self.args.num_clients}')
        print(f'active clients: {self.args.num_active_clients}')
        print(f'non-iid alpha: {self.args.alpha}')
        print(f'total round: {self.args.num_rounds}')
        print(f'epochs of local training: {self.args.num_epochs_local_training}')
        print(f'batch size of local training: {self.args.batch_size_local_training}')
        print(f'random seed: {self.args.seed}')
        print(f'device: {self.args.device}')

    def _prepare_data(self):
        seed = 123456  # Do not change
        random_state = np.random.RandomState(seed=seed)
        # Load data
        raw_data = load_dataset(dataset=self.args.dataset)
        # Distribute data
        raw_class2indices = classify_indices_by_class(raw_data)
        if CONFIGS[self.args.dataset]['proportion'] != 1:
            class2indices = {}
            for cls in raw_class2indices:
                indices = raw_class2indices[cls]
                random_state.shuffle(indices)
                class2indices[cls] = indices[:int(len(indices) * CONFIGS[self.args.dataset]['proportion'])]
        else:
            class2indices = raw_class2indices
        id2indices = dirichlet_partition_class(
            class2indices=class2indices,
            num_clients=self.args.num_clients,
            alpha=self.args.alpha,
            seed=seed
        )
        id2data = {}
        for client_id in id2indices:
            indices = id2indices[client_id]
            data = Indices2Dataset(data=raw_data, indices=indices)
            id2data[client_id] = data

        show_clients_data_distribution(raw_data=raw_data,
                                       id2indices=id2indices,
                                       num_classes=self.num_classes)

        return id2data

    def _print_results(self, w='latest'):
        assert w == 'latest' or 'global'
        if w == 'latest':
            results = self.results_latest
        else:
            results = self.results_global
        loss_per_round = []
        acc_per_round = []
        var_per_round = []
        min_per_round = []
        for res in results:
            loss_per_round.append(res['avg_loss'])
            acc_per_round.append(res['avg_acc'])
            var_per_round.append(res['var_acc'])
            min_per_round.append(min(res['acc_per_client']))
        str_loss = ''
        for loss in loss_per_round:
            str_loss += '%.4f, ' % loss
        str_acc = ''
        for acc in acc_per_round:
            str_acc += '%.4f, ' % acc
        str_var = ''
        for var in var_per_round:
            str_var += '%.6f, ' % var
        str_min = ''
        for mn in min_per_round:
            str_min += '%.4f, ' % mn
        print()
        print('Loss:')
        print(str_loss[:-2])
        print('Acc:')
        print(str_acc[:-2])
        print('Var:')
        print(str_var[:-2])
        print('Min:')
        print(str_min[:-2])
        results.sort(key=lambda x: x['avg_acc'])
        print()
        for res in results:
            print('[round=%3d]' % res['round'], end='')
            print(' [', end='')
            for idx, acc in enumerate(res['acc_per_client']):
                if idx != 0:
                    print(end=' ')
                print('%.4f' % acc, end='')
            print(']', end='')
            print(' avg_acc: %.4f' % res['avg_acc'], end='')
            print(' var_acc: %.6f' % res['var_acc'])

    '''Average with weights'''

    def _avg_weights(self, weights: list, amounts: list):
        avg_weight = copy.deepcopy(self.model.state_dict())
        for key in avg_weight:
            values = []
            for weight, amount in zip(weights, amounts):
                values.append(weight[key] * amount)
            avg_value = sum(values) / sum(amounts)
            avg_weight[key] = avg_value

        return avg_weight

    '''Simple average'''

    # def _avg_weights(self, weights: list, amounts: list):
    #     avg_weight = copy.deepcopy(self.model.state_dict())
    #     for key in avg_weight:
    #         values = []
    #         for weight, amount in zip(weights, amounts):
    #             values.append(weight[key])
    #         avg_value = sum(values) / len(values)
    #         avg_weight[key] = avg_value
    #
    #     return avg_weight

    def aggregation(self, active_clients):
        latest_weights = []
        latest_amounts = []
        global_weights = []
        global_amounts = []
        for client_id in active_clients:
            weight = self.id2weight[client_id]
            amount = self.id2amount[client_id]
            latest_weights.append(weight)
            latest_amounts.append(amount)
        for client_id in self.id2weight:
            weight = self.id2weight[client_id]
            amount = self.id2amount[client_id]
            global_weights.append(weight)
            global_amounts.append(amount)

        self.latest_weight = self._avg_weights(latest_weights, latest_amounts)
        self.global_weight = self._avg_weights(global_weights, global_amounts)

    def aggregate_local_results(self, w='latest'):
        assert w == 'latest' or 'global'
        if w == 'latest':
            weight = self.latest_weight
            results = self.results_latest
        else:
            weight = self.global_weight
            results = self.results_global
        result = {}
        loss_per_client = []
        samples_per_client = []
        corrects_per_client = []
        for client_id in self.clients:
            client = self.clients[client_id]
            num_samples = len(client.data_test)
            local_loss, local_corrects = client.test(weight)
            samples_per_client.append(num_samples)
            loss_per_client.append(local_loss)
            corrects_per_client.append(local_corrects)
        avg_acc = sum(corrects_per_client) / sum(samples_per_client)
        samples_per_client = np.array(samples_per_client)
        loss_per_client = np.array(loss_per_client)
        corrects_per_client = np.array(corrects_per_client)
        acc_per_client = corrects_per_client / samples_per_client
        weights = samples_per_client / samples_per_client.sum()
        avg_loss = np.average(a=loss_per_client, weights=weights)
        var_loss = loss_per_client.var()
        var_acc = acc_per_client.var()
        result['round'] = len(results) + 1
        result['avg_loss'] = avg_loss
        result['avg_acc'] = avg_acc
        result['var_loss'] = var_loss
        result['var_acc'] = var_acc
        result['acc_per_client'] = sorted(acc_per_client)
        results.append(result)
        print('[', end='')
        for idx, acc in enumerate(result['acc_per_client']):
            if idx != 0:
                print(end=' ')
            print('%.4f' % acc, end='')
        print(']', end='')
        print(' loss=%.5f' % avg_loss, end='')
        print(' acc=%.4f' % avg_acc, end='')
        print(' highest_acc=%.4f' % max(results, key=lambda x: x['avg_acc'])['avg_acc'])
