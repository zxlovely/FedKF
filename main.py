import argparse
import torch

from servers.serverFedAvg import ServerFedAvg
from servers.serverFedProx import ServerFedProx
from servers.serverFedGKD import ServerFedGKD
from servers.serverFedKF import ServerFedKF
from servers.serverQFedAvg import ServerQFedAvg
from servers.serverFedGen import ServerFedGen


def args_parser():
    parser = argparse.ArgumentParser()

    # Base Hyper-parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['EMNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--algorithm', type=str, default='FedKF',
                        choices=['FedAvg', 'FedProx', 'FedGen', 'FedGKD', 'qFedAvg', 'FedKF', 'FedKF-'])
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_active_clients', type=int, default=4)
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    parser.add_argument('--prox_mu', type=float, default=0.0001)
    parser.add_argument('--kf_oh', type=float, default=0.1)
    parser.add_argument('--kf_a', type=float, default=0.1)
    parser.add_argument('--kf_gamma', type=float, default=1.0)
    parser.add_argument('--gkd_gamma', type=float, default=0.2)
    parser.add_argument('--q', type=float, default=0.0001)

    args = parser.parse_args()

    return args


def main():
    args = args_parser()
    if args.algorithm == 'FedAvg':
        server = ServerFedAvg(args)
    elif args.algorithm == 'FedProx':
        server = ServerFedProx(args)
    elif args.algorithm == 'FedGen':
        server = ServerFedGen(args)
    elif args.algorithm == 'FedGKD':
        server = ServerFedGKD(args)
    elif args.algorithm == 'qFedAvg':
        server = ServerQFedAvg(args)
    elif args.algorithm == 'FedKF' or 'FedKF-':
        server = ServerFedKF(args)
    else:
        raise KeyError(f'This code have not covered {args.algorithm}!')
    server.run()


if __name__ == '__main__':
    torch.manual_seed(123456)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123456)
    # torch.use_deterministic_algorithms = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    main()