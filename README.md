# Handling Data Heterogeneity in Federated Learning via Knowledge Fusion
It contains implementation of the following algorithms:
* FedAvg
* FedProx
* FedGen
* FedGKD
* q-FedAvg
* FedKF

# Install Requirements
`pip install -r requirements.txt`

# Run Experiments

## FedAvg
`python -u main.py --dataset EMNIST --algorithm FedAvg --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda`

## FedProx
`python -u main.py --dataset EMNIST --algorithm FedProx --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --prox_mu 0.0001`

## FedGen
`python -u main.py --dataset EMNIST --algorithm FedGen --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda`

## FedGKD
`python -u main.py --dataset EMNIST --algorithm FedGKD --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --gkd_gamma 0.001`

## q-FedAvg
`python -u main.py --dataset EMNIST --algorithm qFedAvg --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --q 0.0001`

## FedKF
`python -u main.py --dataset EMNIST --algorithm FedKF --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --kf_gamma 1 --kf_oh 0.1 --kf_a 0.1`

## FedKF-
`python -u main.py --dataset EMNIST --algorithm FedKF- --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --kf_gamma 1 --kf_oh 0.1 --kf_a 0.1`
