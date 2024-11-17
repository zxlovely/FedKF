# [Handling Data Heterogeneity for IoT Devices in Federated Learning: A Knowledge Fusion Approach](https://ieeexplore.ieee.org/abstract/document/10265259)
It contains implementation of the following algorithms:
* FedAvg
* FedProx
* FedGen
* FedGKD
* q-FedAvg
* FedKF

# Install Requirements
```shell
pip install -r requirements.txt
```

# Run Experiments

## FedAvg
```shell
python -u main.py --dataset EMNIST --algorithm FedAvg --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda
```

## FedProx
```shell
python -u main.py --dataset EMNIST --algorithm FedProx --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --prox_mu 0.0001
```

## FedGen
```shell
python -u main.py --dataset EMNIST --algorithm FedGen --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda
```

## FedGKD
```shell
python -u main.py --dataset EMNIST --algorithm FedGKD --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --gkd_gamma 0.001
```

## q-FedAvg
```shell
python -u main.py --dataset EMNIST --algorithm qFedAvg --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --q 0.0001
```

## FedKF
```shell
python -u main.py --dataset EMNIST --algorithm FedKF --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --kf_gamma 1 --kf_oh 0.1 --kf_a 0.1
```

## FedKF-
```shell
python -u main.py --dataset EMNIST --algorithm FedKF- --num_clients 20 --num_active_clients 4 --train_proportion 0.8 --alpha 0.1 --num_rounds 100 --num_epochs_local_training 10 --batch_size_local_training 64 --seed 123456 --device cuda --kf_gamma 1 --kf_oh 0.1 --kf_a 0.1
```

# Citation
If you found this work useful for your research, please cite our paper:
```shell
@article{zhou2023handling,
  title={Handling Data Heterogeneity for IoT Devices in Federated Learning: A Knowledge Fusion Approach},
  author={Zhou, Xu and Lei, Xinyu and Yang, Cong and Shi, Yichun and Zhang, Xiao and Shi, Jingwen},
  publisher={IEEE},
  journal={IEEE Internet of Things Journal},
  year={2024},
  volume={11},
  number={5},
  pages={8090-8104},
  doi={10.1109/JIOT.2023.3319986}
}
```
