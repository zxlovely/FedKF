import numpy as np


def dirichlet_partition_class(class2indices: dict, num_clients=20, alpha=1.0, seed=123456):
    random_state = np.random.RandomState(seed)
    num_classes = len(class2indices)
    num_clients_per_time = []
    num_remain_clients = num_clients
    while num_remain_clients > 0:
        if num_remain_clients >= num_classes:
            num_clients_this_time = num_classes
        else:
            num_clients_this_time = num_remain_clients
        num_clients_per_time.append(num_clients_this_time)
        num_remain_clients -= num_clients_this_time

    class2indices_per_time = [{} for _ in range(len(num_clients_per_time))]
    for cla in class2indices:
        indices = class2indices[cla]
        random_state.shuffle(indices)
        proportions = np.array(num_clients_per_time) / np.array(num_clients_per_time).sum()
        num_samples_per_time = (proportions * len(indices)).astype(int)
        num_remain_samples = len(indices) - num_samples_per_time.sum()
        num_samples_per_time[random_state.randint(len(num_samples_per_time))] += num_remain_samples
        partitions = np.split(indices, np.cumsum(num_samples_per_time))
        for idx, partition in enumerate(partitions[:-1]):
            class2indices_per_time[idx][cla] = partition

    indices_batches = []
    for class2indices_this_time, num_clients_this_time in zip(class2indices_per_time, num_clients_per_time):
        indices_per_client = [[] for _ in range(num_clients_this_time)]
        which_client_with_max_proportion = [0 for _ in range(num_clients_this_time)]
        for cla in range(num_classes):
            indices = class2indices_this_time[cla]
            proportions = random_state.dirichlet(alpha=np.ones(num_clients_this_time) * alpha)
            client_with_max_proportion = proportions.argmax()
            while which_client_with_max_proportion[client_with_max_proportion] == 1:
                proportions = random_state.dirichlet(alpha=np.ones(num_clients_this_time) * alpha)
                client_with_max_proportion = proportions.argmax()
            which_client_with_max_proportion[client_with_max_proportion] = 1
            if sum(which_client_with_max_proportion) == num_clients_this_time:
                which_client_with_max_proportion = [0 for _ in range(num_clients_this_time)]
            num_samples_per_client = (proportions * len(indices)).astype(int)
            num_samples_remian = len(indices) - num_samples_per_client.sum()
            num_samples_per_client[random_state.randint(num_clients_this_time)] += num_samples_remian
            partitions = np.split(indices, np.cumsum(num_samples_per_client))
            for idx, partition in enumerate(partitions[:-1]):
                indices_per_client[idx] += partition.tolist()

        indices_batches += indices_per_client
    id2indices = {}
    for client_id, indices in enumerate(indices_batches):
        id2indices[client_id] = indices

    return id2indices
