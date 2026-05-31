import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class BPRDataset(Dataset):
    def __init__(self, buy_interactions, n_items):
        self.buy_interactions = {int(k): set(v) for k, v in buy_interactions.items()}
        self.all_items = set(range(1, n_items + 1))
        self.users = list(self.buy_interactions.keys())
        self.total_samples = []
        for user in self.users:
            for pos_item in self.buy_interactions[user]:
                self.total_samples.append((user, pos_item))

    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        while True:
            neg_item = random.randint(1, len(self.all_items))
            if neg_item not in self.buy_interactions[user]:
                break
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_item, dtype=torch.long))


def convert_edge(edge_list, n_users):
    edge_list[1] += n_users + 1
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list


def load_data(data_dir, dataset, device, batch_size):
    logger.info(f'Load data from {data_dir}/{dataset}')
    dataset_path = os.path.join(data_dir, dataset)

    with open(f'{dataset_path}/statistics.json', 'r') as f:
        statistics = json.load(f)

    n_users, n_items = statistics['n_users'], statistics['n_items']
    bsg_types = statistics['bsg_types']
    hes_types = statistics['tcb_types']   # hesitant: view_not_buy etc.
    conf_types = statistics['tib_types']  # confident: view_buy etc.
    hc_types = hes_types + conf_types

    edge_dict = {}
    for behavior_type in ['ubg'] + bsg_types + hc_types:
        edge_list = np.loadtxt(f'{dataset_path}/{behavior_type}.txt', dtype=int)
        edge_list = torch.from_numpy(edge_list).to(device).T
        edge_dict[behavior_type] = convert_edge(edge_list, n_users)

    with open(f'{dataset_path}/train.json', 'r') as f:
        train_buy = json.load(f)
    with open(f'{dataset_path}/test.json', 'r') as f:
        test_buy = json.load(f)

    train_dataset = BPRDataset(train_buy, n_items)
    test_dataset = BPRDataset(test_buy, n_items)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    data = {
        'edge_dict': edge_dict,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_gt': train_buy,
        'test_gt': test_buy,
        'dataset': dataset,
        'data_dir': dataset_path,
        'device': device,
    }
    data.update(statistics)
    data['hes_types'] = data['tcb_types']
    data['conf_types'] = data['tib_types']
    return data
