# Originally forked from: https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/listops_data.py 
import os

import numpy
import random

import torch
from torch.utils.data import Dataset


# From https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class ImdbDataset(Dataset):

    def __init__(self, pos_data_file, neg_data_file, pad_idx=50256,
                 max_seq_length=128, device='cuda'):
        # No need for vocab, each token is already an ID

        self.max_seq_length = max_seq_length  # set by text_to_data

        self.labels = [1, 0]  # positive, negative

        self.data = self.text_to_data(
            pos_data_file, neg_data_file, pad_idx, device)

        self.data_size = len(self.data)

    def __len__(self):  # To be used by PyTorch Dataloader.
        return self.data_size

    def __getitem__(self, index):  # To be used by PyTorch Dataloader.
        return self.data[index]

    def text_to_data(self, pos_data_txt, neg_data_txt, pad_idx, device='cuda'):
        # All sequences are padded to the length of the longest sequence
        # of the respective file (lazy padding).

        assert os.path.exists(pos_data_txt)
        assert os.path.exists(neg_data_txt)

        # Check the max length
        min_len = numpy.inf
        max_len = 0
        with open(pos_data_txt, 'r') as text:
            for line in text:
                tokens = line.split()
                length = len(tokens)

                if max_len < length:
                    max_len = length

                if min_len > length:
                    min_len = length

        with open(neg_data_txt, 'r') as text:
            for line in text:
                tokens = line.split()
                length = len(tokens)

                if max_len < length:
                    max_len = length

                if min_len > length:
                    min_len = length

        print(f'mimimum length: {min_len}')
        print(f'maximum length: {max_len}')

        # Construct data
        input_data_list = []
        target_data_list = []
        # length_data_list = []

        count_padded_seq = 0
        data_file_list = [pos_data_txt, neg_data_txt]

        for id in range(2):
            print(f"Loading data file from: {data_file_list[id]}")
            with open(data_file_list[id], 'r') as text:
                for line in text:
                    tokens = line.split()
                    tokens = list(map(int, tokens))

                    seq_len = len(tokens)
                    # input seq
                    var_seq = torch.tensor(
                        tokens, device=device, dtype=torch.int64)
                    if seq_len < self.max_seq_length:
                        count_padded_seq += 1
                        # padding
                        new_seq = var_seq.data.new(
                            self.max_seq_length).fill_(pad_idx)
                        new_seq[:seq_len] = var_seq
                    else:
                        new_seq = var_seq[:self.max_seq_length]
                    input_data_list.append(new_seq)

                    # target, for positives
                    tgt_val = torch.tensor(
                        self.labels[id], device=device, dtype=torch.int64)
                    target_data_list.append(tgt_val)

        # src_file and tgt_file are assumed to be aligned.
        assert len(input_data_list) == len(target_data_list)
        print(f"Number of padded sequences: {count_padded_seq}")

        data_as_list = []
        for i in range(len(input_data_list)):
            data_as_list.append(
                (input_data_list[i], target_data_list[i]))

        return data_as_list
