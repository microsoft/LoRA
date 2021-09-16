#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os, sys
import glob
import random
from collections import Counter, OrderedDict
import numpy as np
import torch
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, eval_len=None, device='cpu', world_size=1, rank=0):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.data = data
        self.bsz = bsz
        self.world_size = world_size
        self.rank = rank
        self.bptt = bptt # tgt_len
        # existing len.
        self.eval_len = bptt if eval_len is None else eval_len

        self.device = device
        
        self.global_bsz = bsz * world_size
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = len(data) // self.global_bsz # bsz

        self.split_data = torch.tensor(
            data[rank * self.n_step * bsz : (rank + 1) * self.n_step * bsz], 
            dtype=torch.long, device=self.device
        )  # data.view(-1)

        self.split_data = self.split_data.view(bsz, -1) 

    def __iter__(self):
        return self.get_fixlen_iter()

    def get_batch(self, i, bptt, eval_len):
        beg_idx = i
        end_idx = i + bptt # seq_len
        
        # batch_size, lengh;
        _input = self.split_data[:, beg_idx : end_idx].contiguous()
        _target = self.split_data[:, beg_idx+1 : end_idx+1].contiguous()

        _msk = torch.cat(
            [
                torch.zeros(bptt-eval_len, dtype=torch.float, device=self.device), 
                torch.ones(eval_len, dtype=torch.float, device=self.device)
            ]
        )
        _msk = _msk.unsqueeze(0).expand_as(_input) # .unsqueeze(-1) # length, 1; 
        return _input, _target, _msk

    def get_fixlen_iter(self, start=0):
        self.data_len = self.split_data.size(1)
        _eval_cursor = 0
        for i in range(start, self.data_len - 1, self.eval_len):
            bptt = min(self.bptt, self.data_len - i - 1)
            _end_idx = i + bptt
            yield self.get_batch(i, bptt, _end_idx - _eval_cursor)
            _eval_cursor = _end_idx 


class Corpus(object):
    def __init__(self, path):
        self.path = path
        self.num_words = 0        
        self.tokens = []
        with open(self.path, "r") as reader:
            for line in reader:
                items = json.loads(line.strip())
                book = items['book']
                tokens = items['tokens']
                num_words = items['num_words']

                self.num_words += num_words
                self.tokens.extend(tokens)


class BinLMOrderedIterator(object):
    def __init__(self, corpus, bsz, bptt, eval_len=None, device='cpu', world_size=1, rank=0):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.corpus = corpus
        self.bsz = bsz
        self.world_size = world_size
        self.rank = rank
        self.bptt = bptt # tgt_len
        # existing len.
        self.eval_len = bptt if eval_len is None else eval_len
        self.device = device
        self.global_bsz = bsz * world_size
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = corpus.length // self.global_bsz # bsz

        self.offset = [(rank * bsz + _b) * self.n_step  for _b in range(bsz)]

    def __iter__(self):
        return self.get_fixlen_iter()

    def get_batch(self, i, bptt, eval_len):
        # batch_size, lengh;
        _inputs = []
        _targets = []
        for _b in range(0, self.bsz):
            _input = self.corpus.get_tokens(self.offset[_b] + i, bptt)
            _target = self.corpus.get_tokens(self.offset[_b] + i + 1, bptt)

            _inputs.append(_input)
            _targets.append(_target)

        _input = torch.tensor(_inputs, dtype=torch.int64, device=self.device).contiguous()
        _target = torch.tensor(_targets, dtype=torch.int64, device=self.device).contiguous()

        _msk = torch.cat(
            [
                torch.zeros(bptt-eval_len, dtype=torch.float, device=self.device), 
                torch.ones(eval_len, dtype=torch.float, device=self.device)
            ]
        )
        _msk = _msk.unsqueeze(0).expand_as(_input) # .unsqueeze(-1) # length, 1; 
        return _input, _target, _msk

    def get_fixlen_iter(self, start=0):
        #self.data_len = self.split_data.size(1)
        _eval_cursor = 0
        for i in range(start, self.n_step - 1, self.eval_len):
            bptt = min(self.bptt, self.n_step - i - 1)
            _end_idx = i + bptt
            yield self.get_batch(i, bptt, _end_idx - _eval_cursor)
            _eval_cursor = _end_idx 


class BinCorpus(object):
    def __init__(self, path):
        self.path = path

        self.book_token_span = []
        self.book_token_span.append(0)
        tokens_sum = 0
        self.num_words = 0    

        with open(path+'.info', 'r') as info_reader:
            for line in info_reader:
                items = json.loads(line.strip())
                book = items['book']
                num_tokens = items['num_subtokens']
                num_words = items['num_words']

                tokens_sum += num_tokens
                self.book_token_span.append(tokens_sum)
                self.num_words += num_words

        self.length = self.book_token_span[-1]
        self.bin_reader = open(path+'.bin', 'rb')

    def get_tokens(self, offset, count):
        INT64_SIZE = 8
        self.bin_reader.seek(offset * INT64_SIZE)
        x = np.fromfile(self.bin_reader, count=count, dtype=np.int)
        return x


def get_lm_corpus(data):
    print('Producing dataset {}...'.format(data))
    corpus = Corpus(data)
    return corpus


def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len


class FT_Dataset(Dataset):
    def __init__(self, ft_file, batch_size, max_seq_length, 
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0, 
                 prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_file = ft_file
        self.ft_samples = self.read_ft_file(ft_file)
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.rng = random.Random(911)
        self.joint_lm = joint_lm

        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size) 

        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_batches * self.batch_size
        
    def __getitem__(self, item):
        if(item >= self.num_examples):
            item = self.rng.randint(0, self.num_examples - 1)

        example = self.ft_samples[item]
        context = example[0]
        completion = example[1]

        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)] 
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)] 

        conditions = pretokens + context + intokens 
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1)

        if not self.joint_lm:
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            _msk = [1.0] * (_input_len - 1)

        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        
        _query, _query_len = padding_tokens(
            conditions, self.max_seq_length, 0, -1, 
            max_context_length = self.max_seq_length - self.max_eval_length
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)

        output["input"] = torch.tensor(_input, dtype=torch.long) 
        output["target"] = torch.tensor(_target, dtype=torch.long) 

        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output

    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        return ft_samples
