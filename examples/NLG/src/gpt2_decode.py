#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import numpy as np
import argparse
import os
import sys
import re
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import encoder


parser = argparse.ArgumentParser()

parser.add_argument('--vocab', type=str, default=None, help='vocab path')

parser.add_argument('--sample_file', default=None, type=str, help='ft sample file')
parser.add_argument('--input_file', default=None, type=str, help='ft input file')

parser.add_argument('--output_ref_file', default=None, type=str, help='output reference file')
parser.add_argument('--output_pred_file', default=None, type=str, help='output predicion file')

parser.add_argument('--ref_unique_file', default=None, type=str, help='reference unique id file')

parser.add_argument('--ref_type', default='e2e', choices=['e2e', 'webnlg', 'dart'], 
                    help='e2e style reference type; webnlg style reference type.')
parser.add_argument('--ref_num', default=4, type=int, help='number of references.')


parser.add_argument('--tokenize', action='store_true', help='')
parser.add_argument('--lower', action='store_true', help='')

parser.add_argument('--filter', default='all', choices=['all', 'seen', 'unseen'], 
                    help='for webnlg only, filter categories that are seen during training, unseen, or all')

args = parser.parse_args()


def stardard_tokenize(sent):
    sent = ' '.join(re.split('(\W)', sent))
    sent = sent.split()
    sent = ' '.join(sent)
    return sent


def post_process(sent, is_tokenize, is_lower):
    if is_lower:
        sent = sent.lower()
    if is_tokenize:
        sent = stardard_tokenize(sent)

    return sent


if __name__ == "__main__":
    enc = encoder.get_encoder(args.vocab)

    ref_unique = None

    if args.ref_unique_file is not None:
        print('reading ref_unique_file.')
        ref_unique = []
        uniques = {}
        with open(args.ref_unique_file, 'r') as ref_unique_reader:
            for line in ref_unique_reader:
                _id = int(line.strip())
                ref_unique.append(_id)
                uniques[_id] = 1
        print('len refer dict', len(ref_unique), 'unique', len(uniques))

    with open(args.sample_file, 'r') as sample_reader, \
             open(args.input_file, 'r', encoding='utf8') as input_reader, \
             open(args.output_pred_file, 'w', encoding='utf8') as pred_writer:

        refer_dict = {}
        context_list = []
        line_id = 0
        for line in input_reader:
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']

            context_list.append(context)

            keep = False

            if args.filter == 'all':
                keep = True
            if args.filter == 'seen' and items['cate']: 
                keep = True
            if args.filter == 'unseen' and not items['cate']:
                keep = True

            if ref_unique is None:
                _key = context
            else:
                _key = ref_unique[line_id]

            if keep:
                if not _key in refer_dict:
                    refer_dict[_key] = {}
                    refer_dict[_key]['references'] = []
                refer_dict[_key]['references'].append(completion.split('<|endoftext|>')[0].split('\n\n')[0].strip())

            line_id += 1

        print('unique refer dict', len(refer_dict))

        for line in sample_reader:
            items = json.loads(line.strip())
            _id = items['id']
            _pred_tokens = items['predict']

            if ref_unique is None:
                _key = context_list[_id]
            else:
                _key = ref_unique[_id]

            #assert _key in refer_dict
            if _key in refer_dict:
                refer_dict[_key]['sample'] = enc.decode(_pred_tokens).split('<|endoftext|>')[0].split('\n\n')[0].strip() 

        references = [refer_dict[s]['references'] for s in refer_dict]
        hypothesis = [refer_dict[s]['sample'] for s in refer_dict]

        if args.ref_type == 'e2e':
            with open(args.output_ref_file, 'w', encoding='utf8') as ref_writer:
                for ref, hyp in zip(references, hypothesis):
                    for r in ref:
                        ref_writer.write(post_process(r, args.tokenize, args.lower) + '\n')
                    ref_writer.write('\n')
                    pred_writer.write(post_process(hyp, args.tokenize, args.lower) + '\n')

        elif args.ref_type in ['webnlg', 'dart']:
            if not os.path.exists(args.output_ref_file):
                os.makedirs(args.output_ref_file)

            reference_writers = [
                open(os.path.join(args.output_ref_file, f'reference{fid}'), 'w', encoding='utf8') 
                for fid in range(0, args.ref_num)
            ] 
            
            for ref, hyp in zip(references, hypothesis):
                for fid in range(0, args.ref_num):
                    if len(ref) > fid:
                        reference_writers[fid].write(post_process(ref[fid], args.tokenize, args.lower) + '\n')
                    else:
                        reference_writers[fid].write(post_process(ref[0], args.tokenize, args.lower) + '\n')
                pred_writer.write(post_process(hyp, args.tokenize, args.lower) + '\n')
                    
            for writer in reference_writers:
                writer.close()
