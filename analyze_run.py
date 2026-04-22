import os
import json
import argparse
import re

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    '--start', '-s',
    type=int,
    default=max([int(x) for x in os.listdir('experiments') if not x.startswith('.')]),
    help='Min experiment number to analyze.'
)
parser.add_argument(
    '--end', '-e',
    type=int,
    default=max([int(x) for x in os.listdir('experiments') if not x.startswith('.')]),
    help='Max experiment number to analyze.'
)
parser.add_argument(
    '--n_train', '-t',
    type=int,
    default=None,
    help='Value of n_train to restrict to.'
)
parser.add_argument(
    '--noise_scale', '-n',
    type=float,
    default=None,
    help='Value of noise_scale to restrict to.'
)
parser.add_argument(
    '--soln',
    type=int,
    default=None,
    help='Solution number to restrict to.'
)
parser.add_argument(
    '--just', '-j',
    type=int,
    default=None,
    help='Just this experiment.'
)

args = parser.parse_args()

start = args.start
end = args.end
just = args.just

if end is None:
    end = start
    
if just is not None:
    start = just
    end = just

for i in range(start, end+1):

    exp_dir = os.path.join('experiments', f'{i}')
    
    if not os.path.exists(exp_dir):
        print(f'no such experiment: {i}')
        continue

    data = json.load(open(os.path.join(exp_dir, 'hparams.json')))
    
    df = pd.read_csv(
        os.path.join('experiments', f'{i}', 'log.tsv'),
        sep = '\t'
    )
    
    if len(df) < 200:
        continue
    
    print(f'Processing experiment {i}...')
    print(f'soln:         {data['soln']}')
    print(f'seed:         {data['seed']}')
    print(f'add_bc:       {data['add_bc']}')
    print(f'n_train:      {data['n_train']}')
    print(f'width:        {data['width']}')
    # print(f'noise scale:   {data['noise_scale']}')
    print(f'lr:           {data['lr']}')
    print(f'wd:           {data['wd']}')
    print(f'T_max_factor: {data['T_max_factor']}')
    print(f'eta_min:      {data['eta_min']}')
    # print(f'activation:    {data['activation']}')
    # print(f'init:          {data['init']}')
    # print(f'training:      {data['train_mins']} -> {data['train_maxes']}')
    # print(f'validation:    {data['val_mins']} -> {data['val_maxes']}')
    # print(f'add_bc:        {data['add_bc']}')

    

    df['rounded_error'] = df['rel_l2_error'].round(3)

    try:
        first_epoch = int(df[df['rel_l2_error'] < 0.01].iloc[0]['epoch'])

        first_time = df[df['epoch'] <= first_epoch]['training_time'].sum().item()

        print(f'Training time to first error <1%: {first_time:.1f}')
    except:
        print('Didn\'t reach error <1%.')
        
    min_rounded_error = df['rounded_error'].min().item()
        
    min_epoch = int(df[df['rounded_error'] == min_rounded_error].iloc[0]['epoch'])

    min_time = df[df['epoch'] <= min_epoch]['training_time'].sum().item()

    print(f'Training time to min error of {100*min_rounded_error:.1f}%: {min_time:.1f}')
    
    print('-' * 50)