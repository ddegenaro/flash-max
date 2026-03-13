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
    '--func', '-f',
    type=str,
    default=None,
    help='Kind of function to restrict to.'
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
        continue

    data = json.load(open(os.path.join(exp_dir, 'hparams.json')))
    
    if args.n_train is not None and data['n_train'] != args.n_train:
        continue
    if args.noise_scale is not None and data['noise_scale'] != args.noise_scale:
        continue

    with open(os.path.join(exp_dir, 'f.py'), 'r') as f:
        if re.findall(r'#\W+R =', f.read()):
            func = 'plane waves'
        else:
            func = 'radial'
    if args.func is not None and func != args.func:
        continue
    
    print(f'Processing experiment {i}...')
    print(f'function:      {func}')
    print(f'training size: {data['n_train']}')
    print(f'noise scale:   {data['noise_scale']}')

    df = pd.read_csv(
        os.path.join('experiments', f'{i}', 'log.tsv'),
        sep = '\t'
    )

    df['rounded_error'] = df['rel_l2_error'].round(3)

    try:
        first_epoch = int(df[df['rounded_error'] < 0.01].iloc[0]['epoch'])

        first_time = df[df['epoch'] <= first_epoch]['training_time'].sum().item()

        print(f'Training time to first error <1%: {first_time:.1f}')
    except:
        print('Didn\'t reach error <1%.')
        
    min_rounded_error = df['rounded_error'].min().item()
        
    min_epoch = int(df[df['rounded_error'] == min_rounded_error].iloc[0]['epoch'])

    min_time = df[df['epoch'] <= min_epoch]['training_time'].sum().item()

    print(f'Training time to min error of {100*min_rounded_error:.1f}%: {min_time:.1f}')
    
    print('-' * 50)