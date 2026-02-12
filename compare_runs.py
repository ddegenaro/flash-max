import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    '--start', '-s',
    type=int,
    required=True
)

parser.add_argument(
    '--end', '-e',
    type=int,
    required=True
)

args = parser.parse_args()

s = args.start
e = args.end

data1 = json.load(open(os.path.join('experiments', str(s), 'hparams.json')))

data2 = json.load(open(os.path.join('experiments', str(e), 'hparams.json')))

print('-' * 80)

for key in data1:
    if key not in data2:
        print(f'{key} in {s} but not in {e}.')
    elif data1[key] != data2[key]:
        print(f'{s} has {key}={data1[key]}, but {e} has {key}={data2[key]}.')
        
for key in data2:
    if key not in data1:
        print(f'{key} in {e} but not in {s}.')
        
print('-' * 80)