import os
import glob
import json
import shutil

runs = list(range(1522, 1582))

for run in runs:
    
    hparams = json.load(open(os.path.join('experiments', f'{run}', 'hparams.json')))
    
    soln=hparams['soln']
    seed=hparams['seed']
    
    for gif in glob.glob(os.path.join('experiments', f'{run}', '*.gif')):
        os.makedirs(
            os.path.join('flash_max_gifs', f'soln_{soln}', f'seed_{seed}'),
            exist_ok=True
        )
        
        fname = gif.split(os.path.sep)[-1]
        
        shutil.copy(
            gif,
            os.path.join('flash_max_gifs', f'soln_{soln}', f'seed_{seed}', fname)
        )