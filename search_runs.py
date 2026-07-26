import json, os, glob
import pandas as pd

val_err_threshold = 0.01

search_terms = {
    'soln': 3,
    'add_bc': True,
    'do_masking': True,
    "train_mins": [
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "train_maxes": [
        1.0,
        1.0,
        1.0,
        1.0
    ],
    "val_mins": [
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "val_maxes": [
        1.0,
        1.0,
        1.0,
        1.0
    ],
}

results = []

for file in glob.glob(os.path.join('experiments', '*', 'hparams.json')):
    hparams = json.load(open(file))
    include = True
    for key, val in search_terms.items():
        if key not in hparams or hparams[key] != val:
            include = False
            break
    if include:
        results.append(file.split(os.path.sep)[-2])
        
results = sorted(results)

for result in results:
    
    try:
        log = pd.read_csv(os.path.join('experiments', result, 'log.tsv'), sep='\t')
    except:
        print(f'Missing log {result}')
        continue
    
    rows = log[log['rel_l2_error'] < val_err_threshold]
    
    if len(rows) > 0:
        # breakpoint()
        first_epoch = log[log['rel_l2_error'] < val_err_threshold].iloc[0]['epoch'].item()
        training_time = log[log['epoch'] <= first_epoch]['training_time'].sum().item()
    
        print(result, training_time)