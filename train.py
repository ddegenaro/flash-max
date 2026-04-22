import os
import json
import shutil
import argparse
from time import time
from inspect import signature
import random
from math import sqrt
import sys

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LinearLR
)

from data_sampler import random_data, grid_data
try:
    from function import p_array
except:
    pass
from wave_equation import Wave, WaveSimplified
from maxwell_equation import Maxwell, MaxwellSimple
from utils import get_device, PCNN
from symlog import symlog

def reset_optimizer(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state[p] = {}

def train_epoch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizers: tuple[torch.optim.AdamW],
    schedulers: list[SequentialLR],
    loss_fn: torch.nn.MSELoss,
    epoch: int,
    epochs: int,
    log_freq: int,
    verbose: bool,
    device: str,
    preload_data: bool
) -> float:
    
    # log message
    es = epoch + 1
    print(f'Epoch {es}/{epochs}...', end='\r')
    
    # training mode
    model.train()
    
    # track loss carefully
    total_loss_train = 0.
    num_train_examples = 0

    # tqdm if wanted
    if verbose:
        enum_train_loader = enumerate(tqdm(train_loader))
        enum_val_loader = enumerate(tqdm(val_loader))
    else:
        enum_train_loader = enumerate(train_loader)
        enum_val_loader = enumerate(val_loader)

    start_train = time() # start timing
    for i, (inputs_train, targets_train, mask_train) in enum_train_loader:
        
        # move to GPU if needed
        if not preload_data:
            inputs_train, targets_train, mask_train = inputs_train.to(device), targets_train.to(device), mask_train.to(device)
        
        # zero all optimizers (in case using bi-level)
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        # forward pass
        outputs_train = model(inputs_train[:, 0].unsqueeze(1), inputs_train[:, 1:])
        
        # mask is important when using boundary conditions, else it's all True
        mse_loss_train = loss_fn(outputs_train[mask_train], targets_train[mask_train])
        mse_loss_train.backward()
        if args.max_norm < float('inf'):
            clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            
        for optimizer in optimizers:
            optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

        # loss
        lv_train = mse_loss_train.item()
        # print(f'Epoch: {es:02d} - Loss: {lv:.4f} - Grad Norm: {grad_norm:.4f} - Update Norm: {update_norm:.4f}')
        total_loss_train += lv_train * targets_train.shape[0] # times number of examples in case differing batch sizes
        num_train_examples += targets_train.shape[0]
        s = i + 1
        if s % log_freq == 0: # console log
            al = total_loss_train / s
            print(f'Epoch: {es:02d} - Step: {s:04d} - Train Loss: {lv_train:.4f} - Val Loss: {lv_train:.4f} - Avg: {al:.4f}')
    time_train = time() - start_train # stop timing
    
    # breakpoint()

    # validation, same setup
    model.eval()
    total_loss_val = 0.
    num_val_examples = 0

    start_val = time()
    with torch.no_grad():
        for i, (inputs_val, targets_val, _) in enum_val_loader:
            
            if not preload_data:
                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            
            outputs_val = model(inputs_val[:, 0].unsqueeze(1), inputs_val[:, 1:])
            
            mse_loss_val = loss_fn(outputs_val, targets_val)

            lv_val = mse_loss_val.item()
            total_loss_val += lv_val * targets_val.shape[0]
            num_val_examples += targets_val.shape[0]
            # breakpoint()
    time_val = time() - start_val
    
    # assert num_train_examples == args.n_train
    # assert num_val_examples == args.n_val

    # mean loss per example (training and val) and time needed to do pass
    return (total_loss_train / num_train_examples), (total_loss_val / num_val_examples), time_train, time_val

def main(args):
    
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    print(f'Beginning training.')
    
    if args.soln == 1:
        shutil.copyfile(os.path.join('neurips_functions', 'f1_plane_waves.py'), 'function.py')
    elif args.soln == 2:
        shutil.copyfile(os.path.join('neurips_functions', 'f2_radial_waves.py'), 'function.py')
    elif args.soln == 3:
        shutil.copyfile(os.path.join('neurips_functions', 'f3_hopf_fibration.py'), 'function.py')
    elif args.soln == 4:
        shutil.copyfile(os.path.join('neurips_functions', 'f4_random_sol.py'), 'function.py')
    elif args.soln == 5:
        shutil.copyfile(os.path.join('neurips_functions', 'f5_new_radial_waves.py'), 'function.py')
    
    

    # overwrite function.py with the chosen solution first (existing code) ...

    if 'function' in sys.modules:
        del sys.modules['function']

    from function import u, c
    
    # breakpoint()
    
    # whether to sample random points or use a grid
    if args.data_from_lattice:
        data_fn = grid_data
    else:
        data_fn = random_data

    # training pairs and mask
    # mask can be used to compute loss only over certain parts of the targets
    train_inputs, train_targets, train_mask = data_fn(
        num_samples=args.n_train,
        spatial_dim=args.input_dim,
        mins=args.train_mins,
        maxes=args.train_maxes,
        f=u,
        noise_scale=args.noise_scale,
        restrict_time=args.restrict_time,
        add_bc=args.add_bc
    )
    
    # ensure input-target pairs are matched up on dim 0
    if train_inputs.shape[0] != train_targets.shape[0]:
        train_targets = train_targets.reshape(train_inputs.shape[0], -1)
        train_mask = train_mask.reshape(train_inputs.shape[0], -1)
    
    # compute mean and standard deviation of everything, useful to have
    # also used if normalizing
    mu_inputs = train_inputs.mean(dim=0)
    sigma_inputs = train_inputs.std(dim=0)
    
    mu_targets = train_targets.mean(dim=0)
    sigma_targets = train_targets.std(dim=0)
    
    # normalization if using
    if args.norm_inputs:
        train_inputs[:, 1:] = (train_inputs[:, 1:] - mu_inputs[1:].unsqueeze(0)) / sigma_inputs[1:].unsqueeze(0)
        train_targets = torch.nan_to_num((train_targets - mu_targets.unsqueeze(0)) / sigma_targets.unsqueeze(0))
        assert not train_inputs.isnan().any().item() and not train_targets.isnan().any().item()
    
    # avoid a silly batch size
    if args.batch_size > args.n_train:
        batch_size = args.n_train
    else:
        batch_size = args.batch_size
        
    # retrieve appropriate gpu name if using, else cpu
    device = get_device(args.use_cpu)
        
    if args.preload_data:
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        train_mask = train_mask.to(device)

    # load into loader
    train_loader = DataLoader(
        TensorDataset(
            train_inputs, train_targets, train_mask
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    # same for validation, inputs, targets, mask (mask should be avoided to get accurate error)
    val_inputs, val_targets, val_mask = data_fn(
        num_samples=args.n_val,
        spatial_dim=args.input_dim,
        mins=args.val_mins,
        maxes=args.val_maxes,
        f=u,
        noise_scale=args.noise_scale
    ) # restrict_time and add_bc are False by default, desirable here.
    
    # reshape to assure inputs/targets matched up
    if val_inputs.shape[0] != val_targets.shape[0]:
        val_targets = val_targets.reshape(val_inputs.shape[0], -1)
        val_mask = val_mask.reshape(val_inputs.shape[0], -1)
    
    # normalize if using
    if args.norm_inputs:
        val_inputs[:, 1:] = (val_inputs[:, 1:] - mu_inputs[1:].unsqueeze(0)) / sigma_inputs[1:].unsqueeze(0)
        val_targets = torch.nan_to_num((val_targets - mu_targets.unsqueeze(0)) / sigma_targets.unsqueeze(0))
        assert not val_inputs.isnan().any().item() and not val_targets.isnan().any().item()
    
    # mean function value on validation set for L2 error
    mean_val_f_sq = (val_targets ** 2).mean().item()
    
    if args.preload_data:
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        val_mask = val_mask.to(device)

    # load into validation loader
    val_loader = DataLoader(
        TensorDataset(
            val_inputs, val_targets, val_mask
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # easy way to get model from string name
    model_class = eval(args.model_class)
    
    # init model
    model: PCNN = model_class(
        width=args.width,
        c=c,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        activation=args.activation,
        do=args.do,
        init=args.init,
        gain=args.gain
    ).to(device)

    # bilevel optimization, more-or-less deprecated
    if args.bilevel:
        if args.inner_lr == args.outer_lr:
            print(f'WARNING! Using bilevel with identical learning rates.')
        inner_optimizer = torch.optim.AdamW(
            [param for name, param in model.named_parameters() if 'output' not in name],
            lr=args.inner_lr,
            weight_decay=args.inner_wd
        )
        outer_optimizer = torch.optim.AdamW(
            [param for name, param in model.named_parameters() if 'output' in name],
            lr=args.outer_lr,
            weight_decay=args.outer_wd
        )
        optimizers = [inner_optimizer, outer_optimizer]
    # standard optimization
    else:
        optimizers = [torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2)
        )]
    
    if args.scheduler.lower() == 'cosine':
        schedulers = [
            CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs * args.T_max_factor, # total number of epochs
                eta_min=args.eta_min
            ) for optimizer in optimizers
        ]
    else:
        schedulers = []

    # symlog loss, found to be helpful in some papers with disparate targets
    if args.symlog:
        loss_fn = lambda x, y: torch.nn.MSELoss()(x, symlog(y))
    else:
        loss_fn = torch.nn.MSELoss()
    
    # create a directory for this experiment, next smallest number available
    os.makedirs('experiments', exist_ok=True)
    experiments_list = os.listdir('experiments')
    try:
        experiments_list.remove('.DS_Store')
    except:
        pass
    this_experiment = str(1 + max(
        [int(d) for d in experiments_list] + [0]
    ))
    os.makedirs(os.path.join('experiments', this_experiment))

    # write hyperparameters to file
    hparams = vars(args)
    hparams['c'] = c
    hparams['param_count'] = sum([p.numel() for p in model.parameters()])
    hparams['mu_inputs'] = mu_inputs.tolist()
    hparams['sigma_inputs'] = sigma_inputs.tolist()
    hparams['mu_targets'] = mu_targets.tolist()
    hparams['sigma_targets'] = sigma_targets.tolist()
    hparams['output_dim'] = model.output_dim
    json.dump(
        hparams,
        open(
            os.path.join('experiments', this_experiment, 'hparams.json'),
            'w+', encoding='utf-8'
        ),
        indent=4
    )
    
    # copy true solution function definition for posterity too
    shutil.copyfile('function.py',  os.path.join('experiments', this_experiment, f'f.py'))

    # training/validation logging
    with open(
        os.path.join('experiments', this_experiment, 'log.tsv'),
        'w+', encoding='utf-8'
    ) as fp:
        fp.write('epoch\tmse_train\tmse_val\ttraining_time\tval_time\trel_l2_error\n')
    
    # loss logging
    best_loss = torch.inf
    last_k_train_losses = []
    
    # console message
    print(f'(Exp. {this_experiment}) Training {model_class.__name__} on {device}...')
    for key, value in hparams.items():
        print(f'\t{key}: {value}')

    # keep logging file open
    with open(
            os.path.join('experiments', this_experiment, 'log.tsv'),
            'a', encoding='utf-8'
        ) as fp:
        
        # epoch loop
        for epoch in range(args.max_epochs):
            mse_train, mse_val, training_time, val_time = train_epoch(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                loss_fn=loss_fn,
                epoch=epoch,
                epochs=args.max_epochs,
                log_freq=args.log_freq,
                verbose=args.verbose,
                device=device,
                preload_data=args.preload_data
            )
            
            if mean_val_f_sq == 0:
                mean_val_f_sq = 1e-12
            rel_l2_error = sqrt(mse_val) / sqrt(mean_val_f_sq) # relative L2 error
            
            fp.write( # log immediately, don't wait, it doesn't count towards training time
                f'{epoch+1}\t{mse_train}\t{mse_val}\t{training_time}\t{val_time}\t{rel_l2_error}\n'
            )
            fp.flush()

            # if new best performance
            if mse_val < best_loss:
                torch.save( # save model weights
                    model.state_dict(),
                    os.path.join('experiments', this_experiment, 'model.pth')
                )
                
                best_loss = mse_val
            
            if args.restarts:
                last_k_train_losses.append(mse_train)
                if len(last_k_train_losses) > args.restart_patience:
                    del last_k_train_losses[0]
                    if len(last_k_train_losses) == args.restart_patience:
                        window = last_k_train_losses
                        rel_range = (max(window) - min(window)) / (sum(window) / len(window))
                        if rel_range < args.restart_tolerance / 100:
                            for optimizer in optimizers:
                                reset_optimizer(optimizer)
                            print(f'Reset at epoch {epoch}.')
                            last_k_train_losses = []

    # console message
    print('Done.')
    print(f'Results can be found at {os.path.join('experiments', this_experiment)}.')

def validate(args):
    assert args.n_train > 0
    assert args.n_val > 0
    assert args.batch_size > 0
    assert args.width > 0
    assert args.input_dim > 0
    assert args.output_dim > 0
    assert args.max_epochs > 0
    assert args.lr > 0
    assert args.wd >= 0
    assert args.log_freq > 0
    for i in range(len(args.val_mins)):
        assert args.train_mins[i] <= args.train_maxes[i]
        assert args.val_mins[i] <= args.val_maxes[i]

if __name__ == "__main__":
    
    input_dim = 3

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dim',
        type=int,
        default=input_dim,
        help='Number of spatial dimensions to be input. Default 3.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=6,
        help='Number of wave displacement dimensions to be output. Default 1.'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10_000,
        help='Maximum number of times to show the data to the model.'
    )
    parser.add_argument(
        '--log_freq',
        type=int,
        default=100,
        help='Show a log message every log_freq steps during training.'
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='Log with tqdm.'
    )
    parser.add_argument(
        '--restart_patience',
        type=int,
        default=10,
        help='Patience for restarts.'
    )
    parser.add_argument(
        '--restart_tolerance',
        type=float,
        default=1e-1,
        help='Last {patience} losses must be within _ percent of each other to restart.'
    )
    parser.add_argument(
        '--restarts',
        action='store_true',
        default=False,
        help='Whether to restart the optimizer periodically.'
    )
    parser.add_argument(
        '--model_class',
        type=str,
        default='MaxwellSimple',
        help='Type of model to use.'
    )
    parser.add_argument(
        '--norm_inputs',
        action='store_true',
        default=False,
        help='Normalize inputs via z-scaling.'
    )
    parser.add_argument(
        '--restrict_time',
        action='store_true',
        default=True,
        help='Restrict the time inputs in training to be only the endpoints of the time interval.'
    )
    parser.add_argument(
        '--data_from_lattice',
        action='store_true',
        default=False,
        help='Whether to get data from a lattice (random otherwise).'
    )
    parser.add_argument(
        '--n_val',
        type=int,
        default=10_000,
        help='Number of samples to generate for validation. Default 10,000.'
    )
    parser.add_argument(
        '--init',
        type=str,
        default='xavier normal',
        help='Normal or Kaiming initialization.'
    )
    parser.add_argument(
        '--noise_scale',
        type=float,
        default=0,
        help='Standard deviation of the noise to be added.'
    )
    parser.add_argument(
        '--inner_lr',
        type=float,
        default=0.01,
        help='Learning rate.'
    )
    parser.add_argument(
        '--inner_wd',
        type=float,
        default=5e-6,
        help='Weight decay.'
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=1e-2,
        help='Learning rate.'
    )
    parser.add_argument(
        '--outer_wd',
        type=float,
        default=5e-5,
        help='Weight decay.'
    )
    parser.add_argument(
        '--bilevel',
        action='store_true',
        default=False,
        help='Use different optimizers for the two layers.'
    )
    parser.add_argument(
        '--symlog',
        action='store_true',
        default=False,
        help='Whether to use symlog loss.'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        default=False,
        help='Whether to use early stopping.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed.'
    )
    parser.add_argument(
        '--do',
        type=float,
        default=0.0,
        help='Dropout probability.'
    )
    
    # ABOVE GENERALLY FIXED
    
    parser.add_argument(
        '--n_train',
        type=int,
        default=100_000,
        help='Number of samples to generate for training. Default 1,000.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1_000,
        help='Batch size for training and validation. Default 1_000.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1_000,
        help='Width of the hidden layer of the neural network. Default 1000.'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='tanh',
        help='Activation function to use.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-2,
        help='Learning rate.'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=5e-5,
        help='Weight decay.'
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help="Beta 1 for AdamW."
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help="Beta 2 for AdamW."
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=float('inf'),
        help="Gradient clipping."
    )
    
    # ABOVE OF LESS CONCERN, ABLATIONS ETC.
    
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        default=False,
        help='Whether to train on CPU.'
    )
    parser.add_argument(
        '--preload_data',
        action='store_true',
        default=True,
        help='Move all train and val tensors to device before training.'
    )
    parser.add_argument(
        '--add_bc',
        action='store_true',
        default=True,
        help='Whether to use boundary conditions.'
    )
    parser.add_argument(
        '--train_mins',
        type=float,
        nargs='+',
        default=[0.0] + [0.0] * input_dim,
        help='Minimum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--train_maxes',
        type=float,
        nargs='+',
        default=[1.0] + [1.0] * input_dim,
        help='Maximum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_mins',
        type=float,
        nargs='+',
        default=[0.0] + [0.0] * input_dim,
        help='Minimum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_maxes',
        type=float,
        nargs='+',
        default=[1.0] + [1.0] * input_dim,
        help='Maximum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--soln',
        type=int,
        default=4,
        help='Which solution to work with. (1-4). 0 means use whatever is in function.py as is.'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='',
        help="'cosine' else no scheduler."
    )
    parser.add_argument(
        '--eta_min',
        type=float,
        default=0.0,
        help="eta_min if using a scheduler."
    )
    parser.add_argument(
        '--T_max_factor',
        type=float,
        default=1.0,
        help="Multiplied by max_epochs to get T_max for cosine annealing."
    )
    parser.add_argument(
        '--gain',
        type=float,
        default=1.0,
        help="Gain for Xavier."
    )

    args = parser.parse_args()
    validate(args)
    main(args)
