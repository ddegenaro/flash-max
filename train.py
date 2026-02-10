import os
import json
import shutil
import argparse
from time import time
from inspect import signature

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_sampler import random_data, grid_data
from function import u, c
try:
    from function import p_array
except:
    pass
from wave_equation import Wave, WaveSimplified
from maxwell_equation import Maxwell
from utils import DEVICE

torch.manual_seed(42)

def train_epoch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    loss_fn: torch.nn.MSELoss,
    epoch: int,
    epochs: int,
    log_freq: int,
    verbose: bool
) -> float:
    
    es = epoch + 1
    print(f'Epoch {es}/{epochs}...', end='\r')
    model.train()
    total_loss_train = 0.

    if verbose:
        enum_train_loader = enumerate(tqdm(train_loader))
        enum_val_loader = enumerate(tqdm(val_loader))
    else:
        enum_train_loader = enumerate(train_loader)
        enum_val_loader = enumerate(val_loader)

    training_start = time()
    for i, (inputs, targets) in enum_train_loader:
        
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs[:, 0].unsqueeze(1), inputs[:, 1:])
        mse_loss = loss_fn(outputs, targets.unsqueeze(1))
        mse_loss.backward()
        optimizer.step()

        lv = mse_loss.item()
        total_loss_train += lv
        s = i + 1
        if s % log_freq == 0:
            al = total_loss_train / s
            print(f'Epoch: {es:02d} - Step: {s:04d} - Loss: {lv:.4f} - Avg: {al:.4f}')
    training_time = time() - training_start

    model.eval()
    total_loss_val = 0.
    num_examples = 0

    val_start = time()
    with torch.no_grad():
        for i, (inputs, targets) in enum_val_loader:
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs[:, 0].unsqueeze(1), inputs[:, 1:])
            loss = loss_fn(outputs, targets.unsqueeze(1))

            total_loss_val += loss.item() * targets.shape[0]
            
            num_examples += targets.shape[0]
    val_time = time() - val_start

    return (total_loss_val / num_examples), training_time, val_time

def main(args):

    print(f'Beginning training.')
    
    if args.data_from_lattice:
        data_fn = grid_data
    else:
        data_fn = random_data
        
    train_inputs, train_targets = data_fn(
        num_samples=args.n_train,
        spatial_dim=args.input_dim,
        mins=args.train_mins,
        maxes=args.train_maxes,
        f=u,
        noise_scale=args.noise_scale,
        restrict_time=args.restrict_time
    )
    
    mu = train_inputs.mean(dim=0)
    sigma = train_inputs.std(dim=0)
    
    if args.norm_inputs:
        train_inputs = (train_inputs - mu.unsqueeze(0)) / sigma.unsqueeze(0)
        
    if args.batch_size > args.n_train:
        batch_size = args.n_train
    else:
        batch_size = args.batch_size

    train_loader = DataLoader(
        TensorDataset(
            train_inputs, train_targets
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_inputs, val_targets = data_fn(
        num_samples=args.n_val,
        spatial_dim=args.input_dim,
        mins=args.val_mins,
        maxes=args.val_maxes,
        f=u,
        noise_scale=args.noise_scale
    ) # restrict_time is False by default, desirable here.
    
    mean_val_f = (val_targets ** 2).mean()
    
    if args.norm_inputs:
        val_inputs = (val_inputs - mu.unsqueeze(0)) / sigma.unsqueeze(0)

    val_loader = DataLoader(
        TensorDataset(
            val_inputs, val_targets
        ),
        batch_size=batch_size,
        shuffle=True
    )

    if args.use_simplified:
        model_class = WaveSimplified
    else:
        model_class = Wave
    
    if args.maxwell:
        model = Maxwell(
            width=args.width,
            c=args.c,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            # dropout_val=args.do
            p_array=p_array
        ).to(DEVICE)
    else:
        model = model_class(
            width=args.width,
            c=args.c,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            # dropout_val=args.do
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    loss_fn = torch.nn.MSELoss()
    
    os.makedirs('experiments', exist_ok=True)
    experiments_list = os.listdir('experiments')
    experiments_list.remove('.DS_Store')
    this_experiment = str(1 + max(
        [int(d) for d in experiments_list] + [0]
    ))
    os.makedirs(os.path.join('experiments', this_experiment))

    hparams = vars(args)
    hparams['param_count'] = sum([p.numel() for p in model.parameters()])
    hparams['mu'] = mu.tolist()
    hparams['sigma'] = sigma.tolist()
    json.dump(
        hparams,
        open(
            os.path.join('experiments', this_experiment, 'hparams.json'),
            'w+', encoding='utf-8'
        ),
        indent=4
    )
    
    shutil.copyfile('function.py',  os.path.join('experiments', this_experiment, f'f.py'))

    with open(
        os.path.join('experiments', this_experiment, 'log.tsv'),
        'w+', encoding='utf-8'
    ) as fp:
        fp.write('epoch\tmse\ttraining_time\tval_time\trel_l2_error\n')
            
    best_loss = torch.inf
    k = args.k
    last_k_losses = []
    
    print(f'(Exp. {this_experiment}) Training {model_class.__name__} on {DEVICE}...')
    for key, value in hparams.items():
        print(f'\t{key}: {value}')

    with open(
            os.path.join('experiments', this_experiment, 'log.tsv'),
            'a', encoding='utf-8'
        ) as fp:
            
        for epoch in range(args.max_epochs):
            mse, training_time, val_time = train_epoch(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                epochs=args.max_epochs,
                log_freq=args.log_freq,
                verbose=args.verbose
            )
            
            rel_l2_error = mse / mean_val_f
            
            fp.write(
                f'{epoch+1}\t{mse}\t{training_time}\t{val_time}\t{rel_l2_error}\n'
            )

            if mse < best_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join('experiments', this_experiment, 'model.pth')
                )
                
                best_loss = mse

                if mse < args.tol: # must improve to break
                    print(f'Stopping early at epoch {epoch+1} (mse {mse} < args.tol {args.tol}).')
                    break
                
                if len(last_k_losses) == k:
                    last_k_losses_tensor = torch.tensor(last_k_losses)
                    if torch.allclose(
                        last_k_losses_tensor,
                        last_k_losses_tensor.mean(),
                        atol=args.atol
                    ):
                        print(f'Stopping early at epoch {epoch+1} (last k losses: {last_k_losses}).')
                        break
                
            if len(last_k_losses) == k:
                del last_k_losses[0]
            
            last_k_losses.append(mse)

    print('Done.')
    print(f'Results can be found at {os.path.join('experiments', this_experiment)}.')

def validate(args):
    assert args.n_train > 0
    assert args.n_val > 0
    assert args.batch_size > 0
    assert args.width > 0
    assert args.c > 0
    assert args.input_dim > 0
    assert args.output_dim > 0
    assert args.max_epochs > 0
    assert args.lr > 0
    assert args.wd >= 0
    assert args.log_freq > 0
    assert args.tol > 0
    for i in range(len(args.val_mins)):
        assert args.train_mins[i] < args.train_maxes[i]
        assert args.val_mins[i] < args.val_maxes[i]

if __name__ == "__main__":

    # inferring dimension from f
    input_dim = len(signature(u).parameters) - 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        help='Activation function to use. Default Tanh.'
    )
    parser.add_argument(
        '--c',
        type=float,
        default=c,
        help='Speed of propagation in the medium. Default inferred from function.py.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=input_dim,
        help='Number of spatial dimensions to be input. Default inferred from function.py.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=1,
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
        '--tol',
        type=float,
        default=1e-8,
        help='Stop training if MSE is less than this tolerance.'
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-8,
        help='Stop training if MSE is not changing by more than this tolerance.'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Patience for atol.'
    )
    parser.add_argument(
        '--use_simplified',
        action='store_true',
        default=False,
        help='Use simplified architecture.'
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
        '--train_mins',
        type=float,
        nargs='+',
        default=[0.] * (1 + input_dim),
        help='Minimum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--train_maxes',
        type=float,
        nargs='+',
        default=[1.] * (1 + input_dim),
        help='Maximum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_mins',
        type=float,
        nargs='+',
        default=[0.] + ([0.25] * input_dim),
        help='Minimum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_maxes',
        type=float,
        nargs='+',
        default=[1.] + ([0.75] * input_dim),
        help='Maximum value for each dimension. First dimension interpreted as time.'
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
        '--maxwell',
        action='store_true',
        default=False,
        help='Solve Maxwell\'s instead of Wave.'
    )
    
    # ABOVE GENERALLY FIXED
    
    parser.add_argument(
        '--n_train',
        type=int,
        default=1_000,
        help='Number of samples to generate for training. Default 1,000.'
    )
    parser.add_argument(
        '--noise_scale',
        type=float,
        default=0,
        help='Standard deviation of the noise to be added.'
    )
    
    # ABOVE FIXED PER EXPERIMENT
    
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-2,
        help='Learning rate.'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=5e-3,
        help='Weight decay.'
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
    # parser.add_argument(
    #     '--do',
    #     type=float,
    #     default=0.0,
    #     help='Dropout probability.'
    # )

    args = parser.parse_args()
    validate(args)
    main(args)