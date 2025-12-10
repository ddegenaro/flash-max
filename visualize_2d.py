import os
import sys
import json
import argparse
from typing import Any

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

from wave_equation import Wave, WaveSimplified
# from utils import DEVICE
DEVICE = 'cpu'

def func(current_frame: int, total_frames: int) -> Any:
    print(f'Saving frame {current_frame+1}...', end='\r', flush=True)

def main(args):
    
    if args.experiment_num < 1:
        experiment_num = str(max([int(x) for x in os.listdir('experiments')]))
    else:
        experiment_num = str(args.experiment_num)
    print(f'Visualizing experiment {experiment_num}...')
    path = os.path.join('experiments', experiment_num)
    
    sys.path.append(path)
    
    from f import u

    hparams = json.load(
        open(os.path.join(path, 'hparams.json'), 'r', encoding='utf-8')
    )

    if hparams['use_simplified']:
        model_class = WaveSimplified
    else:
        model_class = Wave
    
    model = model_class(
        width=hparams['width'],
        c=hparams['c'],
        input_dim=hparams['input_dim'],
        output_dim=hparams['output_dim'],
        activation = hparams['activation']
    ).to(DEVICE)

    model.load_state_dict(torch.load(
        os.path.join(path, 'model.pth'),
        map_location=DEVICE
    ))

    if args.mins is None:
        mins = hparams['mins']
    else:
        mins = args.mins
    if args.maxes is None:
        maxes = hparams['maxes']
    else:
        maxes = args.maxes
    
    tr = args.temporal_resolution
    sr = args.spatial_resolution
    
    input_cols = [torch.arange(mins[0], maxes[0], (maxes[0] - mins[0]) / tr)] + [
        torch.arange(start, end, (end - start) / sr)
        for start, end in zip(mins[1:], maxes[1:])
    ]
    
    if hparams['norm_inputs']:
        mu = hparams['mu']
        sigma = hparams['sigma']
        
        assert len(mu) == len(sigma) == len(input_cols)
        for i in range(len(mu)):
            input_cols[i] = (input_cols[i] - mu[i]) / sigma[i]

    X, Y = torch.meshgrid(input_cols[1], input_cols[2])
    X_flat = X.flatten().to(DEVICE)
    Y_flat = Y.flatten().to(DEVICE)
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    for plot_true_sol in (True, False):
        
        fname = f'animation_true.{args.ext}' if plot_true_sol else f'animation.{args.ext}'
        fp = os.path.join(path, fname)
        if os.path.exists(fp):
            continue
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        artists = []

        frame_count = len(input_cols[0])
        
        Zs = []
        
        print()
        for i in range(frame_count):
            
            t = input_cols[0][i].expand(X_flat.shape[0]).unsqueeze(1).to(DEVICE)
            pos = torch.stack((X_flat, Y_flat)).transpose(-1, 0)
            
            if plot_true_sol:
                Z = u(
                    t,
                    pos[:, 0].unsqueeze(1),
                    pos[:, 1].unsqueeze(1)
                )
            else:
                Z = model(t, pos)
                
            Zs.append(Z.detach().reshape(X.shape))
            
            print(f'Computing progress: {i+1}/{frame_count}...', end='\r', flush=True)
        
        print()
        for i in range(frame_count):
        
            surf = ax.plot_surface(
                X_np, Y_np, Zs[i].cpu().numpy(),
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False
            )
            
            artists.append([surf])
            
            print(f'Plotting progress: {i+1}/{frame_count}...', end='\r', flush=True)
            
        ani = animation.ArtistAnimation(
            fig=fig,
            artists=artists,
            interval=tr
        )
        
        print()
        ani.save(
            filename=fp,
            writer=args.writer,
            fps=args.fps,
            dpi=args.dpi,
            progress_callback=func
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment_num',
        type=int,
        default=-1,
        help='Assumes there is an experiment with this number in the experiments directory. Default: most recent.'
    )
    parser.add_argument(
        '--temporal_resolution',
        type=float,
        default=100,
        help='Number of time steps to evaluate at per unit time. Default 100.'
    )
    parser.add_argument(
        '--spatial_resolution',
        type=float,
        default=300,
        help='Number of space steps to evaluate at per unit space. Default 100.'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=2,
        help='z-value maximum height for the visualization. Default: 2'
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='gif',
        help='Filetype to save animation to. Default: gif.'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='DPI for animation. Default: 100.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='FPS for animation. Default: 30.'
    )
    parser.add_argument(
        '--writer',
        type=str,
        default='ffmpeg',
        help='Writer for animation. Default: ffmpeg. pillow also an option.'
    )
    parser.add_argument(
        '--mins',
        type=float,
        nargs='+',
        default=None,
        help='Minimum value for each dimension. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--maxes',
        type=float,
        nargs='+',
        default=None,
        help='Maximum value for each dimension. First dimension interpreted as time.'
    )

    args = parser.parse_args()

    main(args)