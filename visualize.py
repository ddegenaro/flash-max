import os
import sys
import json
import argparse
from typing import Any

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

from wave_equation import Wave, WaveSimplified
from maxwell_equation import Maxwell, MaxwellSimple
from utils import DEVICE
from symlog import symexp
# DEVICE = 'cpu'

def main(args):
    
    if args.experiment_num < 1:
        experiment_num = str(max([
            int(x) for x in os.listdir('experiments') if not x.startswith('.')
        ]))
    else:
        experiment_num = str(args.experiment_num)
    
    print(f'Visualizing experiment {experiment_num}...')
    path = os.path.join('experiments', experiment_num)
    
    sys.path.append(path)
    
    from f import u

    hparams = json.load(
        open(os.path.join(path, 'hparams.json'), 'r', encoding='utf-8')
    )
    input_dim = hparams['input_dim']
    output_dim = hparams['output_dim']
    
    is_3d_input = (input_dim == 3)
    double_quiver = (output_dim == 6)

    model_class = eval(hparams['model_class'])
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
    model.eval()

    mins = args.mins if args.mins is not None else hparams['val_mins']
    maxes = args.maxes if args.maxes is not None else hparams['val_maxes']
    
    tr = args.temporal_resolution
    sr = args.spatial_resolution
    
    if args.length is None:
        length = (maxes[1] - mins[1]) / sr / 1.5
    
    input_cols = [torch.arange(mins[0], maxes[0], (maxes[0] - mins[0]) / tr)] + [
        torch.arange(start, end, (end - start) / sr)
        for start, end in zip(mins[1:], maxes[1:])
    ]
    for i in range(len(input_cols)):
        input_cols[i] += 1e-8
    
    frame_count = len(input_cols[0])
    
    if hparams['norm_inputs']:
        mu_inputs = torch.tensor(hparams['mu_inputs'])
        sigma_inputs = torch.tensor(hparams['sigma_inputs'])
        
        mu_targets = torch.tensor(hparams['mu_targets'])
        sigma_targets = torch.tensor(hparams['sigma_targets'])
        
        assert len(mu_inputs) == len(sigma_inputs) == len(input_cols)
            
        def norm_pos(pos_batch):
            return (pos_batch - mu_inputs[1:].to(pos_batch.device)) / sigma_inputs[1:].to(pos_batch.device)
        
        def unnorm_outputs(U_batch):
            return (U_batch * sigma_targets.to(U_batch.device)) + mu_targets.to(U_batch.device)
    else:
        def norm_pos(pos_batch):
            return pos_batch
        
        def unnorm_outputs(U_batch):
            return U_batch
            
    if output_dim == 1:
        X, Y = torch.meshgrid(input_cols[1], input_cols[2])
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        if is_3d_input:
            if args.z_val is None:
                z_val = (hparams['val_mins'][-1] + hparams['val_maxes'][-1]) / 2
            else:
                z_val = args.z_val
            Z_flat = torch.ones(X_flat.size()) * (z_val - mu_inputs[-1]) / sigma_inputs[-1] # fixed z-value
    else:
        X, Y, Z = torch.meshgrid(input_cols[1], input_cols[2], input_cols[3])
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
    
    if is_3d_input:
        pos = torch.stack((X_flat, Y_flat, Z_flat)).transpose(-1, 0).to(DEVICE)
    else:
        pos = torch.stack((X_flat, Y_flat)).transpose(-1, 0).to(DEVICE)
        
    spatial_size = X_flat.shape[0]
        
    del X_flat, Y_flat, Z_flat
    
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    del X, Y
    try:
        Z_np = Z.cpu().numpy()
        del Z
    except:
        pass
    
    batch_size = spatial_size // args.num_batches
    remainder = spatial_size % args.num_batches
    
    with torch.no_grad():
        
        Us = {
            'true': {
                'E': torch.zeros((frame_count, spatial_size, output_dim // 2)),
                'B': torch.zeros((frame_count, spatial_size, output_dim // 2))
            },
            'pred': {
                'E': torch.zeros((frame_count, spatial_size, output_dim // 2)),
                'B': torch.zeros((frame_count, spatial_size, output_dim // 2))
            }
        } if double_quiver else torch.zeros((frame_count, spatial_size, output_dim))
        
        for plot_true_sol in (True, False):
            
            if args.trueonly and plot_true_sol == False:
                continue
            
            if double_quiver:
                fname_E = f'animation_true_E.{args.ext}' if plot_true_sol else f'animation_E.{args.ext}'
                fp_E = os.path.join(path, fname_E)
                if os.path.exists(fp_E) and not args.overwrite:
                    continue
                
                fname_B = f'animation_true_B.{args.ext}' if plot_true_sol else f'animation_B.{args.ext}'
                fp_B = os.path.join(path, fname_B)
                if os.path.exists(fp_B) and not args.overwrite:
                    continue
            else:
                fname = f'animation_true.{args.ext}' if plot_true_sol else f'animation.{args.ext}'
                fp = os.path.join(path, fname)
                if os.path.exists(fp) and not args.overwrite:
                    continue
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            
            if plot_true_sol:
                print('Plotting true solution...')
                key = 'true'
            else:
                print('Plotting predictions...\n')
                key = 'pred'
            
            for i in tqdm(range(frame_count), total=frame_count):
                
                t = input_cols[0][i].expand(spatial_size).unsqueeze(1)
                
                for batch_idx in range(args.num_batches):
                    
                    start_idx = batch_idx * batch_size + min(batch_idx, remainder)
                    end_idx = start_idx + batch_size + (1 if batch_idx < remainder else 0)
                    
                    pos_batch = pos[start_idx:end_idx].to(DEVICE)
                    t_batch = t[:pos_batch.shape[0]].to(DEVICE)
                    
                    if plot_true_sol:
                        if is_3d_input:
                            U_batch = u(t_batch.squeeze(-1), pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2]).T
                        else:
                            U_batch = u(t_batch.squeeze(-1), pos_batch[:, 0], pos_batch[:, 1]).T
                    else:
                        if 'symlog' in hparams and hparams['symlog']:
                            U_batch = unnorm_outputs(symexp(model(t_batch, norm_pos(pos_batch))))
                        else:
                            U_batch = unnorm_outputs(model(t_batch, norm_pos(pos_batch)))
                    
                    U_batch = U_batch.detach().cpu().reshape(-1, hparams['output_dim'])
                            
                    Us[key]['E'][i][start_idx:end_idx] = U_batch[:, 0:3]
                    Us[key]['B'][i][start_idx:end_idx] = U_batch[:, 3:6]
                            
                    del U_batch, pos_batch, t_batch
            
            # frame loop done, all Us computed
            
            print('Writing frames...')
            
            print(f'True sol: {plot_true_sol}')
            print(f'E: {Us[key]["E"].nanmean()}, {torch.isnan(Us[key]["E"]).sum().item()} nan')
            print(f'B: {Us[key]["B"].nanmean()}, {torch.isnan(Us[key]["B"]).sum().item()} nan')
            
            if args.fix_nan:
                Us[key]['E'] = torch.nan_to_num(Us[key]['E'])
                Us[key]['B'] = torch.nan_to_num(Us[key]['B'])
            
            X_shape = X_np.shape
            
            if double_quiver:
                
                E_magnitude = torch.sqrt((Us[key]['E']**2).sum(2))
                B_magnitude = torch.sqrt((Us[key]['B']**2).sum(2))
                
                if plot_true_sol and not args.trueonly:
                    E_norm = plt.Normalize(vmin=E_magnitude.min(), vmax=E_magnitude.max())
                    B_norm = plt.Normalize(vmin=B_magnitude.min(), vmax=B_magnitude.max())
                
                E_colors = plt.cm.RdBu(E_norm(E_magnitude.flatten()))
                B_colors = plt.cm.RdBu(B_norm(B_magnitude.flatten()))
                
                assert E_magnitude.shape == B_magnitude.shape == Us[key]['E'].shape[:2] == Us[key]['B'].shape[:2]
                
                writer = animation.PillowWriter(fps=args.fps)
                writer.setup(fig, fp_E, dpi=args.dpi)
                
                for j in tqdm(range(frame_count)):
                
                    ax.clear()
                    ax.quiver(
                        X_np, Y_np, Z_np,
                        Us[key]['E'][j, :, 0].reshape(X_shape),
                        Us[key]['E'][j, :, 1].reshape(X_shape),
                        Us[key]['E'][j, :, 2].reshape(X_shape),
                        length=length, normalize=True, arrow_length_ratio=args.alr,
                        colors=E_colors
                    )
                    writer.grab_frame()
                
                writer.finish()

                writer = animation.PillowWriter(fps=args.fps)
                writer.setup(fig, fp_B, dpi=args.dpi)
                
                for j in tqdm(range(frame_count)):
                
                    ax.clear()
                    ax.quiver(
                        X_np, Y_np, Z_np,
                        Us[key]['B'][j, :, 0].reshape(X_shape),
                        Us[key]['B'][j, : , 1].reshape(X_shape),
                        Us[key]['B'][j, :, 2].reshape(X_shape),
                        length=length, normalize=True, arrow_length_ratio=args.alr,
                        colors=B_colors
                    )
                    writer.grab_frame()
                
                writer.finish()
                
                plt.close(fig)
                
            else:
                
                ani = animation.ArtistAnimation(
                    fig=fig,
                    artists=[
                        ax.plot_surface(
                            X_np, Y_np, Us[j],
                            cmap=cm.coolwarm,
                            linewidth=0,
                            antialiased=False
                        ) for j in range(frame_count)
                    ],
                    interval=tr
                )
                ani.save(
                    filename=fp,
                    writer=args.writer,
                    fps=args.fps,
                    dpi=args.dpi
                )
        
        # breakpoint()
        if double_quiver:    
            plt.figure(1)
            plt.plot(
                input_cols[0].cpu().numpy(),
                [
                    torch.nn.MSELoss()(Us['true']['E'][k], Us['pred']['E'][k]).item()
                    for k in range(len(Us['true']['E']))
                ],
                label='E'
            )
            plt.plot(
                input_cols[0].cpu().numpy(),
                [
                    torch.nn.MSELoss()(Us['true']['B'][k], Us['pred']['B'][k]).item()
                    for k in range(len(Us['true']['B']))
                ],
                label='B'
            )
            plt.legend()
            plt.savefig(os.path.join(path, 'loss_over_video.png'), dpi=300)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_batches',
        type=int,
        default=1,
        help='Batches per frame to avoid GPU overload.'
    )
    parser.add_argument(
        '--experiment_num', '-e',
        type=int,
        default=-1,
        help='Assumes there is an experiment with this number in the experiments directory. Default: most recent.'
    )
    parser.add_argument(
        '--temporal_resolution',
        type=float,
        default=50,
        help='Number of time steps to evaluate at per unit time. Default 100.'
    )
    parser.add_argument(
        '--spatial_resolution',
        type=float,
        default=10,
        help='Number of space steps to evaluate at per unit space. Default 100.'
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
    parser.add_argument(
        '--z_val',
        type=float,
        default=None,
        help='Value to fix z to project 3D to 2D.'
    )
    parser.add_argument(
        '--trueonly',
        action='store_true',
        default=False,
        help='Plot true solution only.'
    )
    parser.add_argument(
        '--length',
        type=float,
        default=None,
        help='Quiver plot arrow length.'
    )
    parser.add_argument(
        '--alr',
        type=float,
        default=0.5,
        help='Quiver plot arrow length ratio.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=True,
        help='Overwrite existing viz.'
    )
    parser.add_argument(
        '--fix_nan',
        action='store_true',
        default=True,
        help='Replace nan\'s with 0.'
    )

    args = parser.parse_args()

    main(args)