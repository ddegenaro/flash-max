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

    try:
        model_class = eval(hparams['model_class'])
    except:
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
    model.train()
    
    # tp = torch.tensor([[0.5]]).to(DEVICE)
    # pp = torch.tensor([[5.0, 5.0, 5.0]]).to(DEVICE)
    # print('loaded model pred:', model(tp, pp))
    # print('true:             ', u(tp[0,0], pp[0,0], pp[0,1], pp[0,2]).T)
    # breakpoint()
    # print(hparams['activation'], '----', model.activation)
    # arch_path = os.path.join(path, 'model_architecture.txt')
    # if os.path.exists(arch_path):
    #     model_str = model.__str__().strip()
    #     # for name, param in model.named_parameters():
    #     #     model_str += '\n' + name + ' ' + str(param.shape)
    #     file_contents = open(arch_path, 'r', encoding='utf-8').read().strip()
    #     sep = '\n' + '-' * 20
    #     assert model_str == file_contents, sep + '\n' + model_str + sep + '\n' + file_contents + sep
    # else:
    #     print(f'Warning: no model architecture file found. Assuming correct architecture.')
    # for name, param in model.named_parameters():
    #     print(f'{name}: mean={param.mean():.6f}, std={param.std():.6f}, shape={param.shape}')

    mins = args.mins if args.mins is not None else hparams['val_mins']
    maxes = args.maxes if args.maxes is not None else hparams['val_maxes']
    
    tr = args.temporal_resolution
    sr = args.spatial_resolution
    
    input_cols = [torch.arange(mins[0], maxes[0], (maxes[0] - mins[0]) / tr)] + [
        torch.arange(start, end, (end - start) / sr)
        for start, end in zip(mins[1:], maxes[1:])
    ]
    
    # if hparams['norm_inputs']:
    #     mu_inputs = hparams['mu_inputs']
    #     sigma_inputs = hparams['sigma_inputs']
        
    #     mu_targets = hparams['mu_targets']
    #     sigma_targets = hparams['sigma_targets']
        
    #     assert len(mu_inputs) == len(sigma_inputs) == len(input_cols)
    #     for i in range(len(mu_inputs)):
    #         if sigma_inputs[i] != 0:
    #             input_cols[i] = (input_cols[i] - mu_inputs[i]) / sigma_inputs[i]
            
    if output_dim == 1:
        X, Y = torch.meshgrid(input_cols[1], input_cols[2])
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        if is_3d_input:
            if args.z_val is None:
                z_val = (hparams['val_mins'][-1] + hparams['val_maxes'][-1]) / 2
            else:
                z_val = args.z_val
            Z_flat = torch.ones(X_flat.size()) * z_val # fixed z-value
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
        for plot_true_sol in (True, False):
            
            if double_quiver:
                fname_E = f'animation_true_E.{args.ext}' if plot_true_sol else f'animation_E.{args.ext}'
                fp_E = os.path.join(path, fname_E)
                if os.path.exists(fp_E):
                    continue
                
                fname_B = f'animation_true_B.{args.ext}' if plot_true_sol else f'animation_B.{args.ext}'
                fp_B = os.path.join(path, fname_B)
                if os.path.exists(fp_B):
                    continue
            else:
                fname = f'animation_true.{args.ext}' if plot_true_sol else f'animation.{args.ext}'
                fp = os.path.join(path, fname)
                if os.path.exists(fp):
                    continue
                
            frame_count = len(input_cols[0])
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            Us = {
                'E': torch.zeros((frame_count, spatial_size, output_dim // 2)),
                'B': torch.zeros((frame_count, spatial_size, output_dim // 2))
            } if double_quiver else torch.zeros((frame_count, spatial_size, output_dim))
            
            if plot_true_sol:
                print('Plotting true solution...')
            else:
                print('Plotting predictions...\n')
            
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
                        if batch_idx == 0:
                            tp = torch.tensor([[0.5]]).to(DEVICE)
                            pp = torch.tensor([[5.0, 5.0, 5.0]]).to(DEVICE)
                            print('\n')
                            # print(f'coords: ({tp[0,0].item():.4f}, {pp[0,0].item():.4f}, {pp[0,1].item():.4f}, {pp[0,2].item():.4f})')
                            # print(f'pred:', [round(x, 4) for x in model(tp, pp, True)[0].tolist()])
                            print(model.activation)
                            breakpoint()
                        U_batch = model(t_batch, pos_batch)
                    
                    U_batch = U_batch.detach().cpu().reshape(-1, hparams['output_dim'])
                    
                    # if hparams['norm_inputs']:
                    #     for output_dim in range(len(mu_targets)):
                    #         U_batch[:, 0] = (U_batch[:, output_dim] * sigma_targets[output_dim]) + mu_targets[output_dim]
                            
                    Us['E'][i][start_idx:end_idx] = U_batch[:, 0:3]
                    Us['B'][i][start_idx:end_idx] = U_batch[:, 3:6]
                    
                    # if i % 10 == 0:
                        # p_idx = int(555)
                        # tp = t_batch[p_idx].item()
                        # pp = pos_batch[p_idx].tolist()
                        # if plot_true_sol:
                        #     print(f'coords: ({tp:.1f}, {pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}), true:', U_batch[0])
                        # else:
                        #     print(f'coords: ({tp:.1f}, {pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}), pred:', U_batch[0])
                        # print(model(t_batch[:1], pos_batch[:1]))
                        # print(model(torch.tensor([[t_batch[0,0].item()]]).to(DEVICE), torch.tensor([[pos_batch[0,0].item(), pos_batch[0,1].item(), pos_batch[0,2].item()]]).to(DEVICE)))
                            
                    del U_batch, pos_batch, t_batch
            
            # print("True solution frame 0 spatial variation:")
            # print(f"E[0] min={Us['E'][0].min():.6f}, max={Us['E'][0].max():.6f}")
            # print("\nModel predictions frame 0 spatial variation:")
            # (same for model Us)
            
            # frame loop done, all Us computed
            
            print(Us['E'][0, :3])
            print(Us['E'][1, :3])
            
            print('Writing frames...')
            
            X_shape = X_np.shape
            
            if double_quiver:
                
                E_magnitude = torch.sqrt((Us['E']**2).sum(2))
                B_magnitude = torch.sqrt((Us['B']**2).sum(2))
                E_norm = plt.Normalize(vmin=E_magnitude.min(), vmax=E_magnitude.max())
                B_norm = plt.Normalize(vmin=B_magnitude.min(), vmax=B_magnitude.max())
                E_colors = plt.cm.RdBu(E_norm(E_magnitude.flatten()))
                B_colors = plt.cm.RdBu(B_norm(B_magnitude.flatten()))
                
                writer = animation.PillowWriter(fps=args.fps)
                writer.setup(fig, fp_E, dpi=args.dpi)
                
                for j in tqdm(range(frame_count)):
                
                    ax.clear()
                    ax.quiver(
                        X_np, Y_np, Z_np,
                        Us['E'][j, :, 0].reshape(X_shape), Us['E'][j, : , 1].reshape(X_shape), Us['E'][j, :, 2].reshape(X_shape),
                        length=0.1, normalize=True, arrow_length_ratio=0.5,
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
                        Us['B'][j, :, 0].reshape(X_shape), Us['B'][j, : , 1].reshape(X_shape), Us['B'][j, :, 2].reshape(X_shape),
                        length=0.1, normalize=True, arrow_length_ratio=0.5,
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

    args = parser.parse_args()

    main(args)