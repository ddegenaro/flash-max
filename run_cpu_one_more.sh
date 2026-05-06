#!/bin/bash
#SBATCH --job-name="cpu_pcnn"
#SBATCH --output="slurm_logs/%x_%j.o"
#SBATCH --mem=16gb
#SBATCH --time=48:00:00
#SBATCH --mincpus=2

source ~/.bashrc

cd ~/shallow-nn-wave-eq

conda activate shallownn

python -m uv pip install -r requirements.txt

# always good to check
python --version

# SOLUTIONS 1-4 EACH w/ and w/o BC, 5 seeds

for s in 1 2 3 4 5; do

    python train.py \
        --train_mins  0.1 0.1 0.1 0.1 \
        --train_maxes 1.1 1.1 1.1 1.1 \
        --val_mins    0.1 0.1 0.1 0.1 \
        --val_maxes   1.1 1.1 1.1 1.1 \
        --seed ${s} \
        --soln 2 \
        --add_bc \
        --do_masking \
        --use_cpu \

done
