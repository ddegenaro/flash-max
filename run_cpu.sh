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

for i in 4 3 2 1; do

    for s in 1 2 3 4 5; do

        # python train.py \
        #     --train_mins  0.0 0.0 0.0 0.0 \
        #     --train_maxes 1.0 1.0 1.0 1.0 \
        #     --val_mins    0.0 0.0 0.0 0.0 \
        #     --val_maxes   1.0 1.0 1.0 1.0 \
        #     --seed ${s} \
        #     --soln ${i} \
        #     --add_bc \
	    #     --do_masking \

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 1.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.0 0.0 0.0 \
            --val_maxes   1.0 1.0 1.0 1.0 \
            --seed ${s} \
            --soln ${i} \
            --add_bc \
            --do_masking \
            --use_cpu \

        # python train.py \
        #     --train_mins  0.0 0.0 0.0 0.0 \
        #     --train_maxes 0.0 1.0 1.0 1.0 \
        #     --val_mins    0.0 0.2 0.2 0.2 \
        #     --val_maxes   0.1 0.8 0.8 0.8 \
        #     --seed ${s} \
        #     --soln ${i} \

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 0.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.2 0.2 0.2 \
            --val_maxes   0.1 0.8 0.8 0.8 \
            --seed ${s} \
            --soln ${i} \
            --use_cpu \
    
    done

done
