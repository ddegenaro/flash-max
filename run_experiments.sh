conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# SOLUTIONS 1-4 EACH 4 TIMES - GPU+BC, CPU+BC, GPU-BC, CPU-BC

SOLNS=('1', '2', '3', '4')

for i in "${!SOLNS[@]}"; do

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 1.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.0 0.0 0.0 \
        --val_maxes   1.0 1.0 1.0 1.0 \
        --soln ${i} \
        --add_bc \

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 1.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.0 0.0 0.0 \
        --val_maxes   1.0 1.0 1.0 1.0 \
        --soln ${i} \
        --add_bc \
        --use_cpu \

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 0.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.2 0.2 0.2 \
        --val_maxes   0.1 0.8 0.8 0.8 \
        --soln ${i} \

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 0.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.2 0.2 0.2 \
        --val_maxes   0.1 0.8 0.8 0.8 \
        --soln ${i} \
        --use_cpu \

done
