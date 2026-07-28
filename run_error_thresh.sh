conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# python train.py \
#     --train_mins  0.0 0.0 0.0 0.0 \
#     --train_maxes 1.0 1.0 1.0 1.0 \
#     --val_mins    0.0 0.0 0.0 0.0 \
#     --val_maxes   1.0 1.0 1.0 1.0 \
#     --seed 42 \
#     --soln 3 \
#     --add_bc \
#     --do_masking \
#     --max_time 360 \
#     --max_epochs 100000 \
#     --width 50000 \
#     --n_train ${n} \
#     --visible_device 1 \

# python train.py \
#     --train_mins  0.0 0.0 0.0 0.0 \
#     --train_maxes 0.0 1.0 1.0 1.0 \
#     --val_mins    0.0 0.2 0.2 0.2 \
#     --val_maxes   0.1 0.8 0.8 0.8 \
#     --seed 42 \
#     --soln 3 \
#     --max_time 360 \
#     --max_epochs 100000 \
#     --width 10000 \
#     --n_train 50000 \
#     --visible_device 0 \

python train.py \
    --train_mins  0.0 0.0 0.0 0.0 \
    --train_maxes 0.0 1.0 1.0 1.0 \
    --val_mins    0.0 0.2 0.2 0.2 \
    --val_maxes   0.1 0.8 0.8 0.8 \
    --seed 42 \
    --soln 3 \
    --max_time 360 \
    --max_epochs 100000 \
    --width 100000 \
    --n_train 1000000 \
    --visible_device 1 \
    --batch_size 10000

python train.py \
    --train_mins  0.0 0.0 0.0 0.0 \
    --train_maxes 0.0 1.0 1.0 1.0 \
    --val_mins    0.0 0.2 0.2 0.2 \
    --val_maxes   0.1 0.8 0.8 0.8 \
    --seed 42 \
    --soln 3 \
    --max_time 360 \
    --max_epochs 100000 \
    --width 10000000 \
    --n_train 1000000000 \
    --visible_device 1 \
    --batch_size 10000