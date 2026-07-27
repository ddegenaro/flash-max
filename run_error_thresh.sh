conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

for n in 50000; do

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 1.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.0 0.0 0.0 \
        --val_maxes   1.0 1.0 1.0 1.0 \
        --seed 42 \
        --soln 3 \
        --add_bc \
        --do_masking \
        --max_time 360 \
        --max_epochs 100000 \
        --width 50000 \
        --n_train ${n} \
        --visible_device 1 \

done
