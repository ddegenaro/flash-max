conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

for n in 2000 5000 10000 20000 50000; do

    for w in 10000 20000 50000; do

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 1.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.0 0.0 0.0 \
            --val_maxes   1.0 1.0 1.0 1.0 \
            --seed ${s} \
            --soln ${i} \
            --add_bc \
            --do_masking \
            --max_time 60 \
            --max_epochs 100000 \
            --width ${w} \
            --n_train ${n} \
            --visible_device 0 \

    done

done
