conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

for i in 1; do

    for n in 100 200 500 1000 2000 5000 10000 12000; do

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 0.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.2 0.2 0.2 \
            --val_maxes   0.1 0.8 0.8 0.8 \
            --seed 42 \
            --soln ${i} \
            --n_train ${n} \
    
    done

done
