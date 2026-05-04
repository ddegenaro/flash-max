conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# widths, activations

for w in 100 200 500 1000 2000 5000 10000 12000; do

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 1.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.0 0.0 0.0 \
        --val_maxes   1.0 1.0 1.0 1.0 \
        --seed 42 \
        --soln 4 \
        --add_bc \
   	--do_masking \
	--width ${w}

done

for a in "relu" "silu" "cosine" "gelu" "sigmoid"; do

    python train.py \
        --train_mins  0.0 0.0 0.0 0.0 \
        --train_maxes 1.0 1.0 1.0 1.0 \
        --val_mins    0.0 0.0 0.0 0.0 \
        --val_maxes   1.0 1.0 1.0 1.0 \
        --seed 42 \
        --soln 4 \
        --add_bc \
        --do_masking \
        --activation ${a}

done

