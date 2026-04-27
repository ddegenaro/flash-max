conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# SOLUTIONS 1-4 EACH w/ and w/o BC, 5 seeds

for i in 4 3 2 1; do

    for s in 1 2 3 4 5; do

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 1.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.0 0.0 0.0 \
            --val_maxes   1.0 1.0 1.0 1.0 \
            --seed ${s} \
            --soln ${i} \
            --add_bc \
	    --do_masking \

        # python train.py \
        #     --train_mins  0.0 0.0 0.0 0.0 \
        #     --train_maxes 1.0 1.0 1.0 1.0 \
        #     --val_mins    0.0 0.0 0.0 0.0 \
        #     --val_maxes   1.0 1.0 1.0 1.0 \
        #     --seed ${s} \
        #     --soln ${i} \
        #     --add_bc \
        #     --use_cpu \

        # python train.py \
        #     --train_mins  0.0 0.0 0.0 0.0 \
        #     --train_maxes 0.0 1.0 1.0 1.0 \
        #     --val_mins    0.0 0.2 0.2 0.2 \
        #     --val_maxes   0.1 0.8 0.8 0.8 \
        #     --seed ${s} \
        #     --soln ${i} \

        # python train.py \
        #     --train_mins  0.0 0.0 0.0 0.0 \
        #     --train_maxes 0.0 1.0 1.0 1.0 \
        #     --val_mins    0.0 0.2 0.2 0.2 \
        #     --val_maxes   0.1 0.8 0.8 0.8 \
        #     --seed ${s} \
        #     --soln ${i} \
        #     --use_cpu \
    
    done

done
