conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# SOLUTIONS 1-4 EACH w/ and w/o BC, 5 seeds

for n in 0.01 0.001 0.0001 0.00001; do

    for i in 2 3 4; do

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
                --noise_scale ${n} \

            # python train.py \
            #     --train_mins  0.0 0.0 0.0 0.0 \
            #     --train_maxes 1.0 1.0 1.0 1.0 \
            #     --val_mins    0.0 0.0 0.0 0.0 \
            #     --val_maxes   1.0 1.0 1.0 1.0 \
            #     --seed ${s} \
            #     --soln ${i} \
            #     --add_bc \
            #     --use_cpu \
            #     --noise_scale ${n} \

            python train.py \
                --train_mins  0.0 0.0 0.0 0.0 \
                --train_maxes 0.0 1.0 1.0 1.0 \
                --val_mins    0.0 0.2 0.2 0.2 \
                --val_maxes   0.1 0.8 0.8 0.8 \
                --seed ${s} \
                --soln ${i} \
                --noise_scale ${n} \

            # python train.py \
            #     --train_mins  0.0 0.0 0.0 0.0 \
            #     --train_maxes 0.0 1.0 1.0 1.0 \
            #     --val_mins    0.0 0.2 0.2 0.2 \
            #     --val_maxes   0.1 0.8 0.8 0.8 \
            #     --seed ${s} \
            #     --soln ${i} \
            #     --use_cpu \
            #     --noise_scale ${n} \
        
        done

    done

done
