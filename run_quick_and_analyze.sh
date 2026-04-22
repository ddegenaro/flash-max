conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

PYTHON_PATH="$(which python)"

# SOLUTIONS 1-4 EACH 2 TIMES

for s in 1 2; do

    for i in 2 3 4 1; do

        python train.py \
            --train_mins  0.0 0.0 0.0 0.0 \
            --train_maxes 1.0 1.0 1.0 1.0 \
            --val_mins    0.0 0.0 0.0 0.0 \
            --val_maxes   1.0 1.0 1.0 1.0 \
            --seed ${s} \
            --soln ${i} \
            --add_bc \
            --max_epochs 1000
    
    done

done

PARENT_DIR="experiments"
K=8  # change to whichever rank you want

KTH_DIR=$(ls -d "$PARENT_DIR"/[0-9]* 2>/dev/null \
  | xargs -I{} basename {} \
  | sort -rn \
  | sed -n "${K}p")

python analyze_run.py -s ${KTH_DIR}