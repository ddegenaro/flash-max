conda init bash

conda activate shallownn

cd ~/flash-max

for i in $(seq 1522 1582); do python visualize.py -e ${i}; done
