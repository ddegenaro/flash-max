conda init bash

conda activate shallownn

cd ~/shallow-nn-wave-eq

for i in $(seq 1522 1582); do python visualize.py -e ${i}; done
