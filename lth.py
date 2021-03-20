# script to allow multiple slices to be taken more autonomously
import os
import sys
import random

seed = 1337 # random.randint(1,10000)

models = []
base_path = "/home/rbain/git/LTH/saves/lenet5/cifar10/"
for f in os.listdir(base_path):
    # missed some the 1st time thru
    if f.endswith(".pth.tar") and ("30_" in f or "27_" in f):
        models.append(f)

assert len(models) > 0

for model in models:
    model_path = os.path.join(base_path, model)
    bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --x=-1:1:125 --eval_count=250 '
    bash_cmd += '--y=-1:1:125 --dataset cifar10 --model lenet5 '
    bash_cmd += "--model_file " + model_path + " "
    bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  '
    bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file lth_same_dir_random_masks/lth_' + model + '_lenet5_' + str(seed) + "_250eval"
    os.system(''.join(list(bash_cmd)))
    #os.system('cp *' + str(seed) + '*3dsurface.pdf /mnt/fast_storage/media/loss_landscape')