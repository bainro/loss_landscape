# script to evaluate batchsize experiment using lenet5 (ie kaggle model)
import os
import sys
import random

seed = 4491 # random.randint(1,10000)

models = []
base_path = "/home/rbain/git/loss_landscape/models"
for f in os.listdir(base_path):
    if f.endswith(".pth") and "fmnist" in f:
        models.append(f)

assert len(models) > 0

for model in models:

    model_path = os.path.join(base_path, model)
    bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --x=-1:1:125 --eval_count=250 '
    bash_cmd += '--y=-1:1:125 --dataset fmnist --model lenet5 '
    bash_cmd += "--model_file " + model_path + " "
    bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  '
    bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file lenet5_fmnist/' + model + "_seed_" + str(seed) + "_250eval"
    os.system(''.join(list(bash_cmd)))
