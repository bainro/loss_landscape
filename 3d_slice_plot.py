# script to allow multiple slices to be taken more autonomously
import os
import sys
import random

num_slices = sys.argv[1] if len(sys.argv) > 1 else 1
num_slices = int(num_slices)

# @TODO update this for 3d slices
#os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_*.h5'")
os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5'")

for j in range(1):
    for i in range(num_slices):
        seed = 33451+j #random.randint(1,10000)

        bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model mobilenet --x=-1:1:25 --z=-1:1:25 --eval_count=10 '# + str(i+1) + ' '
        bash_cmd += '--y=-1:1:25 --model_file results/99_acc_mnist_mobilev2.pth '#--model_file2 results/init_99_acc_mnist_mobilev2.pth '
        #bash_cmd += '--y=-1:1:5 --model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 '
        bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --znorm filter --zignore biasbn '
        bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file resnet56_3D+1tD_seed=' + str(seed) + "_--eval_count=10 "# + str(i+1)
        bash_cmd += ' --dataset mnist' #--z=-1:1:5 --znorm filter --zignore biasbn'
        os.system(''.join(bash_cmd))

        # bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet56_noshort --x=-1:1:15 --z=-1:1:15 --eval_count=' + str(i+1) + ' '
        # bash_cmd += '--y=-1:1:15 --model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 '
        # bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --znorm filter --zignore biasbn '
        # bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file resnet56_no_short_seed=' + str(seed) + "_--eval_count=" + str(i+1)
        # os.system(''.join(bash_cmd))