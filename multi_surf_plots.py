# script to allow multiple slices to be taken more autonomously
import os
import sys
import random

num_slices = sys.argv[1] if len(sys.argv) > 1 else 2
num_slices = int(num_slices)

#os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_*.h5'")
#os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5'")

for model in ["kaggle_cifar10_BS=12800"]:
    os.system("export loss_plane_file='/home/rbain/git/loss_landscape/models/" + model + ".pth_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5'")
    for i in range(num_slices):
        seed = random.randint(1,10000)

        # bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet56 --x=-2:2:10 '
        # bash_cmd += '--y=-2:2:10 --model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 '
        bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model ' + model + ' --x=-1:1:60 --eval_count=' + str(10000) + ' '
        bash_cmd += '--y=-1:1:60 --dataset cifar10 --model_file models/' + model + '.pth '
        bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  '
        # bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file resnet56_' + str(seed) + "__"
        bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file ' + model + '_' + str(seed) + "__"
        bash_cmd  = list(bash_cmd)

        print("Print the required BASH var $loss_plane_file:")
        os.system('echo $loss_plane_file')
        # delete old file for new directions/slice
        os.system('rm -f $loss_plane_file')
        bash_cmd[-1] = str(i)
        #print(''.join(bash_cmd))
        #exit()
        os.system(''.join(bash_cmd))
        os.system('cp *' + str(seed) + '*3dsurface.pdf /mnt/fast_storage/media/loss_landscape')
