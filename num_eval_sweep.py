# script to allow multiple slices to be taken more autonomously
import os
import sys
import random

# num_slices = sys.argv[1] if len(sys.argv) > 1 else 2
# num_slices = int(num_slices)
seed = random.randint(1,10000)

os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_*.h5'")

for i in [3000]:

    bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --x=-1:1:125 --eval_count=' + str(i) + ' '
    bash_cmd += '--y=-1:1:125 --dataset cifar10 --model resnet56 '
    bash_cmd += "--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 "
    bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  '
    bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file resnet56_' + str(seed) + "__"
    bash_cmd  = list(bash_cmd)
    bash_cmd[-1] = str(i)

    print("Print the required BASH var $loss_plane_file:")
    os.system('echo $loss_plane_file')
    # delete old file for new directions/slice
    os.system('rm -f $loss_plane_file')
    os.system(''.join(bash_cmd))
    os.system('cp *' + str(seed) + '*3dsurface.pdf /mnt/fast_storage/media/loss_landscape')

# os.system("export loss_plane_file='/home/rbain/git/loss_landscape/cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5'")

# for i in [3000]:

#     bash_cmd  = 'mpirun -n 4 python plot_surface.py --mpi --cuda --x=-1:1:125 --eval_count=' + str(i) + ' '
#     bash_cmd += '--y=-1:1:125 --dataset cifar10 --model resnet56_noshort '
#     bash_cmd += "--model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 "
#     bash_cmd += '--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  '
#     bash_cmd += '--seed ' + str(seed) + ' --plot --surf_file resnet56_noshort_' + str(seed) + "__"
#     bash_cmd  = list(bash_cmd)
#     bash_cmd[-1] = str(i)

#     print("Print the required BASH var $loss_plane_file:")
#     os.system('echo $loss_plane_file')
#     # delete old file for new directions/slice
#     os.system('rm -f $loss_plane_file')
#     os.system(''.join(bash_cmd))
#     os.system('cp *' + str(seed) + '*3dsurface.pdf /mnt/fast_storage/media/loss_landscape')