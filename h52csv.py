import pandas as pd
import numpy as np
import h5py
import os
import pandas as pd
import numpy as np
import h5py
import os

def h5_to_csv(surf_files, surf_name='test_loss'):

    surfs_and_acc = [
        # "lth_0_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_1_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_2_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_3_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_4_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_5_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_6_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_7_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_8_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_9_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_10_model_lt.pth.tar_lenet5_1337_250eval", "26%"),
        # "lth_11_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_12_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_13_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_14_model_lt.pth.tar_lenet5_1337_250eval", "24%"),
        # "lth_15_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_16_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_17_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_18_model_lt.pth.tar_lenet5_1337_250eval", "21%"),
        # "lth_19_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_20_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_21_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_22_model_lt.pth.tar_lenet5_1337_250eval", "15%"),
        # "lth_23_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_24_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_25_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_26_model_lt.pth.tar_lenet5_1337_250eval", "11%"),
        # "lth_27_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_28_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_29_model_lt.pth.tar_lenet5_1337_250eval",
        ("lth_30_model_lt.pth.tar_lenet5_1337_250eval", "10%"),
        # "lth_31_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_32_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_33_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_34_model_lt.pth.tar_lenet5_1337_250eval",
        # "lth_0_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_1_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_2_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_3_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_4_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_5_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_6_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_7_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_8_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_9_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_10_model_lt.pth.tar_lenet5_3717_250eval", "28%"),
        # "lth_11_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_12_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_13_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_14_model_lt.pth.tar_lenet5_3717_250eval", "25%"),
        # "lth_15_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_16_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_17_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_18_model_lt.pth.tar_lenet5_3717_250eval", "24%"),
        # "lth_19_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_20_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_21_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_22_model_lt.pth.tar_lenet5_3717_250eval", "25%"),
        # "lth_23_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_24_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_25_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_26_model_lt.pth.tar_lenet5_3717_250eval", "27%"),
        # "lth_27_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_28_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_29_model_lt.pth.tar_lenet5_3717_250eval",
        ("lth_30_model_lt.pth.tar_lenet5_3717_250eval", "28%"),
        # "lth_31_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_32_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_33_model_lt.pth.tar_lenet5_3717_250eval",
        # "lth_34_model_lt.pth.tar_lenet5_3717_250eval"
    ]

    df = pd.DataFrame(columns=['id', 'mask', "% unpruned", "test acc", 'x', 'y', 'z'])

    for j, (surf_file, acc) in enumerate(surfs_and_acc):
        if os.path.isdir(surf_file): continue

        if surf_file[5] == "_":
            p = surf_file[4]
        else:
            p = surf_file[4:6]

        prune_percent = 10
        p = ((1 - prune_percent / 100) ** float(p)) * 100
        p = round(p, 1)

        print(surf_file)
        f = h5py.File("/home/rbain/git/loss_landscape/2nd_paper/surfs/pruning_combo/" + surf_file, 'r')

        # seed for random masks was 1337
        if '1337' in surf_file:
            pruning = "random"
        else:
            pruning = "IMP"

        [xcoordinates, ycoordinates] = np.meshgrid(f['xcoordinates'][:], f['ycoordinates'][:][:])
        vals = f[surf_name]

        x_array = xcoordinates[:].ravel()
        y_array = ycoordinates[:].ravel()
        z_array = np.log(vals[:].ravel())

        for i in range(len(z_array)):
            row = {'id': j, 'mask': pruning, '% unpruned': p, 'test acc': acc, 'x': x_array[i], 'y': y_array[i], 'z': z_array[i]}
            df = df.append(row, ignore_index=True)

    print("Total number of rows:", len(df.index))
    df.to_csv("lenet5_cifar10_bs9600_35iters_lth+random.csv", index=False)
        

if __name__ == "__main__":
    h5_dir = os.path.join("/home/rbain/git/loss_landscape/2nd_paper/surfs/pruning_combo")
    plots = sorted(os.listdir(h5_dir))
    # plots.insert(0, plots[-1])
    # plots = plots[:-1]
    #print(plots);exit()
    h5_to_csv(plots)