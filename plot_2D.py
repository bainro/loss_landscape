"""
    2D plotting funtions
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns
import copy


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    #print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + surf_name + '_2dheat.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    plt.rcParams['grid.color'] = "blue"   
    # params = {"ytick.color" : "blue",
    #           "grid.color" : "blue",
    #           "xtick.color" : "blue",
    #           "axes.labelcolor" : "blue",
    #           "axes.edgecolor" : "black"}
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    #ax.azim = 10
    ax.elev = 40
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    #ax.set_facecolor('black')
    #fig.set_facecolor('black')
    # ax.w_xaxis.pane.fill = False
    # ax.w_yaxis.pane.fill = False
    # ax.w_zaxis.pane.fill = False
    # hacky outlier trimming
    #Z = np.clip(Z, 0, 22)
    Z = np.log(Z)

    radius = X.shape[0] / 2
    # loop thru every index of X.shape
    # keep elements whose indices euclidian distance is less than the radius
    ax.set_zlim(0, np.max(Z))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist_center = ((i-radius)**2 + (j-radius)**2)**0.5
            if dist_center > radius:
                Z[i][j] = np.nan

    Z = np.ma.array(Z, mask=np.isnan(Z))
    #print(np.max(Z))

    # default coolwarm, tried plasma, Greens_r
    cmap = copy.copy(cm.get_cmap("coolwarm"))
    cmap.set_bad(alpha=0)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, rstride=1, cstride=1, linewidth=0, antialiased=True, vmin=0, vmax=np.max(Z))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=600,
                bbox_inches='tight', format='pdf')

    f.close()
    if show: plt.show()


def plot_trajectory(proj_file, dir_file, show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    assert exists(proj_file), 'Projection file does not exist.'
    f = h5py.File(proj_file, 'r')
    fig = plt.figure()
    plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')
    f.close()

    if exists(dir_file):
        f2 = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f2.close()

    fig.savefig(proj_file + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    if show: plt.show()


def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])

    fig = plt.figure()
    CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

    # plot trajectories
    pf = h5py.File(proj_file, 'r')
    plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pf.close()
    if show: plt.show()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig', show=False):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """

    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])

    # Plot 2D heatmaps with color bar using seaborn
    abs_ratio = np.absolute(np.divide(Z1, Z2))
    print(abs_ratio)

    fig = plt.figure()
    sns_plot = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=.5, cbar=True,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_abs_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # Plot 2D heatmaps with color bar using seaborn
    ratio = np.divide(Z1, Z2)
    print(ratio)
    fig = plt.figure()
    sns_plot = sns.heatmap(ratio, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')
    f.close()
    if show: plt.show()

def plot_3d_scatter(surf_file, surf_name='test_loss', show=False):
    """Plot 3D scatterplot with color and size mapping."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    z = np.array(f['zcoordinates'][:])
    X, Y, Z = np.meshgrid(x, y, z)

    if surf_name in f.keys():
        loss = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err':
        print("UMMMM....")
        loss = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    #loss = np.log(loss+1)

    print("loading surface file: " + surf_file)
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(loss), surf_name, np.min(loss)))
    print("the above could be after logging!")

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 3d scatter plot
    # --------------------------------------------------------------------
    fig = plt.figure()
    #plt.rcParams['grid.color'] = "blue"
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    #print(loss.shape, loss)
    s = np.copy(loss)
    # shift to 0
    s = s - np.amin(s)
    # mirror the values
    s = np.amax(s) - s
    # softmax scaling
    s = np.exp(s - np.max(s))
    s /= s.sum()
    clr = np.copy(s)
    # multiply by a constant
    c = 7500
    s *= c
    #print(np.amax(s), np.amin(s))
    #cm = plt.get_cmap("copper_r")
    #ax.elev = 40
    # ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    # ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    # ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    p = ax.scatter(X, Y, Z, c=clr, s=s, depthshade=False, alpha=0.7)
    fig.colorbar(p, fraction=0.013, pad=0.05)
    #plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_3dscatter' + '.png', dpi=1200,
                bbox_inches='tight', format='png')

    if show: plt.show()

def plot_4d_path(surf_file, surf_name='test_loss', show=False):
    """Plot 3D scatterplot over time using color and size mapping."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    z = np.array(f['zcoordinates'][:])
    t = np.array(f['tcoordinates'][:])
    X, Y, Z, T = np.meshgrid(x, y, z, t)

    if surf_name in f.keys():
        loss = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err':
        print("UMMMM....")
        loss = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    #loss = np.log(loss+1)

    print("loading surface file: " + surf_file)
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(loss), surf_name, np.min(loss)))
    print("the above could be after logging!")

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 3d scatter plot
    # --------------------------------------------------------------------
    fig = plt.figure()
    #plt.rcParams['grid.color'] = "blue"
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    #print(loss.shape, loss)
    s = np.copy(loss)
    # shift to 0
    s = s - np.amin(s)
    # mirror the values
    s = np.amax(s) - s
    # softmax scaling
    s = np.exp(s - np.max(s))
    s /= s.sum()
    clr = np.copy(s)
    # multiply by a constant
    c = 7500
    s *= c
    #print(np.amax(s), np.amin(s))
    #cm = plt.get_cmap("copper_r")
    #ax.elev = 40
    # ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    # ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    # ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1))
    p = ax.scatter(X, Y, Z, c=clr, s=s, depthshade=False, alpha=0.7)
    fig.colorbar(p, fraction=0.013, pad=0.05)
    #plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_3dscatter' + '.png', dpi=1200,
                bbox_inches='tight', format='png')

    if show: plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')

    args = parser.parse_args()

    if exists(args.surf_file) and exists(args.proj_file) and exists(args.dir_file):
        plot_contour_trajectory(args.surf_file, args.dir_file, args.proj_file,
                                args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
    elif exists(args.proj_file) and exists(args.dir_file):
        plot_trajectory(args.proj_file, args.dir_file, args.show)
    elif exists(args.surf_file):
        plot_2d_contour(args.surf_file, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
