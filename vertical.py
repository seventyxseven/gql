import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec

# define grid parameters
Z = np.load('Z.npy')
Yv = np.load('Yv.npy')
Yhalf = np.load('Yhalf.npy')
Zhalf = np.load('Zhalf.npy')
fZ = np.load('fZ.npy')  # finer vert grid
fYv = np.load('fYv.npy')  # finer vert grid

# define plot parameters
cmap = plt.get_cmap('viridis')
ticksize = 14
fsize = 16

def compare(nx, nz):
    ld = nx / 2
    # load files
    upvert_plot_nl = np.load('upvert_plot_nl.npy')
    upvert_plot_gql3 = np.load('upvert_plot_gql3.npy')
    upvert_plot_ql = np.load('upvert_plot_ql.npy')
    vp_ql = np.load('vp_ql.npy')
    wp_ql = np.load('wp_ql.npy')
    vp_gql3 = np.load('vp_gql3.npy')
    wp_gql3 = np.load('wp_gql3.npy')
    vp_nl = np.load('vp_nl.npy')
    wp_nl = np.load('wp_nl.npy')
# define max/min for colorbar
    combined = np.array([upvert_plot_ql, upvert_plot_gql3, upvert_plot_nl])
    cbarmin, cbarmax = np.amin(combined), np.amax(combined)
    fig = plt.figure(7, figsize=(15,8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    gs.update(left=0.05, right=0.95, bottom=0.2, top=0.6, wspace=0.2, hspace=0.2)
    # set axes (2 rows, 5 columns per row... [ql, colorbar, extra row for spacing, gql, dns, sharedcolorbar]
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax02 = fig.add_subplot(gs[0, 2], sharey=ax00)
    # Define plots
    plt00 = ax00.pcolormesh(Yv[:, 0:nz/2], Z[:, 0:nz/2], upvert_plot_ql, vmax=cbarmax, vmin=cbarmin, cmap=cmap)
    ax00.quiver(Yhalf[::3, ::3], Zhalf[::3, ::3], vp_ql[::3, ::3], wp_ql[::3, ::3])
    plt01 = ax01.pcolormesh(Yv[:, 0:nz/2], Z[:, 0:nz/2], upvert_plot_gql3, vmax=cbarmax, vmin=cbarmin, cmap=cmap)
    ax01.quiver(Yhalf[::3, ::3], Zhalf[::3, ::3], vp_gql3[::3, ::3], wp_gql3[::3, ::3])
    plt02 = ax02.pcolormesh(Yv[:, 0:nz/2], Z[:, 0:nz/2], upvert_plot_nl, vmax=cbarmax, vmin=cbarmin, cmap=cmap)
    ax02.quiver(Yhalf[::3, ::3], Zhalf[::3, ::3], vp_nl[::3, ::3], wp_nl[::3, ::3])
    # Labels
    # fig.suptitle('Streamwise Fluctuations of Time-Averaged Velocity, Near Wall', size=18)
    ax00.set_title('$\Lambda_x$ = 0 (QL)', size=14)
    ax00.set_ylabel('z', size=fsize, rotation=0)
    ax00.tick_params(axis='both', which='major', labelsize=ticksize)
    ax00.set_xlabel('y', size=fsize)
    ax00.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax01.set_title('$\Lambda_x$ = 3 (GQL)', size=fsize)
    ax01.set_xlabel('y', size=fsize)
    ax01.tick_params(axis='both', which='major', labelsize=ticksize)
    ax02.set_title('$\Lambda_x$ = %i (NL)' %ld, size=fsize)
    ax02.set_xlabel('y', size=fsize)
    ax02.tick_params(axis='both', which='major', labelsize=ticksize)
    # Set colorbars
    cbax03 = plt.subplot(gs[0, 3])
    cb03 = Colorbar(ax=cbax03, mappable=plt01, orientation='vertical', ticklocation='right')
    cb03.ax.set_yticklabels(cb03.ax.get_yticklabels(), fontsize=ticksize)
    cb03.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Housekeeping
    plt.setp(ax01.get_yticklabels(), visible=False)      # Turn off ticks for y axes
    plt.setp(ax02.get_yticklabels(), visible=False)
    # plt.setp(ax75.get_yticklabels(), visible=False)
    # plt.setp(ax76.get_yticklabels(), visible=False)
    fig.savefig('vert_structure_compare.png')
    return fig.show()

def vertslice(nz):
    f_upvert_plot_ql = np.load('f_upvert_plot_ql.npy')
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(fYv, fZ, f_upvert_plot_ql, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    # ax.set_title('Finer Instantaneous NW NL')
    ax.set_ylabel('y', fontsize=fsize, rotation=1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('f_u_xy_nw_nl_instant')
    return fig.show()

def quiver(nz):
    upvert_plot_ql = np.load('upvert_plot_ql.npy')
    vp_ql = np.load('vp_ql.npy')
    wp_ql = np.load('wp_ql.npy')
    fig, ax = plt.subplots()
    velslice = ax.pcolormesh(Yv[:, 0:nz/2], Z[:, 0:nz/2], upvert_plot_ql, cmap=cmap)
    ax.quiver(Yhalf[::3, ::3], Zhalf[::3, ::3], vp_ql[::3, ::3], wp_ql[::3, ::3])
    fig.colorbar(velslice)
    ax.set_title('Perturbation Structure')
    ax.set_xlabel('y', fontsize=14)
    ax.set_ylabel('z', fontsize=14, rotation=0)
    # ax.set_yscale('log')
    # f_upvert_plot_ql = np.load('f_upvert_plot_ql.npy')
    # fig, ax = plt.subplots()
    # plt2 = ax.pcolormesh(fYv, fZ, f_upvert_plot_ql, cmap=cmap)
    # cb = fig.colorbar(plt2)
    # cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    # # ax.set_title('Finer Instantaneous NW NL')
    # ax.set_ylabel('y', fontsize=fsize, rotation=1)
    # ax.set_xlabel('x', fontsize=fsize)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # # ax.tick_params(axis='both', which='minor', labelsize=8)
    # fig.savefig('f_u_xy_nw_nl_instant')
    return fig.show()