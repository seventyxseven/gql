import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec

# define grid parameters
X = np.load('X.npy')
Yh = np.load('Yh.npy')
Z = np.load('Z.npy')
Yv = np.load('Yv.npy')
kx = np.load('kx.npy')
ky = np.load('ky.npy')
ky2 = np.load('ky2.npy')

# define plot parameters
cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('inferno')
ticksize = 14
fsize = 16

# Load spectra files
nkx, nky = kx.shape
ek_xy_nl = np.load('ek_xy_nl.npy')
ek_xy_gql2 = np.load('ek_xy_gql2.npy')
ek_xy_gql3 = np.load('ek_xy_gql3.npy')
ek_xy_gql8 = np.load('ek_xy_gql8.npy')
ek_xy_ql = np.load('ek_xy_ql.npy')
u_xy_NW_nl = np.load('u_xy_NW_nl.npy')
u_xy_NW_gql2 = np.load('u_xy_NW_gql2.npy')
u_xy_NW_gql3 = np.load('u_xy_NW_gql3.npy')
u_xy_NW_gql8 = np.load('u_xy_NW_gql8.npy')
u_xy_NW_ql = np.load('u_xy_NW_ql.npy')

# Find index of max energy spectra
ind_ek_ql = np.unravel_index(np.argmax(ek_xy_ql, axis=None), ek_xy_ql.shape)
ind_ek_gql2 = np.unravel_index(np.argmax(ek_xy_gql2, axis=None), ek_xy_gql2.shape)
ind_ek_gql3 = np.unravel_index(np.argmax(ek_xy_gql3, axis=None), ek_xy_gql3.shape)
ind_ek_gql8 = np.unravel_index(np.argmax(ek_xy_gql8, axis=None), ek_xy_gql3.shape)
ind_ek_dns = np.unravel_index(np.argmax(ek_xy_nl, axis=None), ek_xy_nl.shape)

def nw_Ekxy(nx, ny):
    ld = nx / 2
    # Define max/mins for shared colorbar
    combined_u_xy_NW = np.array([u_xy_NW_ql, u_xy_NW_gql3, u_xy_NW_nl])
    u_xy_NW_min, u_xy_NW_max = np.amin(combined_u_xy_NW), np.amax(combined_u_xy_NW)
    combined_ek_xy = np.array([ek_xy_ql, ek_xy_gql3, ek_xy_nl])  # leave QL out of this for now (or forever)
    ek_xy_min, ek_xy_max = np.amin(combined_ek_xy), np.amax(combined_ek_xy)

    fig = plt.figure(7, figsize=(15, 8))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1, 1])
    gs.update(left=0.05, right=0.95, bottom=0.09, top=0.9, wspace=0.2, hspace=0.2)
    # set axes (2 rows, 5 columns per row... [ql, colorbar, extra row for spacing, gql, dns, sharedcolorbar]
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
    ax02 = fig.add_subplot(gs[0, 2], sharey=ax00)
    ax03 = fig.add_subplot(gs[0, 3], sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)
    ax12 = fig.add_subplot(gs[1, 2], sharey=ax10)
    ax13 = fig.add_subplot(gs[1, 3], sharey=ax10)
    # Define plots
    plt00 = ax00.contourf(X, Yh, np.flipud(u_xy_NW_ql), vmax=u_xy_NW_max, vmin=u_xy_NW_min,cmap=cmap)  # name to create mappable for colorbar
    plt01 = ax01.contourf(X, Yh, np.flipud(u_xy_NW_gql3), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap=cmap)
    plt02 = ax02.contourf(X, Yh, np.flipud(u_xy_NW_gql8), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap=cmap)
    plt03 = ax03.contourf(X, Yh, np.flipud(u_xy_NW_nl), vmax=u_xy_NW_max, vmin=u_xy_NW_min,cmap=cmap)  # set vmax/vmin to normalize cbar
    plt10 = ax10.contourf(kx[1:11, 1:int(ny/5)], ky2[1:11, 1:int(ny/5)], ek_xy_ql[0:10, 0:15], vmax=ek_xy_max, vmin=ek_xy_min, cmap=cmap2)  # name to create mappable for colorbar
    plt11 = ax11.contourf(kx[1:11, 1:int(ny/5)], ky2[1:11, 1:int(ny/5)], ek_xy_gql3[0:10, 0:15], vmax=ek_xy_max, vmin=ek_xy_min, cmap=cmap2)  # name to create mappable for colorbar
    plt12 = ax12.contourf(kx[1:11, 1:int(ny/5)], ky2[1:11, 1:int(ny/5)], ek_xy_gql8[0:10, 0:15], vmax=ek_xy_max, vmin=ek_xy_min, cmap=cmap2)
    plt13 = ax13.contourf(kx[1:11, 1:int(ny/5)], ky2[1:11, 1:int(ny/5)], ek_xy_nl[0:10, 0:15], vmax=ek_xy_max, vmin=ek_xy_min,cmap=cmap2)
    # Labels
    # fig.suptitle('2D Horizontal (x-y) Energy Spectrum, Near Wall', size=18)
    ax00.set_title('$\Lambda_x$ = 0 (QL)', size=14)
    ax00.set_ylabel('y', size=fsize, rotation=0)
    ax00.tick_params(axis='both', which='major', labelsize=ticksize)
    ax00.set_xlabel('x', size=fsize)
    ax01.set_title('$\Lambda_x$ = 3 (GQL)', size=fsize)
    ax01.set_xlabel('x', size=fsize)
    ax01.tick_params(axis='both', which='major', labelsize=ticksize)
    ax02.set_title('$\Lambda_x$ = 8 (GQL)', size=fsize)
    ax02.set_xlabel('x', size=fsize)
    ax02.tick_params(axis='both', which='major', labelsize=ticksize)
    ax03.set_title('$\Lambda_x$ = %i (NL)' % ld, size=fsize)
    ax03.set_xlabel('x', size=fsize)
    ax03.tick_params(axis='both', which='major', labelsize=ticksize)
    ax10.set_ylabel('$k_y$', size=fsize, rotation=0)
    ax10.set_xlabel('$k_x$', size=fsize)
    ax10.tick_params(axis='both', which='major', labelsize=ticksize)
    ax10.set_yscale('log')
    ax10.set_xscale('log')
    ax11.set_xlabel('$k_x$', size=fsize)
    ax11.tick_params(axis='both', which='major', labelsize=ticksize)
    ax11.set_yscale('log')
    ax11.set_xscale('log')
    ax12.set_xlabel('$k_x$', size=fsize)
    ax12.tick_params(axis='both', which='major', labelsize=ticksize)
    ax12.set_xscale('log')
    ax12.set_yscale('log')
    ax13.set_xlabel('$k_x$', size=fsize)
    ax13.tick_params(axis='both', which='major', labelsize=ticksize)
    ax13.xaxis.set_minor_locator(plt.MaxNLocator(0))
    ax13.set_xscale('log')
    ax13.set_yscale('log')
    # Set colorbars
    cbax04 = plt.subplot(gs[0, 4])
    cbax14 = plt.subplot(gs[1, 4])
    cb04 = Colorbar(ax=cbax04, mappable=plt01, orientation='vertical', ticklocation='right')
    cb14 = Colorbar(ax=cbax14, mappable=plt10, orientation='vertical', ticklocation='right')
    cb04.ax.set_yticklabels(cb04.ax.get_yticklabels(), fontsize=ticksize)
    cb04.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cb14.ax.set_yticklabels(cb14.ax.get_yticklabels(), fontsize=ticksize)
    # cb13.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Housekeeping
    plt.setp(ax01.get_yticklabels(), visible=False)  # Turn off ticks for y axes
    plt.setp(ax02.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)
    # plt.setp(ax75.get_yticklabels(), visible=False)
    # plt.setp(ax76.get_yticklabels(), visible=False)
    fig.savefig('xyspectra_nwcompare_zm.png')
    return fig.show()

def maxmode_kx(nx):
    ld = nx / 2

# Spectra maximum by wavenumber kx
    fig, ax = plt.subplots()
    plt.plot(kx[1:nkx, 0], ek_xy_ql[:, ind_ek_ql[1]], 'blue', label='$\Lambda_x$ = 0 (QL)')
    plt.plot(kx[1:nkx, 0], ek_xy_gql2[:, ind_ek_gql2[1]], 'orange', label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_gql3[:, ind_ek_gql3[1]], 'g', label='$\Lambda_x$ = 3 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_gql8[:, ind_ek_gql8[1]], 'r', label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_nl[:, ind_ek_dns[1]], 'purple', label='$\Lambda_x$ = %i (NL)' % ld)
    # ##plt.clabel(contour4, inline=True, fontsize=8)
    # ax.set_title('2D Horizontal (x-y) Energy Spectrum, DNS')
    ax.set_xlabel('$k_x$', fontsize=fsize)
    ax.set_xscale('log')
    ax.set_ylabel('$E_{xy}$', fontsize=fsize)
    ax.set_yscale('log')
    legend = ax.legend(loc='best', shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.yaxis.set_minor_locator(plt.MaxNLocator(3))
    fig.savefig('maxmode_kx.png')
    return plt.show()

def maxmode_kx_zoom(nx):
    ld = nx / 2

# Spectra maximum by wavenumber kx
    fig, ax = plt.subplots()
    plt.plot(kx[1:5, 0], ek_xy_ql[0:4, ind_ek_ql[1]], 'blue', label='$\Lambda_x$ = 0 (QL)')
    plt.plot(kx[1:5, 0], ek_xy_gql2[0:4, ind_ek_gql2[1]], 'orange', label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(kx[1:5, 0], ek_xy_gql3[0:4, ind_ek_gql3[1]], 'g', label='$\Lambda_x$ = 3 (GQL)')
    plt.plot(kx[1:5, 0], ek_xy_gql8[0:4, ind_ek_gql8[1]], 'r', label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(kx[1:5, 0], ek_xy_nl[0:4, ind_ek_dns[1]], 'purple', label='$\Lambda_x$ = %i (NL)' % ld)
    # ##plt.clabel(contour4, inline=True, fontsize=8)
    # ax.set_title('2D Horizontal (x-y) Energy Spectrum, zoomed in, DNS')
    ax.set_xlabel('$k_x$', fontsize=fsize)
    ax.set_xscale('log')
    ax.set_ylabel('$E_{xy}$', fontsize=fsize)
    ax.set_yscale('log')
    legend = ax.legend(loc='best', shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.yaxis.set_minor_locator(plt.MaxNLocator(3))
    fig.savefig('maxmode_kx_zoom.png')
    return plt.show()

def maxmode_kx_noql(nx):
    ld = nx / 2

# Spectra maximum by wavenumber kx
    fig, ax = plt.subplots()
    plt.plot(kx[1:nkx, 0], ek_xy_gql2[:, ind_ek_gql2[1]], 'orange', label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_gql3[:, ind_ek_gql3[1]], 'g', label='$\Lambda_x$ = 3 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_gql8[:, ind_ek_gql8[1]], 'r', label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(kx[1:nkx, 0], ek_xy_nl[:, ind_ek_dns[1]], 'purple', label='$\Lambda_x$ = %i (NL)' % ld)
    # ##plt.clabel(contour4, inline=True, fontsize=8)
    # ax.set_title('2D Horizontal (x-y) Energy Spectrum, DNS')
    ax.set_xlabel('$k_x$', fontsize=fsize)
    ax.set_xscale('log')
    ax.set_ylabel('$E_{xy}$', fontsize=fsize)
    ax.set_yscale('log')
    legend = ax.legend(loc='best', shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.yaxis.set_minor_locator(plt.MaxNLocator(3))
    fig.savefig('maxmode_kx.png')
    return plt.show()

def maxmode_ky(nx):
# Spectra maximum by wavenumber ky
    ld = nx / 2
    fig, ax = plt.subplots()
    plt.plot(ky[0, 1:nkx], ek_xy_ql[ind_ek_ql[0], :], 'blue', label='$\Lambda_x$ = 0 (QL)')
    plt.plot(ky[0, 1:nkx], ek_xy_gql2[ind_ek_gql2[0], :], 'orange', label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(ky[0, 1:nkx], ek_xy_gql3[ind_ek_gql3[0], :], 'g', label='$\Lambda_x$ = 3 (GQL)')
    plt.plot(ky[0, 1:nkx], ek_xy_gql8[ind_ek_gql8[0], :], 'r', label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(ky[0, 1:nkx], ek_xy_nl[ind_ek_dns[0], :], 'purple', label='$\Lambda_x$ = %i (NL)' % ld)
    # ##plt.clabel(contour4, inline=True, fontsize=8)
    # ax.set_title('2D Horizontal (x-y) Energy Spectrum, DNS')
    ax.set_xlabel('$k_y$', fontsize=fsize)
    ax.set_xscale('log')
    ax.set_ylabel('$E_{xy}$', fontsize=fsize)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.yaxis.set_minor_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_minor_locator(plt.MaxNLocator(5))
    legend = ax.legend(loc='best', shadow=True)
    fig.savefig('Ek_ky.png')
    return plt.show()
