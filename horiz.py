import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import gridspec
from scipy.interpolate import griddata

import matplotlib.ticker as ticker

# define grid parameters
X = np.load('X.npy')
Yh = np.load('Yh.npy')
fX = np.load('fX.npy')
fYh = np.load('fYh.npy')
x = np.load('x.npy')
y = np.load('y.npy')
x2 = np.load('x2.npy')
y2 = np.load('y2.npy')

# define plot parameters
cmap = plt.get_cmap('viridis')
ticksize = 14
fsize = 16

# Load horiz slice files
# near wall instantaneous
u_xy_nw_nl = np.load('u_xy_nw_nl.npy')
u_xy_nw_ql = np.load('u_xy_nw_ql.npy')
u_xy_nw_gql2 = np.load('u_xy_nw_gql2.npy')
u_xy_nw_gql3 = np.load('u_xy_nw_gql3.npy')
u_xy_nw_gql8 = np.load('u_xy_nw_gql8.npy')
# near wall average
u_xy_NW_nl = np.load('u_xy_NW_nl.npy')
u_xy_NW_ql = np.load('u_xy_NW_ql.npy')
u_xy_NW_gql2 = np.load('u_xy_NW_gql2.npy')
u_xy_NW_gql3 = np.load('u_xy_NW_gql3.npy')
u_xy_NW_gql8 = np.load('u_xy_NW_gql8.npy')
# near centerline instantaneous
u_xy_cl_nl = np.load('u_xy_cl_nl.npy')
u_xy_cl_ql = np.load('u_xy_cl_ql.npy')
u_xy_cl_gql2 = np.load('u_xy_cl_gql2.npy')
u_xy_cl_gql3 = np.load('u_xy_cl_gql3.npy')
u_xy_cl_gql8 = np.load('u_xy_cl_gql8.npy')
# near centerline averages
u_xy_CL_nl = np.load('u_xy_CL_nl.npy')
u_xy_CL_ql = np.load('u_xy_CL_ql.npy')
u_xy_CL_gql2 = np.load('u_xy_CL_gql2.npy')
u_xy_CL_gql3 = np.load('u_xy_CL_gql3.npy')
u_xy_CL_gql8 = np.load('u_xy_CL_gql8.npy')


def instant_gql3():
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(X, Yh, u_xy_nw_gql3, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Instantaneous NW GQL3')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    fig.savefig('u_xy_nw_gql3_instant')
    return fig.show()

def finer_instant_gql3():
# Define spline on old grid
    interp_spline = RectBivariateSpline(y.T, x.T, u_xy_nw_gql3)
# Interpolate onto finer grid
    f_u_xy_nw_gql3 = interp_spline(y2, x2)
# Plot result
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(fX, fYh, f_u_xy_nw_gql3, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Finer Instantaneous NW GQL3')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('f_u_xy_nw_gql3_instant')
    np.save('f_u_xy_nw_gql3', f_u_xy_nw_gql3)
    return fig.show()

def instant_gql2():

    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(X, Yh, u_xy_nw_gql2, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Instantaneous NW GQL2')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    fig.savefig('u_xy_nw_gql2_instant')
    return fig.show()

def finer_instant_gql2():
# Define spline on old grid
    interp_spline = RectBivariateSpline(y.T, x.T, u_xy_nw_gql2)
# Interpolate onto finer grid
    f_u_xy_nw_gql2 = interp_spline(y2, x2)
# Plot result
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(fX, fYh, f_u_xy_nw_gql2, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Finer Instantaneous NW GQL2')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('f_u_xy_nw_gql2_instant')
    np.save('f_u_xy_nw_gql2', f_u_xy_nw_gql2)
    return fig.show()

def instant_nl():

    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(X, Yh, u_xy_nw_nl, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Instantaneous NW NL')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('u_xy_nw_nl_instant')
    return fig.show()

def finer_instant_nl():
#  Define spline on old grid
    interp_spline = RectBivariateSpline(y.T, x.T, u_xy_nw_nl)
# Interpolate onto finer grid
    f_u_xy_nw_nl = interp_spline(y2, x2)
# Plot result
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(fX, fYh, f_u_xy_nw_nl, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Finer Instantaneous NW NL')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('f_u_xy_nw_nl_instant')
    np.save('f_u_xy_nw_nl', f_u_xy_nw_nl)
    return fig.show()

def instant_ql():
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(X, Yh, u_xy_nw_ql, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Instantaneous NW QL')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    fig.savefig('u_xy_ql_instant')
    return fig.show()

def finer_instant_ql():
# Define spline on old grid
    interp_spline = RectBivariateSpline(y.T, x.T, u_xy_nw_ql)
# Interpolate using finer grid
    f_u_xy_nw_ql = interp_spline(y2, x2)
# Plot result
    fig, ax = plt.subplots()
    plt2 = ax.pcolormesh(fX, fYh, f_u_xy_nw_ql, cmap=cmap)
    cb = fig.colorbar(plt2)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=ticksize)
    ax.set_title('Finer Instantaneous NW QL')
    ax.set_ylabel('y', fontsize=fsize, rotation = 1)
    ax.set_xlabel('x', fontsize=fsize)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig('f_u_xy_nw_ql_instant')
    np.save('f_u_xy_nw_ql', f_u_xy_nw_ql)
    return fig.show()

def finer_all():                # create finer grid arrays, no plots
## INSTANTANEOUS
# Near Wall files
    is_nw_nl = RectBivariateSpline(y.T, x.T, u_xy_nw_nl)
    is_nw_ql = RectBivariateSpline(y.T, x.T, u_xy_nw_ql)
    is_nw_gql2 = RectBivariateSpline(y.T, x.T, u_xy_nw_gql2)
    is_nw_gql3 = RectBivariateSpline(y.T, x.T, u_xy_nw_gql3)
    is_nw_gql8 = RectBivariateSpline(y.T, x.T, u_xy_nw_gql8)
# Near CL fles
    is_cl_nl = RectBivariateSpline(y.T, x.T, u_xy_cl_nl)
    is_cl_ql = RectBivariateSpline(y.T, x.T, u_xy_cl_ql)
    is_cl_gql2 = RectBivariateSpline(y.T, x.T, u_xy_cl_gql2)
    is_cl_gql3 = RectBivariateSpline(y.T, x.T, u_xy_cl_gql3)
    is_cl_gql8 = RectBivariateSpline(y.T, x.T, u_xy_cl_gql8)
# Compute finer grid, near wall
    f_u_xy_nw_nl = is_nw_nl(y2, x2)
    f_u_xy_nw_ql = is_nw_ql(y2, x2)
    f_u_xy_nw_gql2 = is_nw_gql2(y2, x2)
    f_u_xy_nw_gql3 = is_nw_gql3(y2, x2)
    f_u_xy_nw_gql8 = is_nw_gql8(y2, x2)
# Compute finer grid, near CL
    f_u_xy_cl_nl = is_cl_nl(y2, x2)
    f_u_xy_cl_ql = is_cl_ql(y2, x2)
    f_u_xy_cl_gql2 = is_cl_gql2(y2, x2)
    f_u_xy_cl_gql3 = is_cl_gql3(y2, x2)
    f_u_xy_cl_gql8 = is_cl_gql8(y2, x2)
# Save files
    np.save('f_u_xy_nw_nl', f_u_xy_nw_nl)
    np.save('f_u_xy_nw_ql', f_u_xy_nw_ql)
    np.save('f_u_xy_nw_gql2', f_u_xy_nw_gql2)
    np.save('f_u_xy_nw_gql3', f_u_xy_nw_gql3)
    np.save('f_u_xy_nw_gql8', f_u_xy_nw_gql8)
# Compute finer grid, near CL
    np.save('f_u_xy_cl_nl', f_u_xy_cl_nl)
    np.save('f_u_xy_cl_ql', f_u_xy_cl_ql)
    np.save('f_u_xy_cl_gql2', f_u_xy_cl_gql2)
    np.save('f_u_xy_cl_gql3', f_u_xy_cl_gql3)
    np.save('f_u_xy_cl_gql8', f_u_xy_cl_gql8)
## TIME/STREAMWISE AVERAGES
# Near Wall files
    is_NW_nl = RectBivariateSpline(y.T, x.T, u_xy_NW_nl)
    is_NW_ql = RectBivariateSpline(y.T, x.T, u_xy_NW_ql)
    is_NW_gql2 = RectBivariateSpline(y.T, x.T, u_xy_NW_gql2)
    is_NW_gql3 = RectBivariateSpline(y.T, x.T, u_xy_NW_gql3)
    is_NW_gql8 = RectBivariateSpline(y.T, x.T, u_xy_NW_gql8)
# Near CL fles
    is_CL_nl = RectBivariateSpline(y.T, x.T, u_xy_CL_nl)
    is_CL_ql = RectBivariateSpline(y.T, x.T, u_xy_CL_ql)
    is_CL_gql2 = RectBivariateSpline(y.T, x.T, u_xy_CL_gql2)
    is_CL_gql3= RectBivariateSpline(y.T, x.T, u_xy_CL_gql3)
    is_CL_gql8 = RectBivariateSpline(y.T, x.T, u_xy_CL_gql8)
# Compute finer grid, near wall
    f_u_xy_NW_nl = is_NW_nl(y2, x2)
    f_u_xy_NW_ql = is_NW_ql(y2, x2)
    f_u_xy_NW_gql2 = is_NW_gql2(y2, x2)
    f_u_xy_NW_gql3 = is_NW_gql3(y2, x2)
    f_u_xy_NW_gql8 = is_NW_gql8(y2, x2)
# Compute finer grid, near CL
    f_u_xy_CL_nl = is_CL_nl(y2, x2)
    f_u_xy_CL_ql = is_CL_ql(y2, x2)
    f_u_xy_CL_gql2 = is_CL_gql2(y2, x2)
    f_u_xy_CL_gql3 = is_CL_gql3(y2, x2)
    f_u_xy_CL_gql8 = is_CL_gql8(y2, x2)
# Save files
    np.save('f_u_xy_NW_nl', f_u_xy_NW_nl)
    np.save('f_u_xy_NW_ql', f_u_xy_NW_ql)
    np.save('f_u_xy_NW_gql2', f_u_xy_NW_gql2)
    np.save('f_u_xy_NW_gql3', f_u_xy_NW_gql3)
    np.save('f_u_xy_NW_gql8', f_u_xy_NW_gql8)
# Compute finer grid, near CL
    np.save('f_u_xy_CL_nl', f_u_xy_CL_nl)
    np.save('f_u_xy_CL_ql', f_u_xy_CL_ql)
    np.save('f_u_xy_CL_gql2', f_u_xy_CL_gql2)
    np.save('f_u_xy_CL_gql3', f_u_xy_CL_gql3)
    np.save('f_u_xy_CL_gql8', f_u_xy_CL_gql8)

def fcompare_instant(nx):
    ld = nx / 2
    f_u_xy_NW_nl = np.load('f_u_xy_nw_nl.npy')
    f_u_xy_NW_gql3 = np.load('f_u_xy_nw_gql3.npy')
    f_u_xy_NW_gql2 = np.load('f_u_xy_nw_gql2.npy')
    f_u_xy_NW_ql = np.load('f_u_xy_nw_ql.npy')
    u_xy_CL_nl = np.load('f_u_xy_cl_nl.npy')
    u_xy_CL_gql3 = np.load('f_u_xy_cl_gql3.npy')
    u_xy_CL_gql2 = np.load('f_u_xy_cl_gql2.npy')
    u_xy_CL_ql = np.load('f_u_xy_cl_ql.npy')
    # Define max/mins for shared colorbar
    combined_u_xy_NW = np.array([f_u_xy_NW_ql, f_u_xy_NW_gql3, f_u_xy_NW_gql2, f_u_xy_NW_nl])
    u_xy_NW_min, u_xy_NW_max = np.amin(combined_u_xy_NW), np.amax(combined_u_xy_NW)
    combined_u_xy_CL = np.array([u_xy_CL_ql, u_xy_CL_gql3, u_xy_CL_gql2, u_xy_CL_nl])  # leave QL out of this for now (or forever)
    u_xy_CL_min, u_xy_CL_max = np.amin(combined_u_xy_CL), np.amax(combined_u_xy_CL)

    fig = plt.figure(7, figsize=(15,8))
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
    plt00 = ax00.pcolormesh(fX, fYh, np.fliplr(f_u_xy_NW_ql), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)     # name to create mappable for colorbar
    plt01 = ax01.pcolormesh(fX, fYh, np.fliplr(f_u_xy_NW_gql2), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)
    plt02 = ax02.pcolormesh(fX, fYh, np.fliplr(f_u_xy_NW_gql3), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)  # set vmax/vmin to normalize cbar
    plt03 = ax03.pcolormesh(fX, fYh, np.fliplr(f_u_xy_NW_nl), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap=cmap)
    plt10 = ax10.pcolormesh(fX, fYh, np.fliplr(u_xy_CL_ql), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)      # name to create mappable for colorbar
    plt11 = ax11.pcolormesh(fX, fYh, np.fliplr(u_xy_CL_gql2), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)   # name to create mappable for colorbar
    plt12 = ax12.pcolormesh(fX, fYh, np.fliplr(u_xy_CL_gql3), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)
    plt13 = ax13.pcolormesh(fX, fYh, np.fliplr(u_xy_CL_nl), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap=cmap)
    # Labels
    # fig.suptitle('instantanteous Velocity, Near Wall vs. near CL', size=18)
    ax00.set_title('$\Lambda_x$ = 0 (QL)', size=14)
    ax00.set_ylabel('y', size=fsize, rotation=0)
    ax00.tick_params(axis='both', which='major', labelsize=ticksize)
    ax00.set_xlabel('x', size=fsize)
    ax01.set_title('$\Lambda_x$ = 2 (GQL)', size=fsize)
    ax01.set_xlabel('x', size=fsize)
    ax01.tick_params(axis='both', which='major', labelsize=ticksize)
    ax02.set_title('$\Lambda_x$ = 3 (GQL)', size=fsize)
    ax02.set_xlabel('x', size=fsize)
    ax02.tick_params(axis='both', which='major', labelsize=ticksize)
    ax03.set_title('$\Lambda_x$ = %i (NL)' %ld, size=fsize)
    ax03.set_xlabel('x', size=fsize)
    ax03.tick_params(axis='both', which='major', labelsize=ticksize)
    ax10.set_ylabel('y', size=fsize, rotation=0)
    ax10.set_xlabel('x', size=fsize)
    ax10.tick_params(axis='both', which='major', labelsize=ticksize)
    ax11.set_xlabel('x', size=fsize)
    ax11.tick_params(axis='both', which='major', labelsize=ticksize)
    ax12.set_xlabel('x', size=fsize)
    ax12.tick_params(axis='both', which='major', labelsize=ticksize)
    ax13.set_xlabel('x', size=fsize)
    ax13.tick_params(axis='both', which='major', labelsize=ticksize)
    # Set colorbars
    cbax04 = plt.subplot(gs[0, 4])
    cbax14 = plt.subplot(gs[1, 4])
    cb04 = Colorbar(ax=cbax04, mappable=plt01, orientation='vertical', ticklocation='right')
    cb14 = Colorbar(ax=cbax14, mappable=plt10, orientation='vertical', ticklocation='right')
    cb04.ax.set_yticklabels(cb04.ax.get_yticklabels(), fontsize=ticksize)
    cb04.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cb14.ax.set_yticklabels(cb14.ax.get_yticklabels(), fontsize=ticksize)
    cb14.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Housekeeping
    plt.setp(ax01.get_yticklabels(), visible=False)      # Turn off ticks for y axes
    plt.setp(ax02.get_yticklabels(), visible=False)
    plt.setp(ax03.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)
    plt.setp(ax13.get_yticklabels(), visible=False)
    # plt.setp(ax75.get_yticklabels(), visible=False)
    # plt.setp(ax76.get_yticklabels(), visible=False)
    fig.savefig('velocity_f_compare.png')
    return fig.show()

def compare(nx):
    ld = nx / 2
    u_xy_NW_nl = np.load('u_xy_NW_nl.npy')
    u_xy_NW_gql3 = np.load('u_xy_NW_gql3.npy')
    u_xy_NW_gql2 = np.load('u_xy_NW_gql2.npy')
    u_xy_NW_ql = np.load('u_xy_NW_ql.npy')
    u_xy_CL_nl = np.load('u_xy_CL_nl.npy')
    u_xy_CL_gql3 = np.load('u_xy_CL_gql3.npy')
    u_xy_CL_gql2 = np.load('u_xy_CL_gql2.npy')
    u_xy_CL_ql = np.load('u_xy_CL_ql.npy')
    # Define max/mins for shared colorbar
    combined_u_xy_NW = np.array([u_xy_NW_ql, u_xy_NW_gql3, u_xy_NW_gql2, u_xy_NW_nl])
    u_xy_NW_min, u_xy_NW_max = np.amin(combined_u_xy_NW), np.amax(combined_u_xy_NW)
    combined_u_xy_CL = np.array([u_xy_CL_ql, u_xy_CL_gql3, u_xy_CL_gql2, u_xy_CL_nl])  # leave QL out of this for now (or forever)
    u_xy_CL_min, u_xy_CL_max = np.amin(combined_u_xy_CL), np.amax(combined_u_xy_CL)

    fig = plt.figure(7, figsize=(15,8))
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
    plt00 = ax00.contourf(X, Yh, np.flipud(u_xy_NW_ql), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)     # name to create mappable for colorbar
    plt01 = ax01.contourf(X, Yh, np.flipud(u_xy_NW_gql2), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)
    plt02 = ax02.contourf(X, Yh, np.flipud(u_xy_NW_gql3), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)  # set vmax/vmin to normalize cbar
    plt03 = ax03.contourf(X, Yh, np.flipud(u_xy_NW_nl), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap=cmap)
    plt10 = ax10.contourf(X, Yh, np.flipud(u_xy_CL_ql), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)      # name to create mappable for colorbar
    plt11 = ax11.contourf(X, Yh, np.flipud(u_xy_CL_gql2), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)   # name to create mappable for colorbar
    plt12 = ax12.contourf(X, Yh, np.flipud(u_xy_CL_gql3), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)
    plt13 = ax13.contourf(X, Yh, np.flipud(u_xy_CL_nl), vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap=cmap)
    # Labels
    # fig.suptitle('Streamwise Fluctuations of Time-Averaged Velocity, Near Wall', size=18)
    ax00.set_title('$\Lambda_x$ = 0 (QL)', size=14)
    ax00.set_ylabel('y', size=fsize, rotation=0)
    ax00.tick_params(axis='both', which='major', labelsize=ticksize)
    ax00.set_xlabel('x', size=fsize)
    ax01.set_title('$\Lambda_x$ = 2 (GQL)', size=fsize)
    ax01.set_xlabel('x', size=fsize)
    ax01.tick_params(axis='both', which='major', labelsize=ticksize)
    ax02.set_title('$\Lambda_x$ = 3 (GQL)', size=fsize)
    ax02.set_xlabel('x', size=fsize)
    ax02.tick_params(axis='both', which='major', labelsize=ticksize)
    ax03.set_title('$\Lambda_x$ = %i (NL)' %ld, size=fsize)
    ax03.set_xlabel('x', size=fsize)
    ax03.tick_params(axis='both', which='major', labelsize=ticksize)
    ax10.set_ylabel('y', size=fsize, rotation=0)
    ax10.set_xlabel('x', size=fsize)
    ax10.tick_params(axis='both', which='major', labelsize=ticksize)
    ax11.set_xlabel('x', size=fsize)
    ax11.tick_params(axis='both', which='major', labelsize=ticksize)
    ax12.set_xlabel('x', size=fsize)
    ax12.tick_params(axis='both', which='major', labelsize=ticksize)
    ax13.set_xlabel('x', size=fsize)
    ax13.tick_params(axis='both', which='major', labelsize=ticksize)
    # Set colorbars
    cbax04 = plt.subplot(gs[0, 4])
    cbax14 = plt.subplot(gs[1, 4])
    cb04 = Colorbar(ax=cbax04, mappable=plt01, orientation='vertical', ticklocation='right')
    cb14 = Colorbar(ax=cbax14, mappable=plt10, orientation='vertical', ticklocation='right')
    cb04.ax.set_yticklabels(cb04.ax.get_yticklabels(), fontsize=ticksize)
    cb04.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cb14.ax.set_yticklabels(cb14.ax.get_yticklabels(), fontsize=ticksize)
    cb14.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Housekeeping
    plt.setp(ax01.get_yticklabels(), visible=False)      # Turn off ticks for y axes
    plt.setp(ax02.get_yticklabels(), visible=False)
    plt.setp(ax03.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)
    plt.setp(ax13.get_yticklabels(), visible=False)
    # plt.setp(ax75.get_yticklabels(), visible=False)
    # plt.setp(ax76.get_yticklabels(), visible=False)
    fig.savefig('velocity_compare.png')
    return fig.show()

def fcompare(nx):
    ld = nx / 2
    u_xy_NW_nl = np.load('f_u_xy_NW_nl.npy')
    u_xy_NW_gql3 = np.load('f_u_xy_NW_gql3.npy')
    u_xy_NW_gql2 = np.load('f_u_xy_NW_gql2.npy')
    u_xy_NW_ql = np.load('f_u_xy_NW_ql.npy')
    u_xy_CL_nl = np.load('f_u_xy_CL_nl.npy')
    u_xy_CL_gql3 = np.load('f_u_xy_CL_gql3.npy')
    u_xy_CL_gql2 = np.load('f_u_xy_CL_gql2.npy')
    u_xy_CL_ql = np.load('f_u_xy_CL_ql.npy')
    # Define max/mins for shared colorbar
    combined_u_xy_NW = np.array([u_xy_NW_ql, u_xy_NW_gql3, u_xy_NW_gql2, u_xy_NW_nl])
    u_xy_NW_min, u_xy_NW_max = np.amin(combined_u_xy_NW), np.amax(combined_u_xy_NW)
    combined_u_xy_CL = np.array([u_xy_CL_ql, u_xy_CL_gql3, u_xy_CL_gql2, u_xy_CL_nl])  # leave QL out of this for now (or forever)
    u_xy_CL_min, u_xy_CL_max = np.amin(combined_u_xy_CL), np.amax(combined_u_xy_CL)

    fig = plt.figure(7, figsize=(15,8))
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
    plt00 = ax00.pcolormesh(fX, fYh, np.flipud(u_xy_NW_ql), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)     # name to create mappable for colorbar
    plt01 = ax01.pcolormesh(fX, fYh, np.flipud(u_xy_NW_gql2), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)
    plt02 = ax02.pcolormesh(fX, fYh, np.flipud(u_xy_NW_gql3), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap = cmap)  # set vmax/vmin to normalize cbar
    plt03 = ax03.pcolormesh(fX, fYh, np.flipud(u_xy_NW_nl), vmax=u_xy_NW_max, vmin=u_xy_NW_min, cmap=cmap)
    plt10 = ax10.pcolormesh(fX, fYh, u_xy_CL_ql, vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)      # name to create mappable for colorbar
    plt11 = ax11.pcolormesh(fX, fYh, u_xy_CL_gql2, vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)   # name to create mappable for colorbar
    plt12 = ax12.pcolormesh(fX, fYh, u_xy_CL_gql3, vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap = cmap)
    plt13 = ax13.pcolormesh(fX, fYh, u_xy_CL_nl, vmax=u_xy_CL_max, vmin=u_xy_CL_min, cmap=cmap)
    # Labels
    # fig.suptitle('Streamwise Fluctuations of Time-Averaged Velocity, Near Wall', size=18)
    ax00.set_title('$\Lambda_x$ = 0 (QL)', size=14)
    ax00.set_ylabel('y', size=fsize, rotation=0)
    ax00.tick_params(axis='both', which='major', labelsize=ticksize)
    ax00.set_xlabel('x', size=fsize)
    ax01.set_title('$\Lambda_x$ = 2 (GQL)', size=fsize)
    ax01.set_xlabel('x', size=fsize)
    ax01.tick_params(axis='both', which='major', labelsize=ticksize)
    ax02.set_title('$\Lambda_x$ = 3 (GQL)', size=fsize)
    ax02.set_xlabel('x', size=fsize)
    ax02.tick_params(axis='both', which='major', labelsize=ticksize)
    ax03.set_title('$\Lambda_x$ = %i (NL)' %ld, size=fsize)
    ax03.set_xlabel('x', size=fsize)
    ax03.tick_params(axis='both', which='major', labelsize=ticksize)
    ax10.set_ylabel('y', size=fsize, rotation=0)
    ax10.set_xlabel('x', size=fsize)
    ax10.tick_params(axis='both', which='major', labelsize=ticksize)
    ax11.set_xlabel('x', size=fsize)
    ax11.tick_params(axis='both', which='major', labelsize=ticksize)
    ax12.set_xlabel('x', size=fsize)
    ax12.tick_params(axis='both', which='major', labelsize=ticksize)
    ax13.set_xlabel('x', size=fsize)
    ax13.tick_params(axis='both', which='major', labelsize=ticksize)
    # Set colorbars
    cbax04 = plt.subplot(gs[0, 4])
    cbax14 = plt.subplot(gs[1, 4])
    cb04 = Colorbar(ax=cbax04, mappable=plt01, orientation='vertical', ticklocation='right')
    cb14 = Colorbar(ax=cbax14, mappable=plt10, orientation='vertical', ticklocation='right')
    cb04.ax.set_yticklabels(cb04.ax.get_yticklabels(), fontsize=ticksize)
    cb04.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cb14.ax.set_yticklabels(cb14.ax.get_yticklabels(), fontsize=ticksize)
    cb14.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # Housekeeping
    plt.setp(ax01.get_yticklabels(), visible=False)      # Turn off ticks for y axes
    plt.setp(ax02.get_yticklabels(), visible=False)
    plt.setp(ax03.get_yticklabels(), visible=False)
    plt.setp(ax11.get_yticklabels(), visible=False)
    plt.setp(ax12.get_yticklabels(), visible=False)
    plt.setp(ax13.get_yticklabels(), visible=False)
    # plt.setp(ax75.get_yticklabels(), visible=False)
    # plt.setp(ax76.get_yticklabels(), visible=False)
    fig.savefig('velocity_compare.png')
    return fig.show()