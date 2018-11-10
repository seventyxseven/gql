import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import h5py
import time
from IPython import display
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools

def grid(Lx, Ly, Lz, nx, ny, nz):
    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
    y_basis = de.Fourier("y", ny, interval=(0, Ly), dealias=3 / 2)
    z_basis = de.Chebyshev("z", nz, interval=(-Lz, Lz), dealias=3 / 2)
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    # Unscaled grid (physical space)
    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)

    X, Y = np.meshgrid(x, y, indexing='ij')
    Y2, Z = np.meshgrid(y, z, indexing='ij')
    kx, ky = np.meshgrid(x_basis.wavenumbers, y_basis.wavenumbers, indexing='ij')
    x2, ky2 = np.meshgrid(x, y_basis.wavenumbers, indexing='ij')
    nkx, nky = kx.shape

def files():
    # Load Stats npy files

    # # Turbulent stats
    # # u_tau_dns = np.load('u_tau_dns.npy')
    # u_tau_gql = np.load('u_tau_gql.npy')
    # # u_tau_ql = np.load('u_tau_ql.npy')
    #
    # # velocities for xy slices
    # # near wall (z=0.95)
    # u_xy_nw_dns = np.load('u_xy_nw_dns.npy')
    # u_xy_nw_ss_dns = np.load('u_xy_dns_avg.npy')
    u_xy_nw_gql = np.load('u_xy_nw_gql.npy')
    return u_xy_nw_gql
    # u_xy_nw_ss_gql = np.load('u_xy_gql_avg.npy')
    # u_xy_nw_ql = np.load('u_xy_nw_ql.npy')
    # u_xy_nw_ss_ql = np.load('u_xy_ql_avg.npy')
    # # near centerline (z=0.1)
    # u_xy_cl_ss_dns = np.load('u_xy_CL_dns.npy')
    # u_xy_cl_ss_gql = np.load('u_xy_CL_gql.npy')
    # u_xy_cl_ss_ql = np.load('u_xy_CL_ql.npy')
    #
    # # velocity gradient
    # uz_dns = np.load('uz_dns.npy')
    # # uz_gql = np.load('uz_gql.npy')  # uncomment when full DNS run is done
    # uz_ql = np.load('uz_ql.npy')
    # # vertical velocity profiles
    # umeanvty_dns = np.load('vertuprofile_avg_dns.npy')
    # umeanvty_gql = np.load('vertuprofile_avg_gql.npy')
    # umeanvty_ql = np.load('vertuprofile_avg_ql.npy')
    # uvmeany_dns = np.load('uvmeany_dns.npy')
    # uvert_dns = np.load('uvert_dns.npy')
    # uvert_gql = np.load('uvert_gql.npy')
    # uvert_ql = np.load('uvert_ql.npy')
    #
    # # energy for QL/GQL/DNS
    #
    # ek_y_timeavg2_dns = np.load('ek_y_dns.npy')
    # ek_x_timeavg2_dns = np.load('ek_x_dns.npy')
    # ek_xy_timeavg2_dns = np.load('ek_xy_dns.npy')
    # # ek_y_cl_timeavg2_all = np.load('ek_y_cl_all.npy')
    # # ek_x_cl_timeavg2_all = np.load('ek_x_cl_all.npy')
    # # ek_xy_cl_timeavg2_all = np.load('ek_xy_cl_all.npy')
    # ek_y_timeavg2_gql = np.load('ek_y_gql.npy')
    # ek_x_timeavg2_gql = np.load('ek_x_gql.npy')
    # ek_xy_timeavg2_gql = np.load('ek_xy_gql.npy')
    # # ek_y_cl_timeavg2_gql = np.load('ek_y_cl_gql.npy')
    # # ek_x_cl_timeavg2_gql = np.load('ek_x_cl_gql.npy')
    # # ek_xy_cl_timeavg2_gql = np.load('ek_xy_cl_gql.npy')
    # ek_y_timeavg2_ql = np.load('ek_y_ql.npy')
    # ek_x_timeavg2_ql = np.load('ek_x_ql.npy')
    # ek_xy_timeavg2_ql = np.load('ek_xy_ql.npy')
    # # ek_y_cl_timeavg2_ql = np.load('ek_y_cl_ql.npy')
    # # ek_x_cl_timeavg2_ql = np.load('ek_x_cl_ql.npy')
    # # ek_xy_cl_timeavg2_ql = np.load('ek_xy_cl_ql.npy')