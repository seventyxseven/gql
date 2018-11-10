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

plt.close('all')

Lx, Ly, Lz = (2.*np.pi, np.pi, 1.)
Reynolds = 200.

nx,ny,nz=(84, 84, 108)  # Number of modes in each direction

x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier("y", ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev("z", nz, interval=(-Lz, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Unscaled grid (physical space)
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

X, Y = np.meshgrid(x, y, indexing='ij')
Y2, Z = np.meshgrid(y, z, indexing='ij')
kx, ky = np.meshgrid(x_basis.wavenumbers, y_basis.wavenumbers,  indexing='ij')
x2, ky2 = np.meshgrid(x, y_basis.wavenumbers,  indexing='ij')
nkx, nky = kx.shape

ff = h5py.File('slices_gqlss/slices.h5', 'r')
gg = h5py.File('dump_gqlss/dump.h5', 'r')
u_xy_cl = ff['tasks']['u z 0.1'][:]       # u in xy plane, z=0.1 (near centerline)
u_xy_nw = ff['tasks']['u z 0.95'][:]
uk_xy_NW = ff['tasks']['u z 0.95']
uvert_all = ff['tasks']['u vertical'][:]
vvert_all =ff['tasks']['v vertical'][:]
wvert_all =ff['tasks']['w vertical'][:]
uh_xy_cl = ff['tasks']['uh z 0.1'][:]
uzvert_all = ff['tasks']['uz vertical'][:]
# ulz_all = gg['tasks']['ulz'][:]
ul_dump = gg['tasks']['ul'][:]
vl_dump = gg['tasks']['vl'][:]
wl_dump = gg['tasks']['wl'][:]


tslices = u_xy_nw.shape[0]
t_turb = int(round(4/8*tslices))
tdump = ul_dump.shape[0]
turb_dump = int(round(5/11*tdump))
Nt1 = tdump - turb_dump
Nt0 = tslices - t_turb
Ntstep = t_turb

c = -1
umeanx = np.empty((ny, nz, turb_dump))
for k in range(Nt1, tdump):
    ul_all = ul_dump[k, :, :, :]
    umeanx_all = np.mean(ul_all, 0)
    c = c + 1
    umeanx[:, :, c] = umeanx_all
umeanxt = np.mean(umeanx, 2)

d = -1
up = np.empty((nx, ny, nz, turb_dump))
for l in range(Nt1, tdump):
    ul_box = ul_dump[l, :, :, :]
    up_box = ul_box - umeanxt
    d = d + 1
    up[:, :, :, d] = up_box

uzmeant = np.mean(uzvert_all[Nt0:tslices, 0, :, :], 0)
uzmeanty = np.mean(uzmeant[:, :], 0)
umeanvt = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
umeanvty = np.mean(umeanvt[:, :], 0)
print('min <u>ty:', min(umeanvty))
print('min <du/dz>ty:', min(uzmeanty))
u_tau = np.sqrt(-uzmeanty[0]/Reynolds)
print('u_tau:', u_tau)
np.save('u_tau_gql', u_tau)

count= -1;
umeant = np.mean(u_xy_nw[Nt0:tslices, :, :, 0], 0)
ek_NW_x = np.empty((nx, ny, Ntstep))
ek_NW_y = np.empty((nx, ny, Ntstep))
ek_NW_xy = np.empty((nx, ny, Ntstep))
for ii in range(Nt0, tslices):
    #print(ii)
    u_xy_NW = u_xy_nw[ii, :, :, 0]
    u_xy_CL = u_xy_cl[ii, :, :, 0]
    u_xy_NW_t = u_xy_NW - umeant
    u_xy_CL_t = u_xy_CL - umeant
    umeanx_NW = np.mean(u_xy_NW_t, 0)
    umeanx_CL = np.mean(u_xy_CL_t, 0)
    u_xy_NW_tx = u_xy_NW_t - umeanx_NW
    u_xy_CL_tx = u_xy_CL_t - umeanx_CL
    umeany_NW = np.mean(u_xy_NW_tx, 1)
    umeany_CL = np.mean(u_xy_CL_tx, 1)
    u_xy_NW_end = u_xy_NW_tx - umeany_NW
    u_xy_CL_end = u_xy_CL_tx - umeany_CL

    # 1D FFTs
    count = count + 1
    uk_NW_x = np.fft.fft(u_xy_NW_end, nx, 0) / nx  # axis 0:  x
    uk_NW_y = np.fft.fft(u_xy_NW_end, ny, 1) / ny  # axis 1:  y
    # uk_NW_xyman = np.fft.fft(uk_NW_x, ny, 1)/ny         # 2 1D FFTs
    uk_NW_xy = np.fft.fft2(u_xy_NW_end, s=[nx, ny]) / (nx * ny)  # FFT2
    ek_NW_x[:, :, count] = 0.5 * np.real(np.multiply(uk_NW_x, np.conjugate(uk_NW_x)))
    ek_NW_y[:, :, count] = 0.5 * np.real(np.multiply(uk_NW_y, np.conjugate(uk_NW_y)))
    ek_NW_xy[:, :, count] = 0.5 * np.real(np.multiply(uk_NW_xy, np.conjugate(uk_NW_xy)))

ek_x_timeavg = np.mean(ek_NW_x, 2)
ek_y_timeavg = np.mean(ek_NW_y, 2)
ek_xy_timeavg = np.mean(ek_NW_xy, 2)
ek_x_timeavg2 = ek_x_timeavg[1:nkx, :] + np.flipud(ek_x_timeavg[nkx + 1:, :])
ek_y_timeavg2 = ek_y_timeavg[:, 1:int(ny / 2)] + np.fliplr(ek_y_timeavg[:, int(ny / 2) + 1:])
ek_xy_timeavg2 = 2 * ek_xy_timeavg[1:nkx, 1:int(ny / 2)] + np.flipud(ek_xy_timeavg[nkx + 1:, 1:int(ny / 2)]) + np.fliplr(ek_xy_timeavg[1:nkx, int(ny / 2) + 1:])

np.save('u_xy_CL_gql', u_xy_CL_end)
np.save('uvert_gql', uvert_all)
np.save('ek_xy_gql', ek_xy_timeavg2)
np.save('ek_x_gql', ek_x_timeavg2)
np.save('ek_y_gql', ek_y_timeavg2)
np.save('u_xy_gql', u_xy_nw)
# np.save('u_xy_gql', u_xy_NW_end)
np.save('u_xy_gql_avg', u_xy_NW_end)
np.save('vertuprofile_avg_gql', umeanvty)
np.save('uz_gql', uzmeanty)

# Plots
cmap = plt.get_cmap('viridis')

# # Plot instantaneous velocity slice (confirm turbulence)
fig, ax = plt.subplots()
plt2=ax.pcolormesh(X, Y, u_xy_nw[tslices-1,:,:,0], cmap=cmap)
fig.colorbar(plt2)
ax.set_title('Instantaneous Streamwise Velocity (x-y)')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('x', fontsize=14)
# fig.savefig('u_xy_.png')

# fig.show()

# # Check energy in high modes
fig, ax = plt.subplots()
plt2=ax.pcolormesh(X, Y, uh_xy_cl[47,:,:,0], cmap=cmap)
fig.colorbar(plt2)
ax.set_title('Streamwise Velocity (high modes only) (QL-QL CP)')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('x', fontsize=14)
fig.savefig('uh_gql.png')

# fig.show()

# # Plot averaged velocity slice (x-y plane)
fig, ax = plt.subplots()
plt2=ax.pcolormesh(X, Y, u_xy_NW_end, cmap=cmap)
fig.colorbar(plt2)
ax.set_title('Time-Averaged Streamwise Velocity (x-y)')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('x', fontsize=14)
# fig.savefig('u_xy.png')

# fig.show()
#

#Plot du/dz (time, spanwise averaged)
fig, ax = plt.subplots()
plt.plot(uzmeanty,z[0,0,:])
ax.set_title('Wall-normal Streamwise Velocity Gradient')
ax.set_ylabel('z', fontsize=14)
ax.set_xlabel('<du/dz>ty', fontsize=14)
fig.savefig('uz_ql.png')

# fig.show()


uzmeany = np.mean(uzvert_all[:, 0, :, :], 1)

fig = plt.figure(figsize=(10,8))
plt5 = fig.add_subplot(111)
for i, timeslice in enumerate (uzmeany[t_turb:tslices]):        # Select rows
    plt5.plot(timeslice[:], z[0, 0, :], label='Slice {0}'.format(i+1))
    plt5.set_title('Velocity Gradient du/dz')
    plt5.set_ylabel('z', fontsize=14)
    plt5.set_xlabel('<u>y', fontsize=14)
fig.savefig('gradients_gql.png')
# fig.show()


uvmeany = np.mean(uvert_all[:, 0, :, :], 1)

fig = plt.figure(figsize=(10,8))
plt4 = fig.add_subplot(111)
for i, timeslice in enumerate (uvmeany[t_turb:tslices]):        # Select rows
    plt4.plot(timeslice[:], z[0, 0, :], label='Slice {0}'.format(i+1))
    plt4.set_title('Streamwise Velocity (x-z plane)')
    plt4.set_ylabel('z', fontsize=14)
    plt4.set_xlabel('<u>y', fontsize=14)
fig.savefig('vertuprofile_inst_gql.png')
# fig.show()


fig, ax = plt.subplots()
plt.plot(umeanvty, z[0, 0, :])
ax.set_title('Average Streamwise Velocity Profile (x-z plane)')
ax.set_ylabel('z', fontsize=14)
ax.set_xlabel('<u>yt', fontsize=14)
fig.savefig('vertuprofile_avg_gql.png')

# fig.show()

U_plus = uvert_all[:, 0, :, :]/u_tau
U_py = np.mean(U_plus, 1)
U_pyt = np.abs(np.flipud(np.mean(U_py[Nt0:tslices, nz/2:], 0)))
# Upyt = np.abs(np.pad(U_pyt, (1,0), 'constant'))
z_p = z[0, 0, nz/2:]*Reynolds
# z_p = np.pad(z_plus, (1,0), 'constant')
np.save('Upyt_gql', U_pyt)

uvmeant = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
vvmeant = np.mean(vvert_all[Nt0:tslices, 0, :, :], 0)
wvmeant = np.mean(wvert_all[Nt0:tslices, 0, :, :], 0)
upvert = np.empty((ny, nz, Ntstep))
upvertsq = np.empty((ny, nz, Ntstep))
counter = -1
for jj in range(Nt0, tslices):
    uvert_slices = uvert_all[jj, 0, :, :]
    counter = counter + 1
    upvert[:, :, counter] = (uvert_slices - uvmeant)
    upvertsq[:, :, counter] = (uvert_slices - uvmeant)**2

upvertsq_yavg = np.mean(upvertsq, 0)
upvertsq_ytavg = np.mean(upvertsq_yavg, 1)
upsq_yt = np.flipud(upvertsq_ytavg[nz/2:])


# Velocity Fluctuations using x = 0 slice (many more time slices but not x averaged)
fig, ax = plt.subplots()
plt.loglog(z_p, upsq_yt)
ax.set_title('Velocity Fluctuations Comparison (slices)')
ax.set_xlabel('$z^+$', fontsize=14)
ax.set_ylabel('$[u^{''+}]_{y,t}$', fontsize=14)
fig.savefig('fluctuations.png')
ax.set_xscale('log')
ax.set_yscale('log')
fig.show()
np.save('upsq_yt_gql', upsq_yt)

up_sq = up**2
upsq_xmean = np.mean(up_sq, 0)
upsq_xymean = np.mean(upsq_xmean, 0)
upsq_xytmean = np.mean(upsq_xymean, 1)
upsq_xyt = np.flipud(upsq_xytmean[nz/2:])

# Velocity Fluctuations using dump file (fewer time slices, but actually x averaged)
fig, ax = plt.subplots()
plt.loglog(z_p, upsq_xyt)
ax.set_title('Velocity Fluctuations Comparison (dump)')
ax.set_xlabel('$z^+$', fontsize=14)
ax.set_ylabel('$[u^{''+}]_{x,y,t}$', fontsize=14)
fig.savefig('fluctuations.png')
ax.set_xscale('log')
ax.set_yscale('log')
# fig.show()
np.save('upsq_xyt_gql', upsq_xyt)

fig = plt.figure(figsize=(10,8))
plt5 = fig.add_subplot(111)
for i, timeslice in enumerate (ek_y_timeavg2[:]):        # i, timeslice:  iterable, # of columns
    plt5.plot(ky2[0, 1:int(ny/2)], timeslice[:], label='Slice {0}'.format(i+1))    # plot each row
    plt5.set_title('Ey (by mode)')
    plt5.set_ylabel('Ey', fontsize=14)
    plt5.set_xlabel('ky', fontsize=14)
    plt5.set_yscale('log')
fig.savefig('Ey_modes_gql.png')
# fig.show()

upvert_plot = upvert[:, 0:nz/2, -1]
# up_plot = np.transpose(upvert_plot)
fig, ax = plt.subplots()
velslice = ax.pcolormesh(Y2[:, 0:nz/2], Z[:, 0:nz/2], upvert_plot, cmap=cmap)
fig.colorbar(velslice)
# ax.set_title('Perturbation Structure')
ax.set_xlabel('y', fontsize=14)
ax.set_ylabel('z', fontsize=14, rotation = 0)
# ax.set_yscale('log')
fig.savefig('vert_perturb_gql.png')
fig.show()