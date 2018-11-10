import os
import numpy as np               
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
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

Lx, Ly, Lz = (np.pi, 0.5*np.pi, 1.)
Reynolds = 940.

nx,ny,nz=(128, 128, 256)  # Number of modes in each direction

x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier("y", ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev("z", nz, interval=(-Lz, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Scaled grid
# x = domain.grid(0, scales=domain.dealias)   # [192, 1, 1]
# y = domain.grid(1, scales=domain.dealias)    # [1,192,1]
# z = domain.grid(2, scales=domain.dealias)   # [1,1,192]

# Unscaled grid (physical space)
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

X, Y = np.meshgrid(x, y, indexing='ij')
Y2, Z = np.meshgrid(y, z, indexing='ij')
kx, ky = np.meshgrid(x_basis.wavenumbers, y_basis.wavenumbers,  indexing='ij')
x2, ky2 = np.meshgrid(x, y_basis.wavenumbers,  indexing='ij')
##lx,ly=np.meshgrid(2*np.pi/x_basis.wavenumbers[1:], 2*np.pi/y_basis.wavenumbers[1:-kypos],indexing='ij')
nkx, nky = kx.shape
##nlx,nly=lx.shape
ksq = kx**2+ky**2

np.save('Z_gql', Z)
np.save('Y_gql', Y2)

kypos=np.int((nky+1)/2)

ff= h5py.File('slices/slices.h5', 'r')
u_xy_cl = ff['tasks']['u z 0.1'][:]       # u in xy plane, z=0.1 (near centerline)
u_xy=ff['tasks']['u z 0.95'][:]
uk_xy_NW=ff['tasks']['u z 0.95']
uvert_all=ff['tasks']['u vertical'][:]

#jj= 80
#u_xy_NW=u_xy[jj, :, :, 0]
#umeanxy = np.mean(u_xy_NW, 0)
#u_xy_NW_end = u_xy_NW-umeanxy
count= -1;
Nt0 = 50
Ntstep = 76-Nt0
umeant = np.mean(u_xy[Nt0:Ntstep+Nt0, :, :, 0], 0)
ek_NW_x = np.empty((nx,ny,Ntstep))
ek_NW_y = np.empty((nx,ny,Ntstep))
ek_NW_xy = np.empty((nx,ny,Ntstep))
for ii in range(Nt0, Nt0+Ntstep):
    #print(ii)
    u_xy_NW=u_xy[ii, :, :, 0]
    u_xy_NW_t = u_xy_NW-umeant
    umeanx = np.mean(u_xy_NW_t, 0)
    u_xy_NW_tx = u_xy_NW_t-umeanx
    umeany = np.mean(u_xy_NW_tx, 1)
    u_xy_NW_end = u_xy_NW_tx-umeany
    #u_xy_NW_timeavg = np.mean(u_xy_NW[130:134, :, :, 0], 0)
    #umeanxy = np.mean(u_xy_NW_timeavg, 0)
    #u_xy_NW_end = u_xy_NW_timeavg-umeanxy
    # 1D FFTs
    count = count +1 
    uk_NW_x = np.fft.fft(u_xy_NW_end, nx, 0)/nx        # axis 0:  x
    uk_NW_y = np.fft.fft(u_xy_NW_end, ny, 1)/ny        # axis 1:  y
    #uk_NW_xyman = np.fft.fft(uk_NW_x, ny, 1)/ny         # 2 1D FFTs
    uk_NW_xy = np.fft.fft2(u_xy_NW_end, s=[nx, ny])/(nx*ny)        # FFT2
    ek_NW_x[:,:,count] = 0.5*np.real(np.multiply(uk_NW_x,np.conjugate(uk_NW_x)))
    ek_NW_y[:,:,count] = 0.5*np.real(np.multiply(uk_NW_y,np.conjugate(uk_NW_y)))                 
    ek_NW_xy[:,:,count] = 0.5*np.real(np.multiply(uk_NW_xy,np.conjugate(uk_NW_xy)))
    
ek_x_timeavg = np.mean(ek_NW_x, 2)
ek_y_timeavg = np.mean(ek_NW_y, 2)
ek_xy_timeavg = np.mean(ek_NW_xy, 2)
ek_x_timeavg2 = ek_x_timeavg[1:nkx,:]+np.flipud(ek_x_timeavg[nkx+1:,:])
ek_y_timeavg2 = ek_y_timeavg[:,1:int(ny/2)]+np.fliplr(ek_y_timeavg[:,int(ny/2)+1:])
ek_xy_timeavg2 = 2*ek_xy_timeavg[1:nkx,1:int(ny/2)]+np.flipud(ek_xy_timeavg[nkx+1:,1:int(ny/2)])+np.fliplr(ek_xy_timeavg[1:nkx,int(ny/2)+1:])

count = -1;
Nt0 = 50
Ntstep = 76-Nt0
umeant_cl = np.mean(u_xy_cl[Nt0:Ntstep+Nt0, :, :, 0], 0)
ek_CL_x = np.empty((nx, ny, Ntstep))
ek_CL_y = np.empty((nx, ny, Ntstep))
ek_CL_xy = np.empty((nx, ny, Ntstep))
for ii in range(Nt0, Nt0 + Ntstep):
    #print(ii)
    u_xy_CL=u_xy_cl[ii, :, :, 0]
    u_xy_CL_t = u_xy_CL-umeant_cl
    umeanx_cl = np.mean(u_xy_CL_t, 0)
    u_xy_CL_tx = u_xy_CL_t-umeanx_cl
    umeany_cl = np.mean(u_xy_CL_tx, 1)
    u_xy_CL_end = u_xy_CL_tx-umeany_cl

    # 1D FFTs
    count = count + 1
    uk_CL_x = np.fft.fft(u_xy_CL_end, nx, 0)/nx        # axis 0:  x
    uk_CL_y = np.fft.fft(u_xy_CL_end, ny, 1)/ny        # axis 1:  y
    uk_CL_xy = np.fft.fft2(u_xy_CL_end, s=[nx, ny])/(nx*ny)        # FFT2
    ek_CL_x[:, :, count] = 0.5*np.real(np.multiply(uk_CL_x, np.conjugate(uk_CL_x)))
    ek_CL_y[:, :, count] = 0.5*np.real(np.multiply(uk_CL_y, np.conjugate(uk_CL_y)))
    ek_CL_xy[:, :, count] = 0.5*np.real(np.multiply(uk_CL_xy, np.conjugate(uk_CL_xy)))


ek_x_cl_timeavg = np.mean(ek_CL_x, 2)
ek_y_cl_timeavg = np.mean(ek_CL_y, 2)
ek_xy_cl_timeavg = np.mean(ek_CL_xy, 2)
ek_x_cl_timeavg2 = ek_x_cl_timeavg[1:nkx,:]+np.flipud(ek_x_cl_timeavg[nkx+1:,:])
ek_y_cl_timeavg2 = ek_y_cl_timeavg[:,1:int(ny/2)]+np.fliplr(ek_y_cl_timeavg[:,int(ny/2)+1:])
ek_xy_cl_timeavg2 = 2*ek_xy_cl_timeavg[1:nkx,1:int(ny/2)]+np.flipud(ek_xy_cl_timeavg[nkx+1:,1:int(ny/2)])+np.fliplr(ek_xy_cl_timeavg[1:nkx,int(ny/2)+1:])


## SAVE DATA FILES FOR PLOTS

np.save('u_xy_cl_gql', u_xy_CL_end)
np.save('ek_x_cl_gql', ek_x_cl_timeavg2)
np.save('ek_y_cl_gql', ek_y_cl_timeavg2)
np.save('ek_xy_cl_gql', ek_xy_cl_timeavg2)


# # Plot xy plane
cmap = plt.get_cmap('viridis')

# # Plot velocity to determine default orientation of slice
fig, ax = plt.subplots()
plt2=ax.pcolormesh(X, Y,u_xy_NW_end,cmap=cmap)
fig.colorbar(plt2)
#fig=plt.figure(figsize=(6,8))
#plt2=plt.imshow(u_xy_NW_end, cmap=cmap,origin='lower')
ax.set_title('Streamwise Velocity (x-y)')
ax.set_ylabel('y', fontsize=14)
ax.set_xlabel('x', fontsize=14)
fig.savefig('u_xy_.png')
np.save('u_xy_gql', u_xy_NW_end)
fig.show()
#
u_yz_NW=uvert_all[ii, 0, :, :]
# ekx, eky =Z.shape
# print(ekx)
# print(eky)
#
# fig, ax = plt.subplots()
# vert = ax.pcolormesh(Y2,Z, u_yz_NW, cmap=cmap)
# fig.colorbar(vert)
# ax.set_title('Streamwise Velocity (y-z)')
# ax.set_xlabel('y', fontsize=14)
# ax.set_ylabel('z', fontsize=14)
# fig.savefig('u_vert.png')
np.save('u_yz_gql', u_yz_NW)
# fig.show()
#
# #  1D Energy spectrum in x
# fig, ax = plt.subplots()
# contour1 = ax.contourf(kx[1:nkx, :], Y[1:nkx, 1:], ek_x_timeavg2[:, 1:], cmap=cmap)
# fig.colorbar(contour1)
# ax.set_title('1D Horizontal (x-y) Energy Spectrum in x, Near Wall')
# ax.set_xlabel('kx', fontsize=14)
# ax.set_ylabel('y', fontsize=14)
# ax.set_xscale('log')
# fig.savefig('Ek_x_xy_.png')
# fig.show()
np.save('ek_x_gql', ek_x_timeavg2)
#
# #  1D Energy spectrum in y
# fig, ax = plt.subplots()
# contour2=ax.contourf(X[0:, 1:int(ny/2)], ky2[0:, 1:int(ny/2)], ek_y_timeavg2, cmap=cmap)
# fig.colorbar(contour2)
# ax.set_title('1D Horizontal (x-y) Energy Spectrum in y, Near Wall')
# ax.set_xlabel('x', fontsize=14)
# ax.set_ylabel('ky', fontsize=14)
# ax.set_yscale('log')
# fig.savefig('Ek_y_xy_.png')
# fig.show()
np.save('ek_y_gql', ek_y_timeavg2)
#
# #ekx, eky =kx[1:nkx,1:int(ny/2)].shape
# #print(ekx)
# #print(eky)
#
fig, ax = plt.subplots()
contour3=ax.contourf(kx[1:nkx, 1:int(ny/2)], ky2[1:nkx, 1:int(ny/2)], ek_xy_timeavg2, cmap=cmap)
fig.colorbar(contour3)
# ##plt.clabel(contour4, inline=True, fontsize=8)
ax.set_title('2D Horizontal (x-y) Energy Spectrum of Streamwise Velocity, Near Wall')
ax.set_xlabel('kx', fontsize=14)
ax.set_xscale('log')
ax.set_ylabel('ky', fontsize=14)
ax.set_yscale('log')
fig.savefig('Ek_xy_xy_.png')
# np.save('ek_xy_gql', ek_xy_timeavg2)
plt.show()
