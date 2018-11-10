import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools



def define(Lx, Ly, Lz, nx, ny, nz, gm):
    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
    y_basis = de.Fourier("y", ny, interval=(0, Ly), dealias=3 / 2)
    z_basis = de.Chebyshev("z", nz, interval=(-Lz, Lz), dealias=3 / 2)
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

    # # Unscaled grid (physical space)
    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)

    X, Y = np.meshgrid(x, y, indexing='ij')
    Y2, Z = np.meshgrid(y, z, indexing='ij')
    kx, ky = np.meshgrid(x_basis.wavenumbers, y_basis.wavenumbers, indexing='ij')
    x2, ky2 = np.meshgrid(x, y_basis.wavenumbers, indexing='ij')
    Yhalf = Y2[:, 0:nz/2]
    Zhalf = Z[:, 0:nz/2]
    nkx, nky = kx.shape
    zb, zt = z_basis.interval

# Fine Grid, xy plane
    xmax, ymax = Lx, Ly
    dx2, dy2 = Lx / (gm * nx), Ly / (gm * ny)
    x2 = np.arange(0, xmax, dx2)
    y2 = np.arange(0, ymax, dy2)
    fX, fYh = np.meshgrid(x2, y2, indexing='ij')

# Fine Grid, yz plane
    ymaxv, zmax = Ly, Lz
    dy2, dz2 = Ly / (gm * ny), Lz / (gm * nz)
    y2v = np.arange(0, ymaxv, dy2)
    z2 = np.arange(0, zmax, dz2)
    fYv, fZ = np.meshgrid(y2v, z2, indexing='ij')

    # Save grid data
    # capitals:  mesh grid
    np.save('X', X)  # X mesh grid, xy plane, horizontal
    np.save('Yh', Y)  # Y mesh grid, xy plane, horizontal
    np.save('Yhalf', Yhalf)     # for vertical half plane
    np.save('Yv', Y2)  # Y mesh grid, yz plane, vertical
    np.save('Z', Z)  # z mesh grid
    np.save('Zhalf', Zhalf)     # for vertical half plane
    np.save('fX', fX)       # finer horiz grid
    np.save('fYh', fYh)       # finer horiz grid
    np.save('fZ', fZ)       # finer vert grid
    np.save('fYv', fYv)       # finer vert grid
    # lowercase:  1d
    np.save('x', x)  # x 1d
    np.save('x2', x2)  # 1d finer grid
    np.save('y', y)  # y 1d
    np.save('y2', y2) # 1d finer grid
    np.save('y2v', y2v)
    np.save('z', z)  # z 1d
    np.save('z2', z2) # finer grid
    np.save('zb', zb)
    np.save('zt', zt)
    # wavenumber grid for 1D [kx = 2*pi/Lx * nkx], [ky = 2*pi/Ly * nky]
    np.save('kx', kx)  #
    np.save('ky', ky)  #
    # wavenumber grid for 2D
    np.save('x2', x2)
    np.save('ky2', ky2)
    # mode numbers
    np.save('nkx', nkx)  #
    np.save('nky', nky)  #