#!/mnt/lustre/chini/ck1055/GQL/dedalus/bin/python3


"""
Simulation script for 3D Plane-Poiseuille flowRotating Couette Flow
The shear is in the x-direction and is a function of z.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `process.py` script in this
folder can be used to merge distributed save files from parallel runs and plot
the snapshots from the command line.

The scripts make three different types of outputs: full 3D data dumps, 2D slices
through the data cube, and 1D horizontally averaged profiles.  The `process.py`
script can only plot the 2D slices.

To run, join, and plot vertical slices using 4 processes, for instance, you could
use:
$ mpiexec -n 4 python3 rayleigh_benard.py
$ mpiexec -n 4 python3 process.py join slices
$ mpiexec -n 4 python3 process.py plot_slices slices/*.h5 --type=horizontal
On four processors, the simulation runs in a couple of minutes.

This script is currently parallelized using a 
2D mesh decomposition can be used to run the problem with more processors.  To
do this, you must specify mesh=[N1,N2] in the domain creation, where N1*N2 is the
total number of processors.  In grid space, each processor has all the x data,
and a fraction of the y data (N1/Ny), and a fraction of the z data (N2/Nz).


"""

import os
import numpy as np
from mpi4py import MPI
import time 
from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly, Lz = (2.0*np.pi, np.pi, 1.)
Reynolds = 200.

# Create bases and domain
x_basis = de.Fourier("x", 84, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier("y", 84, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev("z", 108, interval=(-Lz, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[6,4])

# 3D Boussinesq hydrodynamics
problem = de.IVP(domain,
                variables=['pl','ul','vl','wl','ulz','vlz','wlz','ph','uh','vh','wh','uhz','vhz','whz'])

problem.parameters['InvReynolds'] = 1.0/Reynolds

#We shall decouple the temperature equation
#This has the correct sign of the Rossbynumber to agree with Bech & Andersson (1996)

problem.add_equation("dx(ul) + dy(vl) + wlz = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dx(ul) + dy(vl) + wlz = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(ul) - (dx(dx(ul)) + dy(dy(ul)) + dz(ulz)) + dx(pl)     = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(ul)  - InvReynolds*(dx(dx(ul)) + dy(dy(ul)) + dz(ulz)) + dx(pl)     = -1.0 - ul*dx(ul) - vl*dy(ul) - wl*ulz - uh*dx(uh) - vh*dy(uh) - wh*uhz",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(vl) - (dx(dx(vl)) + dy(dy(vl)) + dz(vlz)) + dy(pl)     = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(vl) - InvReynolds*(dx(dx(vl)) + dy(dy(vl)) + dz(vlz)) + dy(pl)     = - ul*dx(vl) - vl*dy(vl) - wl*vlz - uh*dx(vh) - vh*dy(vh) - wh*vhz",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(wl) - (dx(dx(wl)) + dy(dy(wl)) + dz(wlz)) + dz(pl) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(wl) - InvReynolds*(dx(dx(wl)) + dy(dy(wl)) + dz(wlz)) + dz(pl) = - ul*dx(wl) - vl*dy(wl) - wl*wlz - uh*dx(wh) - vh*dy(wh) - wh*whz",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("ulz - dz(ul) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("ulz - dz(ul) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("vlz - dz(vl) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("vlz - dz(vl) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("wlz - dz(wl) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("wlz - dz(wl) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dx(uh) + dy(vh) + whz = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dx(uh) + dy(vh) + whz = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(uh) - InvReynolds*(dx(dx(uh)) + dy(dy(uh)) + dz(uhz)) + dx(ph)     = - ul*dx(uh) - vl*dy(uh) - wl*uhz - uh*dx(ul) - vh*dy(ul) - wh*ulz",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(uh) - (dx(dx(uh)) + dy(dy(uh)) + dz(uhz)) + dx(ph)     = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(vh) - InvReynolds*(dx(dx(vh)) + dy(dy(vh)) + dz(vhz)) + dy(ph)     = - ul*dx(vh) - vl*dy(vh) - wl*vhz - uh*dx(vl) - vh*dy(vl) - wh*vlz",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(vh) - (dx(dx(vh)) + dy(dy(vh)) + dz(vhz)) + dy(ph)     = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("dt(wh) - InvReynolds*(dx(dx(wh)) + dy(dy(wh)) + dz(whz)) + dz(ph) = - ul*dx(wh) - vl*dy(wh) - wl*whz - uh*dx(wl) - vh*dy(wl) - wh*wlz",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("dt(wh) - (dx(dx(wh)) + dy(dy(wh)) + dz(whz)) + dz(ph) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("uhz - dz(uh) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("uhz - dz(uh) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("vhz - dz(vh) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("vhz - dz(vh) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_equation("whz - dz(wh) = 0",condition="abs(ny) >=1000.0000001 or abs(nx) >=1000.0000001")
problem.add_equation("whz - dz(wh) = 0",condition="abs(ny) < 1000.0000001 and abs(nx) < 1000.0000001")
problem.add_bc("left(ul) = 0")
problem.add_bc("left(vl) = 0")
problem.add_bc("left(wl) = 0")
problem.add_bc("right(ul) = 0")
problem.add_bc("right(vl) = 0")
problem.add_bc("right(wl) = 0", condition="nx != 0 or ny != 0")
problem.add_bc("integ_z(pl) = 0", condition="nx == 0 and ny == 0")
problem.add_bc("left(uh) = 0")
problem.add_bc("left(vh) = 0")
problem.add_bc("left(wh) = 0")
problem.add_bc("right(uh) = 0")
problem.add_bc("right(vh) = 0")
problem.add_bc("right(wh) = 0", condition="nx != 0 or ny != 0")
problem.add_bc("integ_z(ph) = 0", condition="nx == 0 and ny == 0")

# Build solver
ts = de.timesteppers.SBDF3
solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
ul = solver.state['ul']
ulz = solver.state['ulz']
vl = solver.state['vl']
vlz = solver.state['vlz']
uh = solver.state['uh']
uhz = solver.state['uhz']
vh = solver.state['vh']
vhz = solver.state['vhz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
seed = 4835+domain.distributor.rank
rand = np.random.RandomState(seed=seed)
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)

pert =  2.0 * rand.standard_normal(gshape)[slices] * (zt - z) * (z - zb)
ul['g'] = 0.5*Reynolds*(z - zt) * (z - zb) + pert
ul.differentiate('z', out=ulz)
#vl['g'] = 
#vl.differentiate('z', out=vlz)

uh['g'] = pert
uh.differentiate('z', out=uhz)
vh['g'] = pert
vh.differentiate('z', out=vhz)

# Integration parameters
dt = 1e-2
solver.stop_sim_time = 750
solver.stop_wall_time = 200000 * 60.
solver.stop_iteration = np.inf
hermitian_cadence = 100

# CFL
safety = 0.3
cfl_cadence = 10
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=cfl_cadence, safety=safety,
                     max_change=1.5, min_change=0.5, max_dt=0.05)
CFL.add_velocities(('ul', 'vl', 'wl'))

evaluator = solver.evaluator
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly

# Analysis
snapshots = evaluator.add_file_handler('slices', sim_dt=0.5, max_writes=50)
snapshots.add_task("interp(uh,z=0.1)", name="uh z 0.1")
snapshots.add_task("interp(vh,z=0.1)", name="vh z 0.1")
snapshots.add_task("interp(wh,z=0.1)", name="wh z 0.1")
snapshots.add_task("interp(ul,z=0.1)", name="ul z 0.1")
snapshots.add_task("interp(vl,z=0.1)", name="vl z 0.1")
snapshots.add_task("interp(wl,z=0.1)", name="wl z 0.1")
snapshots.add_task("interp(vh+vl,z=0.1)", name="v z 0.1")
snapshots.add_task("interp(uh+ul,z=0.1)", name="u z 0.1")
snapshots.add_task("interp(wh+wl,z=0.1)", name="w z 0.1")
snapshots.add_task("interp(vh+vl,z=0.95)", name="v z 0.95")
snapshots.add_task("interp(uh+ul,z=0.95)", name="u z 0.95")
snapshots.add_task("interp(wh+wl,z=0.95)", name="w z 0.95")

snapshots.add_task("interp(ul+uh,x=0)", name="u vertical")
snapshots.add_task("interp(vl+vh,x=0)", name="v vertical")
snapshots.add_task("interp(wl+wh,x=0)", name="w vertical")
snapshots.add_task("interp(ulz+uhz,x=0)", name="uz vertical")

snapshots = evaluator.add_file_handler('dump', sim_dt=20.0, max_writes=10)
snapshots.add_task("ul",name="ul")
snapshots.add_task("wl",name="wl")
snapshots.add_task("vl",name="vl")
snapshots.add_task("uh",name="uh")
snapshots.add_task("wh",name="wh")
snapshots.add_task("vh",name="vh")
snapshots.add_task("ulz",name="ulz")
snapshots.add_task("uhz",name="uhz")

snapshots = evaluator.add_file_handler('checkpoint', sim_dt=0.5, max_writes=50)
snapshots.add_system(solver.state)

##snapshots = evaluator.add_file_handler('profiles', sim_dt=2.0)
##snapshots.add_task("integ(integ(ul,'x'),'y')/Lx/Ly", name="u")
##snapshots.add_task("integ(integ(vl,'x'),'y')/Lx/Ly", name="v")
##snapshots.add_task("integ(integ(wl,'x'),'y')/Lx/Ly", name="w")

# Main loop
try:
    logger.info('dy(10): %e,  dy(100): %e' %(x_basis.wavenumbers[10],x_basis.wavenumbers[20]))    
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration - 1) % cfl_cadence == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

        if (solver.iteration - 1) % hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

