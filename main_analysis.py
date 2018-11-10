import numpy as np
import importlib
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Define parameters
Lx, Ly, Lz = (2.*np.pi, np.pi, 1.)
Reynolds = 200.
nx, ny, nz = (84, 84, 108)
gm = 2                          # finer grid multiplier (2 = twice as many points)

#  Define grid
import grid
importlib.reload(grid)
grid.define(Lx, Ly, Lz, nx, ny, nz, gm)

# Import applicable files
#
import nl_analysis
import gql_analysis    # gql files (Ekx, Eky, means)
import ql_analysis

# import load             # grid(Lx, Ly, Lz, nx, ny, nz)
# Allow updates

importlib.reload(nl_analysis)
importlib.reload(gql_analysis)
importlib.reload(ql_analysis)

# Run simulation data submodules
# nl_analysis.nl(nx, ny, nz, Reynolds)          # t = 208.9369
# gql_analysis.gql1(nx, ny, nz, Reynolds)       # data is from incorrect CP
# gql_analysis.gql2(nx, ny, nz, Reynolds)       # t = 263.9963*
# gql_analysis.gql3(nx, ny, nz, Reynolds)       # t = 420.9264*
# gql_analysis.gql8(nx, ny, nz, Reynolds)       # t = 190.5511*
# ql_analysis.ql(nx, ny, nz, Reynolds)          # t = 42.66525 rerunning sim, 5 Nov

# load.files()

# xy plane plots
import horiz  # instant_ql, instant_gql3, instant_nl
importlib.reload(horiz)
# horiz.instant_ql()
# horiz.instant_gql2()
# horiz.instant_gql3()
# horiz.instant_nl()
# horiz.finer_instant_nl()
# horiz.finer_instant_ql()
# horiz.finer_instant_gql2()
# horiz.finer_instant_gql3()
# horiz.finer_all()
# horiz.compare(nx)          # change y axis to increments of pi
# horiz.fcompare(nx)          # change y axis to increments of pi
# horiz.fcompare_instant(nx)

# Profile plots
import profiles     # mean(nx)
importlib.reload(profiles)
# profiles.clmax()
# profiles.Q()
# profiles.Qplot()
# profiles.curve()
# profiles.meanp(nx)        # w/ poiseuille profile
# profiles.mean(nx)
# profiles.mean_ind()

# vertical plots
import vertical
importlib.reload(vertical)
# vertical.compare(nx, nz)   # quiver overlay isn't working
# vertical.vertslice(nz)         # spline doesn't work this way... :(
# vertical.quiver(nz)

# spectra
import spectra
importlib.reload(spectra)
# spectra.nw_Ekxy(nx, ny)     # 2d spectra, nw, xy plane
spectra.maxmode_kx(nx)          # max spectra modes, kx
spectra.maxmode_kx_zoom(nx)          # max spectra modes, kx, zoomed in
spectra.maxmode_kx_noql(nx)     # no QL to mess up plot
spectra.maxmode_ky(nx)          # max spctra modes, ky
