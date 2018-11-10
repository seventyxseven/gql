import numpy as np
from scipy.interpolate import RectBivariateSpline
import h5py

x = np.load('x.npy')
y = np.load('y.npy')
y2 = np.load('y2.npy')
z = np.load('z.npy')
z2 = np.load('z2.npy')
x2 = np.load('x2.npy')  # finer grid, controlled by gm variable in main
y2 = np.load('y2.npy')  # finer grid

# Load grid values
x = np.load('x.npy')
y = np.load('y.npy')
z = np.load('z.npy')
x2 = np.load('x2.npy')  # finer grid, controlled by gm variable in main
y2 = np.load('y2.npy')  # finer grid

def gql3(nx, ny, nz, Reynolds):
# Load raw data
    ff = h5py.File('slices_gql3/slices.h5', 'r')
    gg = h5py.File('dump_gql3/dump.h5', 'r')
    u_xy_cl = ff['tasks']['u z 0.1'][:]  # u in xy plane, z=0.1 (near centerline)
    u_xy_nw = ff['tasks']['u z 0.95'][:]
    uvert_all = ff['tasks']['u vertical'][:]
    vvert_all = ff['tasks']['v vertical'][:]
    wvert_all = ff['tasks']['w vertical'][:]
    ul_xy_cl = ff['tasks']['ul z 0.1'][:]
    uh_xy_cl = ff['tasks']['uh z 0.1'][:]
    uzvert_all = ff['tasks']['uz vertical'][:]
    ulz_all = gg['tasks']['ulz'][:]
    ul_dump = gg['tasks']['ul'][:]
    vl_dump = gg['tasks']['vl'][:]
    wl_dump = gg['tasks']['wl'][:]

# Load grid files
    nkx = np.load('nkx.npy')

# Establish steady state slices
    tslices = u_xy_nw.shape[0]
    t_turb = int(round(7 / 8 * tslices))
    tdump = ul_dump.shape[0]
    turb_dump = int(round(5 / 11 * tdump))
    Nt1 = tdump - turb_dump
    Nt0 = tslices - t_turb
    Ntstep = t_turb

    u_xy_nw_inst = u_xy_nw[tslices - 1, :, :, 0]
    u_xy_cl_inst = u_xy_cl[tslices - 5, :, :, 0]

# finer grid
    is_nw_instant = RectBivariateSpline(y.T, x.T, u_xy_nw_inst)
    f_u_xy_nw = is_nw_instant(y2, x2)
    is_cl_instant = RectBivariateSpline(y.T, x.T, u_xy_cl_inst)
    f_u_xy_cl = is_cl_instant(y2, x2)

# Compute mean velocity

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
    # return print('gql3 min <u>ty:', np.amin(umeanvty))
   # print('min <du/dz>ty:', min(uzmeanty))
    u_tau = np.sqrt(-uzmeanty[0] / Reynolds)
    # print('u_tau:', u_tau)

# Compute velocity perturbations, and energy spectra

    count = -1;
    umeant = np.mean(u_xy_nw[Nt0:tslices, :, :, 0], 0)
    ek_NW_x = np.empty((nx, ny, Ntstep))
    ek_NW_y = np.empty((nx, ny, Ntstep))
    ek_NW_xy = np.empty((nx, ny, Ntstep))
    for ii in range(Nt0, tslices):

        # print(ii)
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

# Compute average velocity gradient
    uzmeany = np.mean(uzvert_all[:, 0, :, :], 1)
# Compute vertical mean avg in y
    uvmeany = np.mean(uvert_all[:, 0, :, :], 1)

# Compute u,v,w fluctuations in vertical plane (quiver plot)
    U_plus = uvert_all[:, 0, :, :] / u_tau
    U_py = np.mean(U_plus, 1)
    U_pyt = np.abs(np.flipud(np.mean(U_py[Nt0:tslices, nz / 2:], 0)))

    uvmeant = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
    vvmeant = np.mean(vvert_all[Nt0:tslices, 0, :, :], 0)
    wvmeant = np.mean(wvert_all[Nt0:tslices, 0, :, :], 0)
    upvert = np.empty((ny, nz, Ntstep))
    vpvert = np.empty((ny, nz, Ntstep))
    wpvert = np.empty((ny, nz, Ntstep))
    upvertsq = np.empty((ny, nz, Ntstep))
    counter = -1
    for jj in range(Nt0, tslices):
        uvert_slices = uvert_all[jj, 0, :, :]
        vvert_slices = vvert_all[jj, 0, :, :]
        wvert_slices = wvert_all[jj, 0, :, :]
        counter = counter + 1
        upvert[:, :, counter] = (uvert_slices - uvmeant)
        vpvert[:, :, counter] = (vvert_slices - vvmeant)
        wpvert[:, :, counter] = (wvert_slices - wvmeant)
        upvertsq[:, :, counter] = (uvert_slices - uvmeant) ** 2
    upvertsq_yavg = np.mean(upvertsq, 0)
    upvertsq_ytavg = np.mean(upvertsq_yavg, 1)
    upsq_yt = np.flipud(upvertsq_ytavg[nz / 2:])

# fluctuations using dump file (xyt avged)
    up_sq = up ** 2
    upsq_xmean = np.mean(up_sq, 0)
    upsq_xymean = np.mean(upsq_xmean, 0)
    upsq_xytmean = np.mean(upsq_xymean, 1)
    upsq_xyt = np.flipud(upsq_xytmean[nz / 2:])

# vertical plot (quiver plot)
    upvert_plot = upvert[:, 0:nz / 2, -1]
    vpvert_plot = vpvert[:, 0:nz / 2, -1]
    wpvert_plot = wpvert[:, 0:nz / 2, -1]

# finer grid
    zz = z[0, 0, 0:nz/2]
    is_upvert = RectBivariateSpline(y.T, zz, upvert_plot)
    f_upvert_plot = is_upvert(y2, z2)

    mag = np.sqrt(vpvert_plot ** 2 + wpvert_plot ** 2)
    vp = vpvert_plot / mag
    wp = wpvert_plot / mag

# Save all arrays for plots

    np.save('u_xy_nw_gql3', u_xy_nw_inst)            # instantaneous slice, second to last slice
    np.save('u_xy_cl_gql3', u_xy_cl_inst)
    np.save('u_xy_NW_gql3', u_xy_NW_end)                        # NW fluctuations (field - mean)*
    np.save('u_xy_CL_gql3', u_xy_CL_end)                            # CL fluctuations (field - mean)*
    np.save('uh_xy_cl_gql3', uh_xy_cl)           # high modes instantaneous
    np.save('ul_xy_cl_gql3', ul_xy_cl)           # high modes instantaneous
    np.save('uvert_gql3', uvert_all)             # xz plane used for mean calculation
    np.save('ek_xy_gql3', ek_xy_timeavg2)        # 2d spectra*
    np.save('ek_x_gql3', ek_x_timeavg2)          # 1d spectra, x*
    np.save('ek_y_gql3', ek_y_timeavg2)          # 1d spectra, y*
    np.save('u_tau_gql3', u_tau)                 # u_tau calculation
    np.save('uzmeanty_gql3', uzmeanty)           # velocity gradient avg in y and t*
    np.save('uzmeany_gql3', uzmeany)             # velocity gradient avg in y (for plots every time step)*
    np.save('uvmeany_gql3', uvmeany[t_turb:tslices, :])
    np.save('umeanvty_gql3', umeanvty)
    np.save('Upyt_gql3', U_pyt)
    np.save('upsq_yt_gql3', upsq_yt)
    np.save('upsq_xyt_gql3', upsq_xyt)
    np.save('upvert_plot_gql3', upvert_plot)
    np.save('f_upvert_plot_gql3', f_upvert_plot)
    np.save('vp_gql3', vp)
    np.save('wp_gql3', wp)

def gql2(nx, ny, nz, Reynolds):
    # Load raw data
    ff = h5py.File('slices_gql2/slices.h5', 'r')
    gg = h5py.File('dump_gql2/dump.h5', 'r')
    u_xy_cl = ff['tasks']['u z 0.1'][:]  # u in xy plane, z=0.1 (near centerline)
    u_xy_nw = ff['tasks']['u z 0.95'][:]
    uvert_all = ff['tasks']['u vertical'][:]
    vvert_all = ff['tasks']['v vertical'][:]
    wvert_all = ff['tasks']['w vertical'][:]
    ul_xy_cl = ff['tasks']['ul z 0.1'][:]
    uh_xy_cl = ff['tasks']['uh z 0.1'][:]
    uzvert_all = ff['tasks']['uz vertical'][:]
    ulz_all = gg['tasks']['ulz'][:]
    ul_dump = gg['tasks']['ul'][:]
    vl_dump = gg['tasks']['vl'][:]
    wl_dump = gg['tasks']['wl'][:]

    # Load grid files
    nkx = np.load('nkx.npy')

    # Establish steady state slices
    tslices = u_xy_nw.shape[0]
    t_turb = int(round(7 / 8 * tslices))
    tdump = ul_dump.shape[0]
    turb_dump = int(round(5 / 11 * tdump))
    Nt1 = tdump - turb_dump
    Nt0 = tslices - t_turb
    Ntstep = t_turb

    u_xy_nw_inst = u_xy_nw[tslices - 1, :, :, 0]
    u_xy_cl_inst = u_xy_cl[tslices - 1, :, :, 0]

    # finer grid
    is_nw_instant = RectBivariateSpline(y.T, x.T, u_xy_nw_inst)
    f_u_xy_nw = is_nw_instant(y2, x2)
    is_cl_instant = RectBivariateSpline(y.T, x.T, u_xy_cl_inst)
    f_u_xy_cl = is_cl_instant(y2, x2)

    # Compute mean velocity

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
    # return print('gql2 min <u>ty:', np.amin(umeanvty))
    # print('min <du/dz>ty:', min(uzmeanty))
    u_tau = np.sqrt(-uzmeanty[0] / Reynolds)
    # print('u_tau:', u_tau)

    # Compute velocity perturbations, and energy spectra

    count = -1;
    umeant = np.mean(u_xy_nw[Nt0:tslices, :, :, 0], 0)
    ek_NW_x = np.empty((nx, ny, Ntstep))
    ek_NW_y = np.empty((nx, ny, Ntstep))
    ek_NW_xy = np.empty((nx, ny, Ntstep))
    for ii in range(Nt0, tslices):
        # print(ii)
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
    ek_xy_timeavg2 = 2 * ek_xy_timeavg[1:nkx, 1:int(ny / 2)] + np.flipud(
        ek_xy_timeavg[nkx + 1:, 1:int(ny / 2)]) + np.fliplr(ek_xy_timeavg[1:nkx, int(ny / 2) + 1:])

    # Compute average velocity gradient
    uzmeany = np.mean(uzvert_all[:, 0, :, :], 1)
    # Compute vertical mean avg in y
    uvmeany = np.mean(uvert_all[:, 0, :, :], 1)

    # Compute u,v,w fluctuations in vertical plane (quiver plot)
    U_plus = uvert_all[:, 0, :, :] / u_tau
    U_py = np.mean(U_plus, 1)
    U_pyt = np.abs(np.flipud(np.mean(U_py[Nt0:tslices, nz / 2:], 0)))

    uvmeant = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
    vvmeant = np.mean(vvert_all[Nt0:tslices, 0, :, :], 0)
    wvmeant = np.mean(wvert_all[Nt0:tslices, 0, :, :], 0)
    upvert = np.empty((ny, nz, Ntstep))
    vpvert = np.empty((ny, nz, Ntstep))
    wpvert = np.empty((ny, nz, Ntstep))
    upvertsq = np.empty((ny, nz, Ntstep))
    counter = -1
    for jj in range(Nt0, tslices):
        uvert_slices = uvert_all[jj, 0, :, :]
        vvert_slices = vvert_all[jj, 0, :, :]
        wvert_slices = wvert_all[jj, 0, :, :]
        counter = counter + 1
        upvert[:, :, counter] = (uvert_slices - uvmeant)
        vpvert[:, :, counter] = (vvert_slices - vvmeant)
        wpvert[:, :, counter] = (wvert_slices - wvmeant)
        upvertsq[:, :, counter] = (uvert_slices - uvmeant) ** 2
    upvertsq_yavg = np.mean(upvertsq, 0)
    upvertsq_ytavg = np.mean(upvertsq_yavg, 1)
    upsq_yt = np.flipud(upvertsq_ytavg[nz / 2:])

    # fluctuations using dump file (xyt avged)
    up_sq = up ** 2
    upsq_xmean = np.mean(up_sq, 0)
    upsq_xymean = np.mean(upsq_xmean, 0)
    upsq_xytmean = np.mean(upsq_xymean, 1)
    upsq_xyt = np.flipud(upsq_xytmean[nz / 2:])

    # vertical plot (quiver plot)
    upvert_plot = upvert[:, 0:nz / 2, -1]
    vpvert_plot = vpvert[:, 0:nz / 2, -1]
    wpvert_plot = wpvert[:, 0:nz / 2, -1]
    mag = np.sqrt(vpvert_plot ** 2 + wpvert_plot ** 2)
    vp = vpvert_plot / mag
    wp = wpvert_plot / mag

    # Save all arrays for plots

    np.save('u_xy_nw_gql2', u_xy_nw_inst)  # instantaneous slice, second to last slice
    np.save('u_xy_cl_gql2', u_xy_cl_inst)
    np.save('u_xy_NW_gql2', u_xy_NW_end)  # NW fluctuations (field - mean)*
    np.save('u_xy_CL_gql2', u_xy_CL_end)  # CL fluctuations (field - mean)*
    np.save('uh_xy_cl_gql2', uh_xy_cl)  # high modes instantaneous
    np.save('ul_xy_cl_gql2', ul_xy_cl)  # high modes instantaneous
    np.save('uvert_gql2', uvert_all)  # xz plane used for mean calculation
    np.save('ek_xy_gql2', ek_xy_timeavg2)  # 2d spectra*
    np.save('ek_x_gql2', ek_x_timeavg2)  # 1d spectra, x*
    np.save('ek_y_gql2', ek_y_timeavg2)  # 1d spectra, y*
    np.save('u_tau_gql2', u_tau)  # u_tau calculation
    np.save('uzmeanty_gql2', uzmeanty)  # velocity gradient avg in y and t*
    np.save('uzmeany_gql2', uzmeany)  # velocity gradient avg in y (for plots every time step)*
    np.save('uvmeany_gql2', uvmeany[t_turb:tslices, :])
    np.save('umeanvty_gql2', umeanvty)
    np.save('Upyt_gql2', U_pyt)
    np.save('upsq_yt_gql2', upsq_yt)
    np.save('upsq_xyt_gql2', upsq_xyt)
    np.save('upvert_plot_gql2', upvert_plot)
    np.save('vp_gql2', vp)
    np.save('wp_gql2', wp)

def gql8(nx, ny, nz, Reynolds):
    # Load raw data
    ff = h5py.File('slices_gql8/slices.h5', 'r')
    gg = h5py.File('dump_gql8/dump.h5', 'r')
    u_xy_cl = ff['tasks']['u z 0.1'][:]  # u in xy plane, z=0.1 (near centerline)
    u_xy_nw = ff['tasks']['u z 0.95'][:]
    uvert_all = ff['tasks']['u vertical'][:]
    vvert_all = ff['tasks']['v vertical'][:]
    wvert_all = ff['tasks']['w vertical'][:]
    ul_xy_cl = ff['tasks']['ul z 0.1'][:]
    uh_xy_cl = ff['tasks']['uh z 0.1'][:]
    uzvert_all = ff['tasks']['uz vertical'][:]
    ulz_all = gg['tasks']['ulz'][:]
    ul_dump = gg['tasks']['ul'][:]
    vl_dump = gg['tasks']['vl'][:]
    wl_dump = gg['tasks']['wl'][:]

    # Load grid files
    nkx = np.load('nkx.npy')

    # Establish steady state slices
    tslices = u_xy_nw.shape[0]
    t_turb = int(round(7 / 8 * tslices))
    tdump = ul_dump.shape[0]
    turb_dump = int(round(5 / 11 * tdump))
    Nt1 = tdump - turb_dump
    Nt0 = tslices - t_turb
    Ntstep = t_turb

    u_xy_nw_inst = u_xy_nw[tslices - 1, :, :, 0]
    u_xy_cl_inst = u_xy_cl[tslices - 1, :, :, 0]

    # finer grid
    is_nw_instant = RectBivariateSpline(y.T, x.T, u_xy_nw_inst)
    f_u_xy_nw = is_nw_instant(y2, x2)
    is_cl_instant = RectBivariateSpline(y.T, x.T, u_xy_cl_inst)
    f_u_xy_cl = is_cl_instant(y2, x2)
    # Compute mean velocity

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
    # return print('gql8 min <u>ty:', min(umeanvty))
    # print('min <du/dz>ty:', min(uzmeanty))
    u_tau = np.sqrt(-uzmeanty[0] / Reynolds)
    # print('u_tau:', u_tau)

    # Compute velocity perturbations, and energy spectra

    count = -1;
    umeant = np.mean(u_xy_nw[Nt0:tslices, :, :, 0], 0)
    ek_NW_x = np.empty((nx, ny, Ntstep))
    ek_NW_y = np.empty((nx, ny, Ntstep))
    ek_NW_xy = np.empty((nx, ny, Ntstep))
    for ii in range(Nt0, tslices):
        # print(ii)
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
    ek_xy_timeavg2 = 2 * ek_xy_timeavg[1:nkx, 1:int(ny / 2)] + np.flipud(
        ek_xy_timeavg[nkx + 1:, 1:int(ny / 2)]) + np.fliplr(ek_xy_timeavg[1:nkx, int(ny / 2) + 1:])

    # Compute average velocity gradient
    uzmeany = np.mean(uzvert_all[:, 0, :, :], 1)
    # Compute vertical mean avg in y
    uvmeany = np.mean(uvert_all[:, 0, :, :], 1)

    # Compute u,v,w fluctuations in vertical plane (quiver plot)
    U_plus = uvert_all[:, 0, :, :] / u_tau
    U_py = np.mean(U_plus, 1)
    U_pyt = np.abs(np.flipud(np.mean(U_py[Nt0:tslices, nz / 2:], 0)))

    uvmeant = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
    vvmeant = np.mean(vvert_all[Nt0:tslices, 0, :, :], 0)
    wvmeant = np.mean(wvert_all[Nt0:tslices, 0, :, :], 0)
    upvert = np.empty((ny, nz, Ntstep))
    vpvert = np.empty((ny, nz, Ntstep))
    wpvert = np.empty((ny, nz, Ntstep))
    upvertsq = np.empty((ny, nz, Ntstep))
    counter = -1
    for jj in range(Nt0, tslices):
        uvert_slices = uvert_all[jj, 0, :, :]
        vvert_slices = vvert_all[jj, 0, :, :]
        wvert_slices = wvert_all[jj, 0, :, :]
        counter = counter + 1
        upvert[:, :, counter] = (uvert_slices - uvmeant)
        vpvert[:, :, counter] = (vvert_slices - vvmeant)
        wpvert[:, :, counter] = (wvert_slices - wvmeant)
        upvertsq[:, :, counter] = (uvert_slices - uvmeant) ** 2
    upvertsq_yavg = np.mean(upvertsq, 0)
    upvertsq_ytavg = np.mean(upvertsq_yavg, 1)
    upsq_yt = np.flipud(upvertsq_ytavg[nz / 2:])

    # fluctuations using dump file (xyt avged)
    up_sq = up ** 2
    upsq_xmean = np.mean(up_sq, 0)
    upsq_xymean = np.mean(upsq_xmean, 0)
    upsq_xytmean = np.mean(upsq_xymean, 1)
    upsq_xyt = np.flipud(upsq_xytmean[nz / 2:])

    # vertical plot (quiver plot)
    upvert_plot = upvert[:, 0:nz / 2, -1]
    vpvert_plot = vpvert[:, 0:nz / 2, -1]
    wpvert_plot = wpvert[:, 0:nz / 2, -1]
    mag = np.sqrt(vpvert_plot ** 2 + wpvert_plot ** 2)
    vp = vpvert_plot / mag
    wp = wpvert_plot / mag

    # Save all arrays for plots

    np.save('u_xy_nw_gql8', u_xy_nw_inst)  # instantaneous slice, second to last slice
    np.save('u_xy_cl_gql8', u_xy_cl_inst)
    np.save('u_xy_NW_gql8', u_xy_NW_end)  # NW fluctuations (field - mean)*
    np.save('u_xy_CL_gql8', u_xy_CL_end)  # CL fluctuations (field - mean)*
    np.save('uh_xy_cl_gql8', uh_xy_cl)  # high modes instantaneous
    np.save('ul_xy_cl_gql8', ul_xy_cl)  # high modes instantaneous
    np.save('uvert_gql8', uvert_all)  # xz plane used for mean calculation
    np.save('ek_xy_gql8', ek_xy_timeavg2)  # 2d spectra*
    np.save('ek_x_gql8', ek_x_timeavg2)  # 1d spectra, x*
    np.save('ek_y_gql8', ek_y_timeavg2)  # 1d spectra, y*
    np.save('u_tau_gql8', u_tau)  # u_tau calculation
    np.save('uzmeanty_gql8', uzmeanty)  # velocity gradient avg in y and t*
    np.save('uzmeany_gql8', uzmeany)  # velocity gradient avg in y (for plots every time step)*
    np.save('uvmeany_gql8', uvmeany[t_turb:tslices, :])
    np.save('umeanvty_gql8', umeanvty)
    np.save('Upyt_gql8', U_pyt)
    np.save('upsq_yt_gql8', upsq_yt)
    np.save('upsq_xyt_gql8', upsq_xyt)
    np.save('upvert_plot_gql8', upvert_plot)
    np.save('vp_gql8', vp)
    np.save('wp_gql8', wp)

def gql1(nx, ny, nz, Reynolds):
    # Load raw data
    ff = h5py.File('slices_gql1/slices.h5', 'r')
    gg = h5py.File('dump_gql1/dump.h5', 'r')
    u_xy_cl = ff['tasks']['u z 0.1'][:]  # u in xy plane, z=0.1 (near centerline)
    u_xy_nw = ff['tasks']['u z 0.95'][:]
    uvert_all = ff['tasks']['u vertical'][:]
    vvert_all = ff['tasks']['v vertical'][:]
    wvert_all = ff['tasks']['w vertical'][:]
    ul_xy_cl = ff['tasks']['ul z 0.1'][:]
    uh_xy_cl = ff['tasks']['uh z 0.1'][:]
    uzvert_all = ff['tasks']['uz vertical'][:]
    ulz_all = gg['tasks']['ulz'][:]
    ul_dump = gg['tasks']['ul'][:]
    vl_dump = gg['tasks']['vl'][:]
    wl_dump = gg['tasks']['wl'][:]

    # Load grid files
    nkx = np.load('nkx.npy')

    # Establish steady state slices
    tslices = u_xy_nw.shape[0]
    t_turb = int(round(5 / 8 * tslices))
    tdump = ul_dump.shape[0]
    turb_dump = int(round(5 / 11 * tdump))
    Nt1 = tdump - turb_dump
    Nt0 = tslices - t_turb
    Ntstep = t_turb

    u_xy_nw_inst = u_xy_nw[tslices - 1, :, :, 0]
    u_xy_cl_inst = u_xy_cl[tslices - 1, :, :, 0]

    # finer grid
    is_nw_instant = RectBivariateSpline(y.T, x.T, u_xy_nw_inst)
    f_u_xy_nw = is_nw_instant(y2, x2)
    is_cl_instant = RectBivariateSpline(y.T, x.T, u_xy_cl_inst)
    f_u_xy_cl = is_cl_instant(y2, x2)

    # Compute mean velocity

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
    return print('gql1 min <u>ty:', min(umeanvty))
    # print('min <du/dz>ty:', min(uzmeanty))
    u_tau = np.sqrt(-uzmeanty[0] / Reynolds)
    # print('u_tau:', u_tau)

    # Compute velocity perturbations, and energy spectra

    count = -1;
    umeant = np.mean(u_xy_nw[Nt0:tslices, :, :, 0], 0)
    ek_NW_x = np.empty((nx, ny, Ntstep))
    ek_NW_y = np.empty((nx, ny, Ntstep))
    ek_NW_xy = np.empty((nx, ny, Ntstep))
    for ii in range(Nt0, tslices):
        # print(ii)
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
    ek_xy_timeavg2 = 2 * ek_xy_timeavg[1:nkx, 1:int(ny / 2)] + np.flipud(
        ek_xy_timeavg[nkx + 1:, 1:int(ny / 2)]) + np.fliplr(ek_xy_timeavg[1:nkx, int(ny / 2) + 1:])

    # Compute average velocity gradient
    uzmeany = np.mean(uzvert_all[:, 0, :, :], 1)
    # Compute vertical mean avg in y
    uvmeany = np.mean(uvert_all[:, 0, :, :], 1)

    # Compute u,v,w fluctuations in vertical plane (quiver plot)
    U_plus = uvert_all[:, 0, :, :] / u_tau
    U_py = np.mean(U_plus, 1)
    U_pyt = np.abs(np.flipud(np.mean(U_py[Nt0:tslices, nz / 2:], 0)))

    uvmeant = np.mean(uvert_all[Nt0:tslices, 0, :, :], 0)
    vvmeant = np.mean(vvert_all[Nt0:tslices, 0, :, :], 0)
    wvmeant = np.mean(wvert_all[Nt0:tslices, 0, :, :], 0)
    upvert = np.empty((ny, nz, Ntstep))
    vpvert = np.empty((ny, nz, Ntstep))
    wpvert = np.empty((ny, nz, Ntstep))
    upvertsq = np.empty((ny, nz, Ntstep))
    counter = -1
    for jj in range(Nt0, tslices):
        uvert_slices = uvert_all[jj, 0, :, :]
        vvert_slices = vvert_all[jj, 0, :, :]
        wvert_slices = wvert_all[jj, 0, :, :]
        counter = counter + 1
        upvert[:, :, counter] = (uvert_slices - uvmeant)
        vpvert[:, :, counter] = (vvert_slices - vvmeant)
        wpvert[:, :, counter] = (wvert_slices - wvmeant)
        upvertsq[:, :, counter] = (uvert_slices - uvmeant) ** 2
    upvertsq_yavg = np.mean(upvertsq, 0)
    upvertsq_ytavg = np.mean(upvertsq_yavg, 1)
    upsq_yt = np.flipud(upvertsq_ytavg[nz / 2:])

    # fluctuations using dump file (xyt avged)
    up_sq = up ** 2
    upsq_xmean = np.mean(up_sq, 0)
    upsq_xymean = np.mean(upsq_xmean, 0)
    upsq_xytmean = np.mean(upsq_xymean, 1)
    upsq_xyt = np.flipud(upsq_xytmean[nz / 2:])

    # vertical plot (quiver plot)
    upvert_plot = upvert[:, 0:nz / 2, -1]
    vpvert_plot = vpvert[:, 0:nz / 2, -1]
    wpvert_plot = wpvert[:, 0:nz / 2, -1]
    mag = np.sqrt(vpvert_plot ** 2 + wpvert_plot ** 2)
    vp = vpvert_plot / mag
    wp = wpvert_plot / mag

    # Save all arrays for plots

    np.save('u_xy_nw_gql1', u_xy_nw_inst)  # instantaneous slice, second to last slice
    np.save('u_xy_cl_gql1', u_xy_cl_inst)
    np.save('u_xy_NW_gql1', u_xy_NW_end)  # NW fluctuations (field - mean)*
    np.save('u_xy_CL_gql1', u_xy_CL_end)  # CL fluctuations (field - mean)*
    np.save('uh_xy_cl_gql1', uh_xy_cl)  # high modes instantaneous
    np.save('ul_xy_cl_gql1', ul_xy_cl)  # high modes instantaneous
    np.save('uvert_gql1', uvert_all)  # xz plane used for mean calculation
    np.save('ek_xy_gql1', ek_xy_timeavg2)  # 2d spectra*
    np.save('ek_x_gql1', ek_x_timeavg2)  # 1d spectra, x*
    np.save('ek_y_gql1', ek_y_timeavg2)  # 1d spectra, y*
    np.save('u_tau_gql1', u_tau)  # u_tau calculation
    np.save('uzmeanty_gql1', uzmeanty)  # velocity gradient avg in y and t*
    np.save('uzmeany_gql1', uzmeany)  # velocity gradient avg in y (for plots every time step)*
    np.save('uvmeany_gql1', uvmeany[t_turb:tslices, :])
    np.save('umeanvty_gql1', umeanvty)
    np.save('Upyt_gql1', U_pyt)
    np.save('upsq_yt_gql1', upsq_yt)
    np.save('upsq_xyt_gql1', upsq_xyt)
    np.save('upvert_plot_gql1', upvert_plot)
    np.save('vp_gql1', vp)
    np.save('wp_gql1', wp)