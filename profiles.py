import numpy as np
import matplotlib.pyplot as plt

# define grid parameters
z = np.load('z.npy')

# define plot parameters
cmap = plt.get_cmap('viridis')
ticksize = 14
fsize = 16

# Load applicable data

umeanvty_nl = np.load('umeanvty_nl.npy')
# umeanvty_gql1 = np.load('umeanvty_gql1.npy')
umeanvty_gql2 = np.load('umeanvty_gql2.npy')
umeanvty_gql3 = np.load('umeanvty_gql3.npy')
umeanvty_gql8 = np.load('umeanvty_gql8.npy')
umeanvty_ql = np.load('umeanvty_ql.npy')
uvmeany_init = np.load('uvmeany_init.npy')
u_poise = np.load('u_poise.npy')

def clmax():
    nl_max = np.amin(umeanvty_nl)
    ql_max = np.amin(umeanvty_ql)
    gql2_max = np.amin(umeanvty_gql2)
    gql3_max = np.amin(umeanvty_gql3)
    gql8_max = np.amin(umeanvty_gql8)
    return {print('nl <u_cl>ty:', nl_max), print('ql <u_cl>ty:', ql_max), print('gql2 <u_cl>ty:', gql2_max), print('gql3 <u_cl>ty:', gql3_max), print('gql8 <u_cl>ty:', gql8_max)}

def Q():
    nl_mdot = np.trapz(np.negative(umeanvty_nl), z[0, 0, :])
    ql_mdot = np.trapz(np.negative(umeanvty_ql), z[0, 0, :])
    gql2_mdot = np.trapz(np.negative(umeanvty_gql2), z[0, 0, :])
    gql3_mdot = np.trapz(np.negative(umeanvty_gql3), z[0, 0, :])
    gql8_mdot = np.trapz(np.negative(umeanvty_gql8), z[0, 0, :])
    #  return {print('nl mdot', nl_mdot), print('ql mdot: ', ql_mdot), print('gql2 mdot', gql2_mdot), print('gql3 mdot: ', gql3_mdot), print('gql8 mdot: ', gql8_mdot)}
# Create scatter plot array
    Q_normal = [nl_mdot/nl_mdot, ql_mdot/nl_mdot, gql2_mdot/nl_mdot, gql3_mdot/nl_mdot, gql8_mdot/nl_mdot]
    L = [42, 0, 2, 3, 8]

    fig, ax = plt.subplots()
    plt.scatter(L, Q_normal)
    ax.set_title('Mass Flow Rate vs. Streamwise Wavenumber Cutoff')
    ax.set_ylabel('$Q/Q_{NL}$', fontsize=fsize)
    ax.set_xlabel('$\Lambda_x$', fontsize=fsize)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # legend = ax.legend(loc='best', shadow=True)
    fig.savefig('Qplot.png')
    return fig.show()

def curve():
    # >> > x = np.array([0., 1., 1.5, 3.5, 4., 6.], dtype=float)
    # >> > np.gradient(f, x)
    # array([1., 3., 3.5, 6.7, 6.9, 2.5])
    zz = np.array([z[0, 0, 52], z[0, 0, 53], z[0, 0, 54], z[0, 0, 55], z[0, 0, 56]])
    nl = np.array([umeanvty_nl[52], umeanvty_nl[53], umeanvty_nl[54], umeanvty_nl[55], umeanvty_nl[56]])
    ql = np.array([umeanvty_ql[52], umeanvty_ql[53], umeanvty_ql[54], umeanvty_ql[55], umeanvty_ql[56]])
    gql2 = np.array([umeanvty_gql2[52], umeanvty_gql2[53], umeanvty_gql2[54], umeanvty_gql2[55], umeanvty_gql2[56]])
    gql3 = np.array([umeanvty_gql3[52], umeanvty_gql3[53], umeanvty_gql3[54], umeanvty_gql3[55], umeanvty_gql3[56]])
    gql8 = np.array([umeanvty_gql8[52], umeanvty_gql8[53], umeanvty_gql8[54], umeanvty_gql8[55], umeanvty_gql8[56]])
    grad_nl = np.gradient(nl, zz)
    grad_ql = np.gradient(ql, zz)
    grad_gql2 = np.gradient(gql2, zz)
    grad_gql3 = np.gradient(gql3, zz)
    grad_gql8 = np.gradient(gql8, zz)

    return {print('grad u_nl:  ', grad_nl), print('grad u_ql:  ', grad_ql), print('grad u_gql2:  ', grad_gql2), print('grad u_gql3:  ', grad_gql3), print('grad u_gql8:  ', grad_gql8)}

def meanp(nx):         # mean velocity profiles w/ poiseuille and nl_initial
    ld = nx / 2
    # With Poiseuille

    fig, ax = plt.subplots(1,1)
    plt.plot(-1*u_poise[0,0,:], z[0,0,:], 'k:', label='Poiseuille')
    plt.plot(-1*uvmeany_init, z[0,0,:], label='Initial Profile (t=0)')
    plt.plot(-1*umeanvty_ql, z[0, 0, :], label='$\Lambda_x$ = 0 (QL)')
    # plt.plot(-1 * umeanvty_gql1, z[0, 0, :], label='$\Lambda_x$ = 1 (GQL)')
    plt.plot(-1 * umeanvty_gql2, z[0, 0, :], label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(-1*umeanvty_gql3, z[0, 0, :], label = '$\Lambda_x$ = 3 (GQL)')
    plt.plot(-1 * umeanvty_gql8, z[0, 0, :], label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(-1*umeanvty_nl, z[0, 0, :], label = '$\Lambda_x$ = %i (NL)' %ld)

    # ax.set_title('Spanwise/Time Averaged Streamwise Velocity Profile')
    ax.set_ylabel('z', fontsize=fsize, rotation=0)
    ax.set_xlabel('$<u>_{y,t}$', fontsize=fsize)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    legend = ax.legend(loc='best', shadow=True)
    fig.savefig('umean_comp_wpoiseuille.png')
    return fig.show()



def mean(nx):
    # Without poiseuille
    ld = nx / 2
    fig, ax = plt.subplots(1,1)
    plt.plot(-1 * umeanvty_ql, z[0, 0, :], label='$\Lambda_x$ = 0 (QL)')
    # plt.plot(-1 * umeanvty_gql1, z[0, 0, :], label='$\Lambda_x$ = 1 (GQL)')
    plt.plot(-1 * umeanvty_gql2, z[0, 0, :], label='$\Lambda_x$ = 2 (GQL)')
    plt.plot(-1 * umeanvty_gql3, z[0, 0, :], label = '$\Lambda_x$ = 3 (GQL)')
    plt.plot(-1 * umeanvty_gql8, z[0, 0, :], label='$\Lambda_x$ = 8 (GQL)')
    plt.plot(-1 * umeanvty_nl, z[0, 0, :], label = '$\Lambda_x$ = %i (NL)' %ld)
    # ax.set_title('Spanwise/Time Averaged Streamwise Velocity Profile')
    ax.set_ylabel('z', fontsize=fsize, rotation=0)
    ax.set_xlabel('$<u>_{y,t}$', fontsize=fsize)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    legend = ax.legend(loc='best', shadow=True)
    plt.grid(True)
    fig.savefig('umean_comp.png')
    return fig.show()

def mean_ind():
    fig, ax = plt.subplots()
    plt.plot(np.negative(umeanvty_gql3), z[0, 0, :])
    ax.set_title('Spanwise/Time Averaged Streamwise Velocity Profile NL')
    ax.set_ylabel('z', fontsize=fsize, rotation=0)
    ax.set_xlabel('$<u>_{y,t}$', fontsize=fsize)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # legend = ax.legend(loc='best', shadow=True)
    plt.grid(True)
    return fig.show()