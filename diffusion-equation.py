import numpy as np      # External library for numerical calculations
import matplotlib.pyplot as plt   # Plotting library
import matplotlib.animation as animation # Plotting animation


# Function defining the initial and analytic solution
def initialBell(x):
    return np.where(x%1. < 0.5, np.power(np.sin(2*x*np.pi), 2), 0)

#plot the diffusion equation using the FTCS update scheme
def plot_diffusion(nt):
    # Setup space, initial phi profile and Courant number
    nx = 40                 # number of points in space
    nt = 250                 # number of time steps
    K = 0.05                 # diffusion term

    # derived quantities
    dx = 1./nx
    dt = np.power(dx, 2)/K
    t = nt*dt    

    eta =  0.03#((K * dt)/(np.power(dx,2)))


    # Spatial variable going from zero to one inclusive
    x = np.linspace(0.0, 1.0, nx+1)
    # Three time levels of the dependent variable, phi
    phi = initialBell(x)
    phiNew = phi.copy()
    phiOld = phi.copy()

    # FTCS for the first time-step, looping over space
    for j in range(1,nx):
        phi[j] = phiOld[j] + eta*(phiOld[j+1] - 2*phiOld[j] + phiOld[j-1])

    # apply periodic boundary conditions
    phi[0] = phiOld[0] +  eta*(phiOld[1] - 2*phiOld[0] + phiOld[nx-1])
    phi[nx] = phi[0]

    # Loop over remaining time-steps (nt) using CTCS
    for n in range(1,nt):
        # loop over space
        for j in range(1,nx):
            phiNew[j] = phiOld[j] + eta*(phiOld[j+1] - 2*phiOld[j] + phiOld[j-1])
        # apply periodic boundary conditions
        phiNew[0] = phiOld[0] + eta*(phi[1] - 2*phi[0] + phi[nx-1])
        phiNew[nx] = phiNew[0]
        #update phi for the next time-step
        phiOld = phi.copy()
        phi = phiNew.copy()



    plt.plot(x, initialBell(x), 'k', label='Initial condition')

    # Plot the solution in comparison to the analytic solution
    #plt.plot(x, initialBell(x - u*t), 'r', label='analytic solution')
    plt.plot(x, phi, 'b', label='CTCS')
    #plt.legend(loc='best')
    #plt.ylabel('$\phi$')
   # plt.axhline(0, linestyle=':', color='black')
    plt.show()
#fig, ax = plt.subplots()
#ani = animation.FuncAnimation(fig, plot_diffusion, np.arange(1, 10), #init_func=init,
#                              interval=25, blit=True)
#plt.show()

# calculate the diffusion term using j+2 and j-2 terms
def FTCS_noncompact(eta, nx, nt, ini):
    # Spatial variable going from zero to one inclusive
    x = np.linspace(0.0, 1.0, nx+1)
    # Three time levels of the dependent variable, phi
    phi = ini
    phiNew = ini.copy()
    phiOld = ini.copy()

    # FTCS for the first time-step, looping over space
    for j in range(1,nx-1):
        phi[j] = phiOld[j] + (eta/4)*(phiOld[j+2] + phiOld[j-2] - 2*phiOld[j])

    # apply periodic boundary conditions
    phi[0] = phiOld[0] +  (eta/4)*(phiOld[2] + phiOld[nx-2] - 2*phiOld[0])
    phi[1] = phiOld[1] +  (eta/4)*(phiOld[3] + phiOld[nx-1] - 2*phiOld[1])
    phi[nx-1] = phiOld[nx-1] +  (eta/4)*(phiOld[1] + phiOld[nx-3] - 2*phiOld[nx-1])
    phi[nx] = phi[0]

    # Loop over remaining time-steps (nt) using CTCS
    for n in range(1,nt):
        # loop over space
        for j in range(1,nx-1):
            phiNew[j] = phiOld[j] + (eta/4)*(phiOld[j+2] + phiOld[j-2] - 2*phiOld[j])
        # apply periodic boundary conditions
        phiNew[0] = phiOld[0] + (eta/4)*(phiOld[2] + phiOld[nx-2] - 2*phiOld[0])
        phiNew[1] = phiOld[1] +  (eta/4)*(phiOld[3] + phiOld[nx-1] - 2*phiOld[1])
        phiNew[nx-1] = phiOld[nx-1] +  (eta/4)*(phiOld[1] + phiOld[nx-3] - 2*phiOld[nx-1])
        phiNew[nx] = phiNew[0]
        #update phi for the next time-step
        phiOld = phi.copy()
        phi = phiNew.copy()

    return phi


def FTCS_compact(eta, nx, nt, ini):
    phi = ini
    phiNew = ini.copy()
    phiOld = ini.copy()

    # FTCS for the first time-step, looping over space
    for j in range(1,nx):
        phi[j] = phiOld[j] + eta*(phiOld[j+1] - 2*phiOld[j] + phiOld[j-1])

    # apply periodic boundary conditions
    phi[0] = phiOld[0] +  eta*(phiOld[1] - 2*phiOld[0] + phiOld[nx-1])
    phi[nx] = phi[0]

    # Loop over remaining time-steps (nt) using CTCS
    for n in range(1,nt):
        # loop over space
        for j in range(1,nx):
            phiNew[j] = phiOld[j] + eta*(phiOld[j+1] - 2*phiOld[j] + phiOld[j-1])
        # apply periodic boundary conditions
        phiNew[0] = phiOld[0] + eta*(phi[1] - 2*phi[0] + phi[nx-1])
        phiNew[nx] = phiNew[0]
        #update phi for the next time-step
        phiOld = phi.copy()
        phi = phiNew.copy()

    return phi





# Put everything inside a main function to avoid global variables
#use the Euler method for calculating the diffusion term
def main():
    # Setup space, initial phi profile and Courant number
    nx = 40                 # number of points in space
    nt = 250                 # number of time steps
    K = 0.05                 # diffusion term


    # Spatial variable going from zero to one inclusive
    x = np.linspace(0.0, 1.0, nx+1)
    x_hi_res = np.linspace(0.0, 1.0, nx*10+1)
    # Three time levels of the dependent variable, phi
    phi = initialBell(x)
    phi_hi_res = initialBell(x_hi_res)

    # derived quantities
    dx = 1./nx
    dt = np.power(dx, 2)/K
    t = nt*dt    

    eta =  0.03#((K * dt)/(np.power(dx,2)))

    x = np.linspace(0.0, 1.0, nx+1)



    #calculate phi using the Euler and Shaun methods
    #which are different ways of doing the analysis
    phi_Euler = FTCS_compact(eta, nx, nt, phi)
    phi_Shaun = FTCS_noncompact(eta, nx, nt, phi)
    # Calculate a high resolution pseudo analytic solution
    phi_Anal = FTCS_compact(eta, nx*10, nt, phi_hi_res)


    plt.plot(x, initialBell(x), 'k', label='Initial condition')

    # Plot the solution in comparison to the analytic solution
    #plt.plot(x, initialBell(x - u*t), 'r', label='analytic solution')
    plt.plot(x, phi_Euler, 'b', label='FTCS compact')
    plt.plot(x, phi_Shaun, 'g', label='FTCS non-compact')
    plt.plot(x_hi_res, phi_Anal, 'r', label='Analytic')
    plt.legend(loc='best')
    plt.ylabel('$\phi$')
    plt.axhline(0, linestyle=':', color='black')
    plt.show()



# Execute the code
main()







# CMSS
