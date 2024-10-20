
"""
This python function uses finite differencing
to evaluate the neutron flux from a radiative 
source surrounded by a cylindrical body filled with water.
"""

def finite_diff(nsys, trad):
    """
    

    Parameters
    ----------
    nsys : TYPE int
        This is the number of segments used
        for the finite difference method
    trad : TYPE float
        This is the outer tank radius 

    Returns
    -------
    soln : TYPE numpy array
        This is the solution vector describing 
        the neutron flux at every evaluation segment. 

    """
    
    srad = 0.05
    d = 1/(3*(2.20+345*(1-0.324)))
    a =2.2
    rad_points=np.linspace(srad,trad,nsys+1)
    h = ((trad-srad)/nsys)
    matr = np.zeros((nsys, nsys))

    for i in range(1,nsys):
        matr[i,i-1]=-1*d*rad_points[i-1]/h**2           #  M_(i,i-1)
        matr[i,i]=2*d*rad_points[i]/h**2+rad_points[i]*a        #  M_(i,i)
        matr[i,i+1]=-1*d*rad_points[i+1]/h**2

    matr[0,0]=1
    matr[-1,-2]=(d/(2*h))*(-1)*(1/h)
    matr[-1,-1]=(d/(2*h))*1*(1/h)
    rhs = np.zeros(nsys)
    rhs[0]=1

    soln = np.linalg.solve(matr, rhs)
    return soln


############################################################################################################


"""
This Python 3 program solves the centered finite-difference approximation
of axial displacement of a bar with nonuniform modulus of elasticity
and axial body force.

The modulus of elasticity has the form E0*sqrt(1+a*x), and the body
force has the form g0*(1+c*x).  If command-line arguments are provided,
the first is used as a, and the second is used as c.  Otherwise, a
and c are set to 0.
"""

import numpy as np

def solvesys(soln,aa,cc,xr,ur,e0=1.e11,g0=2.5e8):
    """
    Function solvesys generates and solves the algebraic system
    that results from second-order centered differencing.

    Parameters
    ----------
    
    soln : 1D NumPy array (output)
        Memory space for the solution vector

    aa :  Floating-point (input)
        Coefficient for the spatial variation of the modulus

    cc :  Floating-point (input)
        Coefficient for the body force 

    xr :  Floating-point (input)
        Location of the right end of the bar (its length)

    ur :  Floating-point (input)
        Displacement of the right end of the bar

    e0 :  Optional floating point
        Factor for the modulus of elasticity

    g0 :  Optional floating point
        Factor for the body force

    Returns
    -------
    
    soln : 1D NumPy array
        The is the result from solving the system and is
        placed in the provided space.
    """
    
#   Find the system size and establish memory space for the matrix
#   and for the rhs vector.

    nsys=soln.size            #  Note that nsys=N+1.
    matr=np.zeros((nsys,nsys))
    rhs=np.zeros(nsys)

#   Loop over the interior-row locations.

    h=xr/(nsys-1)                    #  Mesh spacing
    xm=0.5*h                         #  Location of x_(1/2)
    efacm=np.sqrt(1+aa*xm)           #  Factor for E_(1/2)
    for i in range(1,nsys-1):
        xp=xm+h
        efacp=np.sqrt(1+aa*xp)       #  Factor for E_(i+1/2)
        matr[i,i-1]=-efacm           #  M_(i,i-1)
        matr[i,i]=efacm+efacp        #  M_(i,i)
        matr[i,i+1]=-efacp           #  M_(i,i+1)
        rhs[i]=g0*h**2*(1+cc*i*h)/e0 #  g_i/E0
        xm=xp                        #  Shift xp to xm for the next i
        efacm=efacp                  #  Shift efacp for the next i
    
#   Edit the first and last rows.

    matr[0,0]=1.
    matr[0,1]=0.
    matr[nsys-1,nsys-2]=0.
    matr[nsys-1,nsys-1]=1.
    rhs[nsys-1]=ur

#   Solve the algebraic system.

    soln=np.linalg.solve(matr,rhs)
    
#   Test the solution.

    tol=1.e-10
    try:
        rhs=rhs-np.dot(matr,soln)
        rhsmag=np.sqrt(np.dot(rhs,rhs))
        # print(rhsmag)
        assert(rhsmag<tol)
    except AssertionError:
        print('Solution not meeting tolerance of %e' %tol)
        raise

    return soln
    
# Main program:  The following condition means that this main program
# is only run when this module is executed.  If the module is imported
# by another program, the solvesys function is imported without running
# the main program that appears here.

if __name__ == '__main__':

    import sys
    import matplotlib.pyplot as plt

    # Have the user specify the size of the system and the value of the
    # solution at x=xr.

    xr=2.
    nseg=int(input('Enter the number of segments in x: '))
    ur=float(input('Enter the value of the displacement at x=%4.2f: ' % xr))

    # Create the space for the solution, and solve the system.

    soln=np.zeros(nseg+1)
    
    # Use the optional command-line arguments to specify the
    # nonuniform profiles.
    
    if len(sys.argv)<3:
        soln=solvesys(soln,0.,0.,xr,ur)
    else:
        soln=solvesys(soln,float(sys.argv[1]),float(sys.argv[2]),xr,ur)
    
    # Plot this solution.
    
    #height of the cantilever:
    h=np.zeros(nseg+1)+3

    xvals=np.linspace(0.,xr,nseg+1)
    plt.figure(1,figsize=[6.,4.])
    plt.subplots_adjust(left=0.17,bottom=0.13,right=0.95,top=0.95)
    plt.plot(xvals,h-soln,'k',linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('X (m)',fontsize=13)
    plt.ylabel('Displacement (m)',fontsize=13)
    
    plt.show()


