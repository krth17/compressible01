# -*- coding: utf-8 -*-
"""
Version 2022.04.06-1+
Edit on
@author: Keerthi
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#sp.init_printing()
#from matplotlib.backends.backend_pdf import PdfPages

# Universal constants
Runi = 8314.32


# Gas-specific constants
gamma = 1.4
Molec = 28.9645

# Derived constants
Rgas = Runi/Molec
cp = gamma*Rgas/(gamma -1)
cv = Rgas/(gamma -1)


def comm(M1, g=gamma):
    ''' Returns a commonly used function of M1 and gamma'''
    return( 1 + ((g - 1)/2)*M1**2 )

def MToCharM(M, g=gamma):
    ''' Returns characteristic Mach number (v/a*) from M'''
    return(math.sqrt( ((g+1)*M**2)/((g-1)*M**2+2)  ))

def charMToM(Mstar, g=gamma): 
    ''' Returns M from charactertistic Mach number (v/a*)'''
    return(math.sqrt( 0.5*( (g+1)/Mstar**2 - (g-1) )  ))


# Isentropic flow mass flow rate formula

def isenMByA(p0, T0, M, g=gamma, R=Rgas):
    ''' Returns mass flow rate per unit area, given stagnation conditions '''
    return(p0*M*math.sqrt(g/(R*T0))*comm(M, g)**(-(g+1)/(2*(g-1))) )
    
def isenMByA_bar(p0, T0, M, g=gamma, R=Rgas):
    ''' Returns mass flow rate per unit area, given stagnation conditions. p0 is specified in bar. '''
    return(isenMByA(p0*100000, T0, M, g=gamma, R=Rgas))
    
def isenMach(p0, T0, mByA, g=gamma, R=Rgas):
    ''' Returns Mach number, given stagnation conditions and mass flow per unit area.'''
    def MachFunc(M):
        return(mByA - p0*M*math.sqrt(g/(R*T0))*comm(M, g)**(-(g+1)/(2*(g-1))) )
    return(opt.fsolve(MachFunc, 0.1)[0])

# Normal shock relations

def norPR(M1, g=gamma):
    ''' Returns static pressure ratio p2/p1 across normal shock of given M1'''
    return( 1 + ((2*g)/(g + 1))*(M1**2 - 1) )

def norDR(M1, g=gamma):
    ''' Returns static density ratio rho2/rho1 across normal shock of given M1'''
    return( 0.5*(g + 1)*M1**2/(comm(M1, g)) )

def norVR(M1, g=gamma):
    ''' Returns velocity ratio v2/v1 across normal shock of given M1'''
    return(1/norDR(M1, g))

def norTR(M1, g=gamma):
    ''' Returns static temperature ratio T2/T1 across normal shock of given M1'''
    return(norPR(M1, g)/norDR(M1, g))

def norM2(M1, g=gamma):
    ''' Returns exit Mach number of a normal shock '''
    return(math.sqrt(comm(M1, g)/(g*M1**2 - 0.5*(g - 1))))

def norP0R(M1, g=gamma):
    ''' Returns total pressure ratio p02/p01 across normal shock of given M1'''
    return(norPR(M1,g) * (comm(norM2(M1, g), g)/comm(M1, g))**(g/(g-1)))

def norDS(M1, g=gamma, R=Rgas):
    ''' Returns specific entropy rise (s2 - s1) across normal shock of given M1 '''
    return(-R*math.log(norP0R(M1, g)))




# Rayleigh flow relations

def RayPR(M, g=gamma):
    ''' Returns static pressure ratio p/p* for Rayleigh flow with Mach no. M'''
    return((1+g)/(1 + g*M**2))

def RayTR(M, g=gamma):
    ''' Returns static temperature ratio T/T* for Rayleigh flow with Mach no. M'''
    return((M*RayPR(M, g))**2)

def RayDR(M, g=gamma):
    ''' Returns static density ratio rho/rho* for Rayleigh flow with Mach no. M'''
    return( (1/M**2)*(1/RayPR(M, g)))

def RayP0R(M, g=gamma):
    ''' Returns total pressure ratio p0/p0* for Rayleigh flow with Mach no. M'''
    return( RayPR(M, g) * ( (2 + (g-1)*M**2)/(g+1)  )**(g/(g-1))   )

def RayT0R(M, g=gamma):
    ''' Returns total temperature ratio T0/T0* for Rayleigh flow with Mach no. M'''
    return( ((g+1)*M**2)*(2 + (g-1)*M**2)/(1 + g*M**2)**2   )

def RaySolveM2(T0R, M1, g=gamma):
    ''' Returns exit Mach number M2 for Rayleigh flow, given 
    inlet Mach number M1 and total temperature ratio T02/T01'''
    def Rfunc(M2):
        return(T0R - ((1+g*M1**2)/(1+g*M2**2))**2  *  (M2/M1)**2 * (comm(M2)/comm(M1)) )
    return(abs(opt.fsolve(Rfunc, M1)[0]))
    

# Oblique shock relations

def oblTbmT(M1, beta, g=gamma):
    '''Solves for theta in the theta-beta-M relation. Theta and beta in degrees'''
    beta = beta*math.pi/180
    return( math.atan((2/math.tan(beta)) * ( (M1*math.sin(beta))**2 - 1)/(M1**2*(g + math.cos(2*beta)) + 2 ))*180/math.pi  )

def oblTbmB(M1, theta, shock='w', g=gamma):
    '''Solves for beta in the theta-beta-M relation. Theta and beta in degrees.
    By default, gives the weak (w) solution. Specify 's' for strong shock solution'''
    theta = theta*math.pi/180
    def TBMfunc(beta):
        return(math.tan(theta) - (2/math.tan(beta)) * ( (M1*math.sin(beta))**2 - 1)/(M1**2*(g + math.cos(2*beta)) + 2 ) )
    temp_var1 = opt.fsolve(TBMfunc, 0.35)[0]*180/math.pi
    temp_var2 = opt.fsolve(TBMfunc, 1.37)[0]*180/math.pi
    if shock == 'w':
        ans1 = temp_var1
    if shock == 's':
        ans1 = temp_var2
    if (abs(temp_var2-temp_var1) < 0.005) or (temp_var1) < 0 or (temp_var2 < 0):
        ans1 = "Beta solution from theta-beta-M solver yielded {:0.2f} deg and {:0.2f} deg. Possible detached shock.".format(temp_var1, temp_var2)
    return(ans1)

def oblTbmM(theta, beta, g=gamma):
    '''Solves for inlet Mach no. in the theta-beta-M relation. Theta and beta in degrees.'''
    beta = beta*math.pi/180
    theta = theta*math.pi/180
    def TBMfunc(M1):
        return(math.tan(theta) - (2/math.tan(beta)) * ( (M1*math.sin(beta))**2 - 1)/(M1**2*(g + math.cos(2*beta)) + 2 ) )
    return(opt.fsolve(TBMfunc, 2)[0])


# Prandtl-Meyer expansion waves

def nu(M, g=gamma):
    '''Outputs the value of v(M) in degrees, given M.'''
    return((math.sqrt((g+1)/(g-1))*math.atan(math.sqrt((g-1)*(M**2 -1)/(g+1))) - math.atan(math.sqrt(M**2 -1)))*180/math.pi)
    
def nuInv(nu, g=gamma):
    '''Outputs the value of M, given v(M) in degrees.'''
    def PMfunc(M, g=gamma):
        return(nu*math.pi/180 - (math.sqrt((g+1)/(g-1))*math.atan(math.sqrt((g-1)*(M**2 -1)/(g+1))) - math.atan(math.sqrt(M**2 -1))))
    return(opt.fsolve(PMfunc, 3)[0])
    
# Area-Mach number relation

def amrA(M, g=gamma):
    '''Outputs area ratio A/A* for given Mach number.'''
    return(math.sqrt((1/M**2)*(2*comm(M,g)/(g+1))**((g+1)/(g-1))))

def amrM(A, g=gamma):
    '''Outputs Mach number for given area ratio A/A*.'''
    def amrfunc(M, g=gamma):
        return(A**2 - (1/M**2)*(2*comm(M,g)/(g+1))**((g+1)/(g-1)))
    return(opt.fsolve(amrfunc, 3)[0])

    