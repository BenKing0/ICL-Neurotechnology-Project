import sys
sys.path.append(__file__)
import numpy as np
import random
from tqdm import trange

def HermiteRodriguez(tau,t,a):
    x = (tau - t) / a
    return 2*x*np.exp(-x**2)

def MexicanHat(tau,t,a):
    x = tau - t
    return 2/(np.sqrt(3*a)*np.pi**0.25)*(1-(x/a)**2)*np.exp(-0.5*(x/a)**2)

class CWT_filter_bank:
    def __init__(self,x,f,a,fs,T):
        self.fs = fs # sampling frequency
        self.x = x
        self.T = T
        self.CWT = np.zeros(len(self.x))
        for i in trange(len(self.x)): # for each `value` of t:
            t = i / fs
            filter_bank = self.make_CWT(t,f,a)
            self.CWT[i] = max(filter_bank)
    
    def make_CWT(self,t,f,a):
        # Each is the values of the len(a) arrays at that `t` value, integrates over all tau:
        filter_bank = np.zeros(len(a))
        for i in range(len(a)):
            filter_bank[i] = self.integrate(f,a[i],t)
            filter_bank[i] *= 1/(a[i]**0.5)
        # Returns the bank of filters for each t:
        return np.array(filter_bank)
    
    def integrate(self,f,a_val,t):
        '''
        integrate phi((tau-t)/a)*x(tau) over tau from -inf (practically 0 as time in x starts from 0),
        to inf (practically T*fs as x is 0 after signal ends). Monte Carlo won't work here due to two 
        functions on different domains.
        '''
        dtau = 1 / self.fs # see below (phi_domain is the tau inputs)
        phi_domain = np.linspace(0,self.T,self.fs*self.T) # outside of this domain x is 0.
        phi = f(phi_domain,t,a_val)
        integrand = np.multiply(self.x,phi)
        area = (dtau / 2) * (2 * sum(integrand[1:-1]) + (integrand[0] + integrand[-1]))
        return area
    
    def __call__(self):
        return self.CWT


