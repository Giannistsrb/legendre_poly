from scipy.special import legendre
from scipy.interpolate import interp1d
from scipy.integrate import trapz

import numpy as np
import math
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Inputs:

#We use the two body scattering law (information in file 6 - ENDF: https://t2.lanl.gov/nis/endf/law2for6.html )
#(2l+1)/2 matrix from 0 to len(coeff):

A0 = 1/2 #First order (for legendre polynomial P(0)=1, (2*i+1)/2 = (2*0 + 1)/2 = 1/2)

A = [(2*(i+1)+1)/2  for i in range(10)] #Other orders for Polynomials P1(x) ... P10(x)  

print("A = ", A)

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Legendre coefficients according to the beam energy (from ENDF):
#We get the sum of (2i+1)/2 * A[i] for i=1,...,10 
coeffs_900keV   = A * np.array([2.176400e-2, -1.012400e-2, 6.253200e-4, 3.620300e-4, 1.328400e-4, 2.337900e-5, 2.116500e-6, 1.396000e-7, 3.946800e-9, 5.42430e-10])
coeffs_2000keV  = A * np.array([5.173400e-2, 4.230700e-3, 5.734300e-3, 3.878700e-3, 2.020600e-3, 5.367600e-4, 7.651200e-5, 7.532800e-6, 3.165600e-7, 7.121200e-8])
coeffs_2600keV  = A * np.array([6.533200e-2, 3.406900e-2, 1.107900e-2, 7.209800e-3, 4.387700e-3, 1.328400e-3, 2.241200e-4, 2.496400e-5, 1.190200e-6, 3.233400e-7])
                                            
#-------------------------------------------------------------------------------------------
#Legendre polynomials for each coefficient, and for each energy: 
#We multiply the sum of (2i+1)/2 * A[i] for i=0,...,9 with Legendre polynomials P1,...,P10 = legendre(i+1):
LM_900keV = [np.poly1d(coeffs_900keV[i]*legendre(i+1)) for i in range(len(coeffs_900keV))]
LM_2000keV = [np.poly1d(coeffs_2000keV[i]*legendre(i+1)) for i in range(len(coeffs_2000keV))]
LM_2600keV = [np.poly1d(coeffs_2600keV[i]*legendre(i+1)) for i in range(len(coeffs_2600keV))]

#-------------------------------------------------------------------------------------------
#Final Polynomials function for each energy:
differential_cross_section_function_900keV  = A0 + sum(LM_900keV)
differential_cross_section_function_2000keV = A0 + sum(LM_2000keV)
differential_cross_section_function_2600keV = A0 + sum(LM_2600keV)

#-------------------------------------------------------------------
#Print the polynomial functions for 2.10MeV, 2.90MeV and 3.45MeV:
print("dσ/dΩ (900keV)  =  ", differential_cross_section_function_900keV)
print("dσ/dΩ (2000keV) =  ", differential_cross_section_function_2000keV)
print("dσ/dΩ (2600keV) =  ", differential_cross_section_function_2600keV)

#-------------------------------------------------------------------------------------------
#plot the function in terms of cos(theta) for 2.10MeV:
theta  = np.linspace(0, math.pi, 100)
cos_theta = np.cos(theta)
y_900keV       = [differential_cross_section_function_900keV(ct)  for ct in cos_theta]
y_2000keV      = [differential_cross_section_function_2000keV(ct) for ct in cos_theta]
y_2600keV      = [differential_cross_section_function_2600keV(ct) for ct in cos_theta]
plt.plot(theta * 180 / math.pi, y_900keV)
plt.plot(theta * 180 / math.pi, y_2000keV)
plt.plot(theta * 180 / math.pi, y_2600keV)
plt.legend(["$E_{d}=900keV$", "$E_{d}=1.98MeV$", "$E_{d}=2.64MeV$"], loc = 0, frameon = True)
plt.xlabel("θ [μοίρες]")
plt.ylabel("Διαφορική ενεργός διατομή dσ/dΩ [barns/sr]")
plt.title("Αντίδραση (d,n) στο $^{3}H$")
plt.grid()
plt.show() 

#See if they are normalized. We calculate the integrals from -1 to 1 for each energy: 
x = np.linspace(-1, 1, 100)
Y_900keV  = [differential_cross_section_function_900keV(ct) for ct in x] #dσ/dΩ for 900keV
Y_2000keV = [differential_cross_section_function_2000keV(ct) for ct in x] #dσ/dΩ for 900keV
Y_2600keV = [differential_cross_section_function_2600keV(ct) for ct in x] #dσ/dΩ for 900keV
print("Integral for 0.9MeV = ", trapz(Y_900keV, x))
print("Integral for 1.98MeV = ", trapz(Y_2000keV, x))
print("Integral for 2.64MeV = ", trapz(Y_2600keV, x))








