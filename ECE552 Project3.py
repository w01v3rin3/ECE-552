# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:04:45 2020

@author: xs27
"""

# import ECE552_Project
# import previous stamps 
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

#####################
# stamp for DC source
#####################

def VS_stamp(G,source,n1,n2,b12,v):
    # b12: branch number
    # n1,n2: node number
    G[b12,n1] += 1
    G[b12,n2] -= 1
    G[n1,b12] += 1
    G[n2,b12] -= 1
    source[b12] += v
    return G, source

#######################
# stamp for MOSFET
#######################

def NMOS_stamp(G,x,s,k,Vg,Vs,Vd,Vth):
    # construct the stamp of a NMOS
    # Input:
    #     Vg:  node number for gate
    #     Vd:  node number for drain
    #     Vs:  node number for source
    #     k:   coefficient
    #     Vth: threshold voltage
    #     G:   NA matrix
    #     x:   current solution
    #     s:   RHS of the NA equation
    # Output:
    #     G,s: updated G & s  
    # check region of operation
    # and construct NA stamp
    # notation from (6.3.23)
    lamda = 0.02
    Vov = x[Vg]-x[Vs]-Vth
    Vds = x[Vd]-x[Vs]
    Vgs = x[Vg]-x[Vs]
    if Vgs <= Vth: # cut-off
        a11 = 0
        a12 = 0 #1e-5# large resistor 
        b1 = 0
    elif Vds < Vov: # linear
        a11 = k*Vds
        a12 = k*(Vov-Vds)
        b1 = k*(Vov*Vds-0.5*Vds*Vds)-a11*Vgs-a12*Vds
    else: # saturation
        a11 = 0.5*k*(2*Vov*(1+lamda*(Vds-Vov)-lamda*Vov*Vov))
        a12 = 0.5*k*lamda*Vov*Vov
        b1 = 0.5*k*Vov*Vov*(1+lamda*(Vds-Vov))-a11*Vgs-a12*Vds
    # update on the matrix
    a12 += 1e-8 # large resistor
    G[Vd,Vg] += a11
    G[Vs,Vg] -= a11
    G[Vd,Vd] += a12
    G[Vs,Vd] -= a12
    G[Vd,Vs] -= a11+a12
    G[Vs,Vs] += a11+a12
    s[Vd] -= b1
    s[Vs] += b1
    return G,s

def PMOS_stamp(G,x,s,k,Vg,Vs,Vd,Vth):
    # construct the stamp of a PMOS
    # Input:
    #     Vg:  node number for gate
    #     Vd:  node number for drain
    #     Vs:  node number for source
    #     k:   coefficient
    #     Vth: threshold voltage
    #     G:   NA matrix
    #     x:   current solution
    #     s:   RHS of the NA equation
    # Output:
    #     G,s: updated G & s  
    # check region of operation
    # and construct NA stamp
    # notation from (6.3.23)
    lamda = 0.02
    Vov = x[Vs]-x[Vg]-Vth
    Vsd = x[Vs]-x[Vd]
    Vsg = x[Vs]-x[Vg]
    if Vsg <= Vth: # cut-off
        a11 = 0
        a12 = 0#1e-5 # large resistor
        b1 = 0
    elif Vsd < Vov: # linear
        a11 = k*Vsd
        a12 = k*(Vov-Vsd)
        b1 = k*(Vov*Vsd-0.5*Vsd**2)-a11*Vsg-a12*Vsd
    else: # saturation
        a11 = 0.5*k*(2*Vov*(1+lamda*(Vsd-Vov)-lamda*Vov**2))
        a12 = 0.5*k*lamda*Vov**2
        b1 = 0.5*k*Vov**2*(1+lamda*(Vsd-Vov))-a11*Vsg-a12*Vsd
    # update on the matrix
    a12 += 1e-8 # large resistor
    G[Vd,Vg] += a11
    G[Vs,Vg] -= a11
    G[Vd,Vd] += a12
    G[Vs,Vd] -= a12
    G[Vd,Vs] -= a11+a12
    G[Vs,Vs] += a11+a12
    s[Vd] += b1
    s[Vs] -= b1
    return G,s

###########################
# companion model for L&C #
###########################
# C_h = C/h
def C_BE(G,source,n1,n2,C_h,v_old):
    G[n1,n1] += C_h
    G[n1,n2] -= C_h
    G[n2,n1] -= C_h
    G[n2,n2] += C_h
    source[n1] += C_h * v_old
    source[n2] -= C_h * v_old
    return G,source

# C_h = 2*C/h
def C_TR(G,source, n1,n2,C_h,v_old,i_old):
    G[n1,n1] += C_h
    G[n1,n2] -= C_h
    G[n2,n1] -= C_h
    G[n2,n2] += C_h
    source[n1] += C_h*v_old+i_old
    source[n2] -= C_h*v_old+i_old
    return G,source

# h_C = h/C
def C_FE(G,source,n1,n2,b12,h_C,v_old,i_old):
    G[n1,b12] += 1
    G[n2,b12] -= 1
    G[b12,n1] += 1
    G[b12,n2] -= 1
    source[b12] += v_old+h_C*i_old
    return G,source

# L_h = L/h
def L_BE(G,source,n1,n2,iL,L_h,i_old):
    G[n1,iL] += 1
    G[n2,iL] -= 1
    G[iL,n1] += 1
    G[iL,n2] -= 1
    G[iL,iL] -= L_h
    source[iL] -= L_h*i_old
    return G,source

# L_h = 2L/h
def L_TR(G,source,n1,n2,b12,L_h,v_old,i_old):
    G[n1,b12] += 1
    G[n2,b12] -= 1
    G[b12,n1] += 1
    G[b12,n2] -= 1
    G[b12,b12] -= L_h
    source[b12] -= L_h*i_old+v_old
    return G,source

# h_L = h/L
def L_FE(source,n1,n2,h_L,v_old,i_old):
    source[n1] -= i_old+h_L*v_old
    source[n2] += i_old+h_L*v_old
    return source

# R stamp
def R_stamp(G,g,i,j):
    # i,j: node number
    G[i,i] += g
    G[i,j] -= g
    G[j,i] -= g
    G[j,j] += g
    return G
#################################################
# find number of nodes and extra banches in MNA
def node_and_branch_number(total_list):
    # return the largest number of node in the netlist
    # starting from 0 (index for python)
    # and determine extra lines for the netlist(inductor current etc)
    node_max = 0
    extra_branch = 0 
    for line in total_list:
        component = line.split()
        node_max = max(node_max,int(component[1]), int(component[2]))
        # find the number of extra branches for MNA
        if 'V' == component[0][0]:
            extra_branch += 1
        elif 'L' == component[0][0]:
            extra_branch += 1
    return node_max, extra_branch

# find the initial solution with short L and open C
def read_netlist_init(netlist,total_list,node_max,G,source,x,iC,method):
    G[:,:] = 0
    source[:] = 0
    C_index = 0
    for line in total_list:
        component = line.split()
        if 'VA' == component[0]: 
            G,source = VS_stamp(G,source,int(component[1])-1,int(component[2])-1, node_max,float(component[3]))
            node_max += 1
        elif 'VB' == component[0]:
            G,source = VS_stamp(G, source, int(component[1])-1, int(component[2])-1, node_max, float(component[3]))
            node_max +=1
        elif 'V' == component[0][0]:
            G,source = VS_stamp(G, source, int(component[1])-1, int(component[2])-1, node_max, float(component[3]))
            node_max +=1
        elif 'NMOS' == component[0][0:4]:
            G, source = NMOS_stamp(G,x,source,kn,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
            if 'BE' == method:
                # Cgs
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, 0, 0)
                # Cgd
                G,source = C_BE(G, source, int(component[1])-1, int(component[3])-1, 0, 0)
            elif 'TR' == method:
                # Cgs
                iC[C_index] = 0 
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 0, 0, iC[C_index])
                C_index += 1
                # Cgd
                iC[C_index] = 0
                G,source = C_TR(G, source, int(component[1])-1, int(component[3])-1, 0, 0, iC[C_index])
                C_index += 1
        elif 'PMOS' == component[0][0:4]:
            G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
            if 'BE' == method:
                # Cgs
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, 0, 0)
                # Cgd
                G,source = C_BE(G, source, int(component[1])-1, int(component[3])-1, 0, 0)
            elif 'TR' == method:
                # Cgs
                iC[C_index] = 0
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 0, 0, iC[C_index])
                C_index += 1
                # Cgd
                iC[C_index] = 0
                G,source = C_TR(G, source, int(component[1])-1, int(component[3])-1, 0, 0, iC[C_index])
                C_index += 1
        elif 'R' == component[0][0]:
            G = R_stamp(G, 1/(float(component[3])*unit_conversion(component[4][0])), int(component[1])-1, int(component[2])-1)
        elif 'C' == component[0][0]:
            if 'BE' == method:
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, 0, 0) 
            elif 'TR' == method:
                iC[C_index] = 0
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 0, 0, iC[C_index])
                C_index += 1
        elif 'L' == component[0][0]:
            if 'BE' == method:
                G,source = L_BE(G, source, int(component[1])-1, int(component[2])-1, node_max, 0, 0)
            elif 'TR' == method:
                G,source = L_TR(G, source, int(component[1])-1, int(component[2])-1, node_max, 0, 0, 0)
            node_max += 1
    return G, source

# read the netlist, and stamp the matrix G  with update nodal voltage
def read_netlist(netlist,total_list,node_max,G,source,x,x_old_time,VA,VB,iC,method,h,update_iC):
    G[:,:] = 0
    source[:] = 0
    C_index = 0
    for line in total_list:
        component = line.split()
        if 'VA' == component[0]: 
            G,source = VS_stamp(G,source,int(component[1])-1,int(component[2])-1, node_max,VA)
            node_max += 1
        elif 'VB' == component[0]:
            G,source = VS_stamp(G, source, int(component[1])-1, int(component[2])-1, node_max, VB)
            node_max +=1
        elif 'V' == component[0][0]:
            G,source = VS_stamp(G, source, int(component[1])-1, int(component[2])-1, node_max, float(component[3]))
            node_max +=1
        elif 'NMOS' == component[0][0:4]:
            G, source = NMOS_stamp(G,x,source,kn,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
            if 'BE' == method:
                # Cgs
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, 1e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1])
                # Cgd
                G,source = C_BE(G, source, int(component[1])-1, int(component[3])-1, 1e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[3])-1])
            elif 'TR' == method:
                # Cgs
                if 1 == update_iC:
                    iC[C_index] = (2e-14/h)*(x[int(component[1])-1]-x[int(component[2])-1]-x_old_time[int(component[1])-1]+x_old_time[int(component[2])-1])-iC[C_index] 
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 2e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1], iC[C_index])
                C_index += 1
                # Cgd
                if 1 == update_iC:
                    iC[C_index] = (2e-14/h)*(x[int(component[1])-1]-x[int(component[3])-1]-x_old_time[int(component[1])-1]+x_old_time[int(component[3])-1])-iC[C_index]
                G,source = C_TR(G, source, int(component[1])-1, int(component[3])-1, 2e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[3])-1], iC[C_index])
                C_index += 1
        elif 'PMOS' == component[0][0:4]:
            G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
            if 'BE' == method:
                # Cgs
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, 1e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1])
                # Cgd
                G,source = C_BE(G, source, int(component[1])-1, int(component[3])-1, 1e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[3])-1])
            elif 'TR' == method:
                # Cgs
                if 1 == update_iC:
                    iC[C_index] = (2e-14/h)*(x[int(component[1])-1]-x[int(component[2])-1]-x_old_time[int(component[1])-1]+x_old_time[int(component[2])-1])-iC[C_index]
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 2e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1], iC[C_index])
                C_index += 1
                # Cgd
                if 1 == update_iC:
                    iC[C_index] = (2e-14/h)*(x[int(component[1])-1]-x[int(component[3])-1]-x_old_time[int(component[1])-1]+x_old_time[int(component[3])-1])-iC[C_index]
                G,source = C_TR(G, source, int(component[1])-1, int(component[3])-1, 2e-14/h, x_old_time[int(component[1])-1]-x_old_time[int(component[3])-1], iC[C_index])
                C_index += 1
        elif 'R' == component[0][0]:
            G = R_stamp(G, 1/(float(component[3])*unit_conversion(component[4][0])), int(component[1])-1, int(component[2])-1)
        elif 'C' == component[0][0]:
            if 'BE' == method:
                G,source = C_BE(G, source, int(component[1])-1, int(component[2])-1, float(component[3])*unit_conversion(component[4][0])/h, (x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1])) 
            elif 'TR' == method:
                if 1 == update_iC:
                    iC[C_index] = (2*float(component[3])*unit_conversion(component[4][0])/h)*(x[int(component[1])-1]-x[int(component[2])-1]-x_old_time[int(component[1])-1]+x_old_time[int(component[2])-1])-iC[C_index]
                G,source = C_TR(G, source, int(component[1])-1, int(component[2])-1, 2*float(component[3])*unit_conversion(component[4][0])/h, (x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1]), iC[C_index])
                C_index += 1
        elif 'L' == component[0][0]:
            if 'BE' == method:
                G,source = L_BE(G, source, int(component[1])-1, int(component[2])-1, node_max, float(component[3])*unit_conversion(component[4][0])/h, x_old_time[node_max-1])
            elif 'TR' == method:
                G,source = L_TR(G, source, int(component[1])-1, int(component[2])-1, node_max, 2*float(component[3])*unit_conversion(component[4][0])/h, x_old_time[int(component[1])-1]-x_old_time[int(component[2])-1],x_old_time[node_max-1])
            node_max += 1
    return G,source

# convert element unit
def unit_conversion(unit):
    if 'Âµ' == unit[0:2]:
        return 1e-6
    if 'p' == unit[0]:
        return 1e-12
    elif 'K' == unit[0]:
        return 1e3
    elif 'm' == unit[0]:
        return 1e-3
    else:
        return 1

def update_G(G,source,x,method,node_max):
# delete the ground node for G and source
    A[0:node_max-1,0:node_max-1] = G[0:node_max-1,0:node_max-1]
    A[0:node_max-1,node_max-1:node_max+extra_branch-1] = G[0:node_max-1,node_max:node_max+extra_branch]
    A[node_max-1:node_max+extra_branch-1,0:node_max-1] = G[node_max:node_max+extra_branch,0:node_max-1]
    A[node_max-1:node_max+extra_branch-1,node_max-1:node_max+extra_branch-1] = G[node_max:node_max+extra_branch,node_max:node_max+extra_branch]
    s[0:node_max-1] = source[0:node_max-1]
    s[node_max-1:node_max + extra_branch-1] = source[node_max:node_max + extra_branch]
    return A,s

# prepare for iteration updates
def iteration_solve_init(G,source,x,x_old,epsilon,node_max,iC,method):
    cnt = 0
    while (max((abs(x_old-x)))>epsilon and (cnt<100) ):
        x_old = x
        node = node_max 
        G,source = read_netlist_init(netlist,total_list,node,G,source,x,iC,method)
        A, s = update_G(G,source,x,method,node_max)
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu,piv),s) 
        cnt += 1
    return x,cnt

def iteration_solve(G,source,x,x_old_iter,x_old_time,VA,VB,epsilon,node_max,iC,method,h):
    cnt = 0
    x_old_iter[:] = 100
    while (max(abs(x_old_iter-x))>epsilon) and (cnt<100) :
        x_old_iter = x
        node = node_max
        G,source = read_netlist(netlist,total_list,node,G,source,x,x_old_time,VA,VB,iC,method,h,0)
        A, s = update_G(G,source,x,method,node_max)
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu,piv),s)
        cnt += 1
    return x,cnt

def LTE(x_old,x,method,Bu,Bl,h,h_old,hmin):
    # using the voltage across CL to estimate LTE
    jj = 3 
    B = 0.5 * ( Bu + Bl )
    if 'BE' == method:
        LTE = (h**2/(h+h_old[0]))*((x[jj]-x_old[jj,0])/h-(x_old[jj,0]-x_old[jj,1])/h_old[0])
        h_new = np.power(B/(LTE/h**2),0.5)
    elif 'TR' == method:
        LTE = (0.5*h**3/(h+h_old[0]+h_old[1]))*\
            (((x[jj]-x_old[jj,0])/h-(x_old[jj,0]-x_old[jj,1])/h_old[0])/(h+h_old[0])-\
            ((x_old[jj,0]-x_old[jj,1])/h_old[0]-(x_old[jj,1]-x_old[jj,2])/h_old[1])/(h_old[0]+h_old[1]))
        h_new = np.power(B/(LTE/h**3),1/3)
    if abs(LTE)>Bu:
        return 'Reject',h_new
    elif abs(LTE)<Bu and abs(LTE)>Bl:
        return 'Accept',h
    elif abs(LTE)<Bl:
        return 'Accept',h_new
    
################
#   Constants  #
################

kn = 1e-4 # 0.2 # 
kp = 5e-5 # 0.2 # 
Vth = 1
epsilon = 1e-8
Bu = 0.08
Bl = 0.02

x = np.ndarray(11,dtype = float)
x_old_iter = np.ndarray(11,dtype = float)
x_old_time = np.ndarray(11,dtype = float)

h_old = 0
hmin = 1e-12

# record capacitor current in previous time for TR
iC = np.ndarray(11,dtype = float)
iC[:] = 0

# generate time samples, index of turning points
T_total = 150 # 150ns in total

plt.plot()
# for h in [3,1,0.5]: # 3ns step 1 # 1ns step 0.5 # 0.5ns step 
    #Begin of single indent
h = 0.85
h_plot = h
# generate input waveform for static time step
N = int(T_total/h)+1
N20 = int(20/h) + 1
N40 = int(40/h) + 1
N60 = int(60/h) + 1
VA = np.ndarray(N,dtype = float)
VA[:] = 0
VB = np.ndarray(N,dtype = float)
VB[:] = 0
T = np.linspace(0,150,N)

VA[0:N40] = 5
VB[N20:N40] = 5
for ii in range(N20):
    VB[ii] = 0.25*ii*h
    VB[ii+N40] = 5 - VB[ii]
    VA[ii+N40] = VB[ii+N40]
# waveform verification
# plt.plot()
# plt.plot(VA, color = 'r')
# plt.plot(VB, color = 'g')
# plt.show()

h = h*1e-9

# for method in ['BE']:#, 'TR' ]: # 'FE' # 
# Begin of double indent
method =  'BE' #    'TR' # 

###########################
###########################
#   Start of the program  #
#    ECE 552 Project 3    #
###########################
###########################

# open and read netlist
netlist = open('netlist_pj3.txt','r')
total_list = netlist.readlines()

# find the number of nodes and determine the number of extra current branches
node_max, extra_branch = node_and_branch_number(total_list)

# construct the NA matrix: G (real) and source (real)
G = np.ndarray((node_max+extra_branch,node_max+extra_branch), dtype = float)
A = np.ndarray((node_max+extra_branch-1,node_max+extra_branch-1), dtype = float)
source = np.ndarray(node_max+extra_branch,dtype = float)
s = np.ndarray(node_max+extra_branch-1,dtype = float)

# find initial solution with open C and short L
node = node_max
x[:] = 0
x_old_time[:] = 0
x_old_iter[:] = 100
iC[:] = 0
G, source = read_netlist_init(netlist,total_list,node,G,source,x,iC,method)
node = node_max
x,cnt = iteration_solve_init(G,source,x,x_old_iter,epsilon,node,iC,method)
solution = np.ndarray(N,dtype = float)
solution[:] = 0

################
# main program #
################

# for each time step ii
for ii in range(N):
    node = node_max
    # update capacitor current
    G,source = read_netlist(netlist,total_list,node,G,source,x,x_old_time,VA[ii],VB[ii],iC,method,h,1)
    x_old_time[:] = x
    if 0 == ii:
        x_old_time[:] = 0
    node = node_max
    x,cnt = iteration_solve(G, source, x, x_old_iter,x_old_time, VA[ii], VB[ii], epsilon,node,iC,method,h)
    solution[ii] = x[3]
    # print(ii,cnt,solution[ii])

# solution[2:6] = 0
plt.plot(T, solution, label=method + ' w. '+str(h_plot)+'ns timestep')

        # end of double indent
        # end of single indent

plt.xlabel('Time(ns)')
plt.ylabel('Output Voltage(V)')
plt.legend()
plt.savefig(method+'_fixed_time_step.png')
plt.show()
########################
# Plotting results
########################

############################
############################
#    End of ECE552 pj3     #
############################
############################









