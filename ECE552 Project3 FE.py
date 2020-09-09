# -*- coding: utf-8 -*-
"""
Created on Tue May  10 16:04:45 2020

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
    a12 += 1e-6 # large resistor
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
    a12 += 1e-6 # large resistor
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
        elif 'C' == component[0][0]:
            extra_branch += 1
    return node_max, extra_branch

# find the initial solution with short L and open C
def read_netlist_init(netlist,total_list,node_max,G,source,x,iL):
    G[:,:] = 0
    source[:] = 0
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
        elif 'PMOS' == component[0][0:4]:
            G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
            
        elif 'R' == component[0][0]:
            G = R_stamp(G, 1/(float(component[3])*unit_conversion(component[4][0])), int(component[1])-1, int(component[2])-1)
        elif 'C' == component[0][0]:
            G, source = C_FE(G, source, int(component[1])-1, int(component[2])-1, node_max, h/(float(component[3])*unit_conversion(component[4][0])), 0, 0)
        elif 'L' == component[0][0]:
            source = L_FE(source,int(component[1])-1, int(component[2])-1,0,0,0)
    return G, source

# read the netlist, and stamp the matrix G  with update nodal voltage
def read_netlist(netlist,total_list,node_max,G,source,x,x_old_time,VA,VB,iL,h,iL_update):
    G[:,:] = 0
    source[:] = 0
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
        elif 'PMOS' == component[0][0:4]:
            G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
        elif 'R' == component[0][0]:
            G = R_stamp(G, 1/(float(component[3])*unit_conversion(component[4][0])), int(component[1])-1, int(component[2])-1)
        elif 'C' == component[0][0]:
            G, source = C_FE(G, source, int(component[1])-1, int(component[2])-1, node_max, h/(float(component[3])*unit_conversion(component[4][0])), x_old_time[int(component[1])-1,0]-x_old_time[int(component[2])-1,0], x_old_time[node_max-1,0])
            node_max += 1
        elif 'L' == component[0][0]:
            if 1 == iL_update:
                iL = iL + (x_old_time[int(component[1])-1,0]-x_old_time[int(component[2])-1,0])*h/(float(component[3])*unit_conversion(component[4][0]))
            source = L_FE(source, int(component[1])-1, int(component[2])-1, h/(float(component[3])*unit_conversion(component[4][0])), x_old_time[int(component[1])-1,0]-x_old_time[int(component[2])-1,0],iL)
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

def update_G(G,source,x,node_max):
# delete the ground node for G and source
    # G[0:5,0:5] = 0
    # source[0:5] = 0
    # for line in total_list:
    #     component = line.split()
    #     if 'NMOS' == component[0][0:4]:
    #         G, source = NMOS_stamp(G,x,source,kn,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
    #     elif 'PMOS' == component[0][0:4]:
    #         G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
    #     elif 'C' == component[0][0]:
    #         if 'BE' == method:
    #             G,source = C_BE(G, source, int(component[1]), int(component[2]), float(component[3])*unit_conversion(component[4][0])/h, x) 
    #     elif 'L' == component[0][0]:
    #         if 'BE' == method:
    #             G,source = L_BE(G, source, int(component[1]), int(component[2]), node_max, float(component[3])*unit_conversion(component[4][0])/h, x)
    A[0:node_max-1,0:node_max-1] = G[0:node_max-1,0:node_max-1]
    A[0:node_max-1,node_max-1:node_max+extra_branch-1] = G[0:node_max-1,node_max:node_max+extra_branch]
    A[node_max-1:node_max+extra_branch-1,0:node_max-1] = G[node_max:node_max+extra_branch,0:node_max-1]
    A[node_max-1:node_max+extra_branch-1,node_max-1:node_max+extra_branch-1] = G[node_max:node_max+extra_branch,node_max:node_max+extra_branch]
    s[0:node_max-1] = source[0:node_max-1]
    s[node_max-1:node_max + extra_branch-1] = source[node_max:node_max + extra_branch]
    return A,s
# G[:,node_max-1:node_max+extra_branch-1] = G[:,node_max:node_max+extra_branch]
# G[node_max-1:node_max+extra_branch-1,:] = G[node_max:node_max+extra_branch,:]

# prepare for iteration updates
def iteration_solve_init(G,source,x,x_old,epsilon,node_max,iL):
    cnt = 0
    while ((max(abs(x_old-x))>epsilon) and (cnt<100)):
        x_old = x
        node = node_max 
        G,source = read_netlist_init(netlist,total_list,node,G,source,x,iL)
        A, s = update_G(G,source,x,node_max)
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu,piv),s) 
        cnt += 1
    return x,cnt

def iteration_solve(G,source,x,x_old_iter,x_old_time,VA,VB,epsilon,node_max,iL,h):
    cnt = 0
    x_old_iter[:] = 100
    while (max(abs(x_old_iter-x))>epsilon) and (cnt<100):
        x_old_iter = x
        node = node_max
        G,source = read_netlist(netlist,total_list,node,G,source,x,x_old_time,VA,VB,iL,h,0)
        A, s = update_G(G,source,x,node_max)
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu,piv),s)     
        cnt += 1
    return x,cnt

def LTE(x_old,x,Bu,Bl,h,h_old,hmin):
    # using the voltage across CL to estimate LTE
    B = 0.5 * ( Bu + Bl )
    LTE = 2*abs(((h**2)/(h+h_old[0]))*((x[3]-x_old[3,0])/h-(x_old[3,0]-x_old[3,1])/h_old[0]))
    h_new = h*np.power(B/LTE,0.5)
    
    print(LTE,h,h_new)
    if abs(LTE)>Bu:
        return 'Reject',max(h_new,hmin)
    elif abs(LTE)<Bu and abs(LTE)>Bl:
        return 'Accept',h
    elif abs(LTE)<Bl:
        return 'Accept',h_new
    
def VA(T):
    if T<40e-9:
        return 5
    elif T<60e-9:
        return -(0.25*T*1e9)+15
    else:
        return 0

def VB(T):
    if T<20e-9:
        return T*1e9*0.25
    else:
        return VA(T)
    
################
#   Constants  #
################
kn = 1e-4 # 0.2 # 
kp = 5e-5 # 0.2 # 
lamda = 0.02
Vth = 1
epsilon = 1e-10
Bu = 1e-6
Bl = 5e-7

x = np.ndarray(13,dtype = float)
x_old_iter = np.ndarray(13,dtype = float)
x_old_time = np.ndarray((13,3),dtype = float)

h_old = [0,0,0,0]
hmin = 1e-15

# record inductor current in previous time for TR
iL = 0

# generate time samples, index of turning points
T_total = 150 # 150ns in total

plt.plot()
# for h in [3,1,0.5]: # 3ns step 1 # 1ns step 0.5 # 0.5ns step 
    #Begin of single indent
h = 1e-12
T_total = 150 # 150ns in total
T = 0.0
T_plot = np.ndarray(10000,dtype=float)
solution = np.ndarray(10000,dtype=float)
T_plot[:] = 0
solution[:] = 0


# for method in ['BE']:#, 'TR' ]: # 'FE' # 
# Begin of double indent

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
x_old_time[:,:] = 0
x_old_iter[:] = 100
G, source = read_netlist_init(netlist,total_list,node,G,source,x,iL)
node = node_max
x,cnt = iteration_solve_init(G,source,x,x_old_iter,epsilon,node,iL)


################
# main program #
################

# for each time step ii
ii = -1
iL_update = 1
while T<T_total*1e-9:
    node = node_max
    # update iL
    iL_old = iL
    G,source = read_netlist(netlist,total_list,node,G,source,x,x_old_time,VA(T+h),VB(T+h),iL,h,iL_update)
    x_old_time[:,1:3] = x_old_time[:,0:2]
    x_old_time[:,0] = x
    node = node_max
    x,cnt = iteration_solve(G, source, x, x_old_iter,x_old_time, VA(T+h), VB(T+h), epsilon,node,iL,h)
    # use constant time for first few time steps
    if ii<3:
        ii += 1
        h_old[1:4] = h_old[0:3]
        h_old[0] = h
        # np.append(T_plot,T*1e9)
        # np.append(solution,x[3])
        T += h
        T_plot[ii] = T*1e9
        solution[ii] = x[3]
        print(ii,x[3],T*1e9)
        continue
    # dynamic time step
    status,h_temp = LTE(x_old_time, x, Bu, Bl, h, h_old, hmin)
    if 'Reject' == status:
        x = x_old_time[:,0]
        x_old_time[:,0:2] = x_old_time[:,1:3]
        h = h_temp
        iL = iL_old
    else:
        ii += 1
        T += h
        h_old[1:4] = h_old[0:3]
        h_old[0] = h
        h = h_temp
        # np.append(T_plot,T*1e9)
        # np.append(solution,x[3])
        T_plot[ii] = T*1e9
        solution[ii] = x[3]
        print(ii,x[3],T*1e9)
    # end of dynamic time step

plt.plot(T, solution, label='FE w. '+str(int(h*1e9))+'ns timestep')

        # end of double indent
        # end of single indent

plt.xlabel('Time(ns)')
plt.ylabel('Output Voltage(V)')
plt.legend()
# plt.savefig(method+'.png')
plt.show()
########################
# Plotting results
########################

############################
############################
#    End of ECE552 pj3     #
############################
############################









