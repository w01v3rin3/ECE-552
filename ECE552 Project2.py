# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:57:00 2020

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

def VS_stamp(G,s,n1,n2,b12,v):
    # b12: branch number
    # n1,n2: node number
    G[b12,n1] += 1
    G[b12,n2] -= 1
    G[n1,b12] += 1
    G[n2,b12] -= 1
    s[b12] += v
    return G, s

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
    a12 += 1e-9 # large resistor
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
    a12 += 1e-9 # large resistor
    G[Vd,Vg] += a11
    G[Vs,Vg] -= a11
    G[Vd,Vd] += a12
    G[Vs,Vd] -= a12
    G[Vd,Vs] -= a11+a12
    G[Vs,Vs] += a11+a12
    s[Vd] += b1
    s[Vs] -= b1
    return G,s

# find number of nodes and extra banches in MNA
def node_and_branch_number(total_list):
    # return the largest number of node in the netlist
    # starting from 0 (index for python)
    # and determine extra lines for the netlist(inductor current etc)
    node_max = 0
    extra_branch = 0 
    for line in total_list:
        component = line.split()
        # find the largest node
        if int(component[1]) > node_max:
            node_max = int(component[1])
        if int(component[2]) > node_max:
            node_max = int(component[2])
        # find the number of extra branches for MNA
        if 'V' == component[0][0]:
            extra_branch += 1
    return node_max, extra_branch

# read the netlist, and stamp the matrix G(real) and C(imaginary)
def read_netlist(netlist,total_list,node_max,G,source):
    for line in total_list:
        component = line.split()
        if 'V' == component[0][0]: 
            G,source = VS_stamp(G,source,int(component[1])-1,int(component[2])-1,\
                                node_max,float(component[3]))
            node_max += 1
    return G,source

def update_G(G,source,x):
# go through the netlist and update the G & source matrices 
    G[0:5,0:5] = 0
    source[0:5] = 0
    for line in total_list:
        component = line.split()
        if 'NMOS' == component[0][0:4]:
            G, source = NMOS_stamp(G,x,source,kn,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
        elif 'PMOS' == component[0][0:4]:
            G, source = PMOS_stamp(G,x,source,kp,int(component[1])-1,int(component[2])-1,int(component[3])-1,Vth)
    A[0:node_max-1,0:node_max-1] = G[0:node_max-1,0:node_max-1]
    A[0:node_max-1,node_max-1:node_max+extra_branch-1] = G[0:node_max-1,node_max:node_max+extra_branch]
    A[node_max-1:node_max+extra_branch-1,0:node_max-1] = G[node_max:node_max+extra_branch,0:node_max-1]
    A[node_max-1:node_max+extra_branch-1,node_max-1:node_max+extra_branch-1] = G[node_max:node_max+extra_branch,node_max:node_max+extra_branch]
    s[0:node_max-1] = source[0:node_max-1]
    s[node_max-1:node_max + extra_branch-1] = source[node_max:node_max + extra_branch]
    return A,s
# G[:,node_max-1:node_max+extra_branch-1] = G[:,node_max:node_max+extra_branch]
# G[node_max-1:node_max+extra_branch-1,:] = G[node_max:node_max+extra_branch,:]

################
#   Constants  #
################

kn = 1e-4
kp = 5e-5
lamda = 0.02
Vth = 1
x = np.ndarray(8,dtype = float)
x[:] = 0
x_old = np.ndarray(8,dtype = float)
x_old[:] = 0

###########################
###########################
#  Start of the program   #
###########################
###########################

netlist = open('netlist_MOS.txt','r')
total_list = netlist.readlines()
# node = np.ndarray((len(total_list),2),dtype = int)

# find the number of nodes and determine the number of extra current branches
node_max, extra_branch = node_and_branch_number(total_list)

# construct the NA matrix: G (real) and source (real)
G = np.ndarray((node_max+extra_branch,node_max+extra_branch), dtype = float)
G[:,:] = 0
A = np.ndarray((node_max+extra_branch-1,node_max+extra_branch-1), dtype = float)
source = np.ndarray(node_max+extra_branch,dtype = float)
source[:] = 0
s = np.ndarray(node_max+extra_branch-1,dtype = float)

G, source = read_netlist(netlist,total_list,node_max,G,source)

# prepare for iteration updates
def iteration_solve(G,source,x,x_old,epsilon):
    cnt = 0
    while (max(abs(x_old-x))>epsilon) and (cnt<50):
        x_old = x
        A, s = update_G(G,source,x)
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu,piv),s)     
        cnt += 1
    return x,cnt

# Q1
# x, cnt = iteration_solve(G, source, x, x_old, 1e-9)

N = 501
solution = np.ndarray((N,8,3),dtype = float)

VA = np.linspace(0,5,N)
Vout = np.ndarray(N+1,dtype = float)

VB = np.array([2, 3.5, 5])

for jj in range(3):
    source[7] = VB[jj]
    for ii in range(N):
        x[4] = VA[ii]
        source[6] = VA[ii]
        x_old[:] = 0
        x,cnt = iteration_solve(G,source,x,x_old,1e-9)
        solution[ii,:,jj] = x


########################
# Plotting results
########################
Vout1 = solution[1:N+1,2,0]
Vout2 = solution[1:N+1,2,1]
Vout3 = solution[1:N+1,2,2]
# Vout = Vout.reshape(501)
plt.plot(VA[1:N+1], Vout1, label='VB = 2V', color = 'r')
plt.plot(VA[1:N+1], Vout2, label='VB = 3.5V', color = 'g')
plt.plot(VA[1:N+1], Vout3, label='VB = 5V', color = 'b')
plt.scatter(3,Vout1[int(3*(N+1)/5)])
plt.text(3.1,Vout1[int(3*(N+1)/5)],'%.4f V'%Vout1[int(3*(N+1)/5)],ha="left")
plt.xlabel('Input Voltage')
plt.ylabel('Output Voltage')
plt.legend()
plt.show()


############################
############################
#    End of ECE552 pj2     #
############################
############################









