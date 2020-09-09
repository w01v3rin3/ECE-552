# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:21:37 2020

@author: xs27
"""

import numpy as np
import math
from numpy.linalg import inv
import scipy
import scipy.linalg as linalg
import matplotlib.pyplot as plt

#############################
# Factorization and solving Ax=b
#############################
def row_switch(A,m,n):
    # switch row m & row n for matrix A
    row,col = A.shape
    t = np.ndarray((1,col),dtype=float)
    t = A[m,:]
    A[m,:] = A[n,:]
    A[n,:] = t
    return A

# need pivoting for the code here
def Eq_solver_gaussian_elimination(a,b):
    row,col = a.shape
    A = np.ndarray((row,col+1),dtype=float)
    A[:,0:col] = a
    A[:,col] = b
    for ii in range(row):
        # if 0==A[ii,ii]:
        #     temp = ii
        #     while 0==A[temp+1,ii]:
        #         temp += 1
        #     A = row_switch(A,ii,temp)
        row_max = np.argmax(A[ii:row,ii],1)
        A = row_switch(A,ii,row_max+ii)
        # pivoting row switch above 
        temp = A[ii,ii]
        A[ii,:] = A[ii,:]/temp
        for jj in range(ii+1,row):
            temp = A[jj,ii]
            A[jj,:] = A[jj,:] - A[ii,:]*temp
    for ii in range(row-1,0,-1):
        for jj in range(ii):
            temp = A[jj,ii]
            A[jj,:] = A[jj,:] - temp*A[ii,:]
    b = A[col,:]
    return b

##############################
# stamps for elements
##############################
def R_stamp(G,g,i,j):
    # i,j: node number
    G[i,i] += g
    G[i,j] -= g
    G[j,i] -= g
    G[j,j] += g
    return G

# reconstruct the stamp to seperate frequency dependet element
def C_stamp(C,c,i,j):
    # c: w(angular speed)*c(capacitance)
    # C: complex part of MNA matrix
    C[i,i] += c
    C[i,j] -= c
    C[j,i] -= c
    C[j,j] += c
    return C
    
def L_stamp(G,C,i,j,k,L):
    # i,j: related nodes
    # k: index for branch current
    # L: w*L
    G[k,i] += 1
    G[i,k] += 1
    G[k,j] -= 1
    G[j,k] -= 1
    C[k,k] -= L
    return G,C

def VS_stamp(G,s,n1,n2,b12,v):
    # b12: branch number
    # n1,n2: node number
    G[b12,n1] += 1
    G[b12,n2] -= 1
    G[n1,b12] += 1
    G[n2,b12] -= 1
    s[b12] += v
    return G, s
    
def VCVS_stamp(G,n1,n2,n3,n4,b34,alpha):
    # n1~n4: node 1234 number
    # b12,b34: branch current number
    # alpha: alpha*v34=v12
    G[b34,n3] -= alpha
    G[b34,n4] += alpha
    G[b34,n1] += 1
    G[b34,n2] -= 1
    G[n1,b34] += 1
    G[n2,b34] -= 1
    return G

# def CCVS_stamp(G,n1,n2,n3,n4,b12,b34,alpha):

# def VCCS_stamp(G,n1,n2,n3,n4,b12,b34,alpha):
    
def CCCS_stamp(G,n1,n2,n3,n4,b34,alpha):
    # i12 = alpha*i34
    G[n1,b34] += alpha
    G[n2,b34] -= alpha
    G[n3,b34] += 1
    G[n4,b34] -= 1
    G[b34,n3] += 1
    G[b34,n4] -= 1
    return G

# function unit conversion :
# uses 'Âµ' as \mu because that's how it appears in python
# read the unit conversion letter and return correct scaling 
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
    
def solve_LU(P,L,U,N,source):
    # solve the LU decomposed system
    # Lz=b : find z
    x = np.ndarray(source.shape,dtype = complex)
    x[:] = 0
    for ii in range(N):
        for jj in range(0,ii):
            source[ii] = source[ii] - L[ii,jj]*source[jj]
    
    # Ux=z : find x
    source[N-1] = source[N-1]/U[N-1,N-1]
    for ii in range(N-2,-1,-1):
        for jj in range(ii,N):
            source[ii] = source[ii] - U[ii,jj]*source[jj]
        source[ii] = source[ii]/U[ii,ii]
    for ii in range(N):
        for jj in range(N):
            x[ii] += P[ii,jj]*source[jj]
    return x

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
        if 'VCVS' == component[0]:
            extra_branch += 1
        elif 'CCCS' == component[0]:
            extra_branch += 1
        elif 'CCVS' == component[0]:
            extra_branch += 2
        elif 'L' == component[0][0]:
            extra_branch += 1
        elif 'V' == component[0][0]:
            extra_branch += 1
    return node_max, extra_branch

# read the netlist, and stamp the matrix G(real) and C(imaginary)
def read_netlist(netlist,total_list,node_max,G,C,source):
    for line in total_list:
        component = line.split()
        if 'VCVS' == component[0]:
            G = VCVS_stamp(G,int(component[1])-1,int(component[2])-1,int(component[3])-1,\
                           int(component[4])-1,node_max,float(component[5]))
            node_max += 1
        elif 'CCCS' == component[0]:
            G = CCCS_stamp(G,int(component[1])-1,int(component[2])-1,int(component[3])-1,\
                                 int(component[4])-1,node_max,float(component[5]))
            node_max += 1
        elif 'V' == component[0][0]: 
            G,source = VS_stamp(G,source,int(component[1])-1,int(component[2])-1,\
                                node_max,float(component[3]))
            node_max += 1
        elif 'C'==component[0][0]:
            C = C_stamp(C,float(component[3])*unit_conversion(component[4]),\
                        int(component[1])-1,int(component[2])-1)
        elif 'R' == component[0][0]:
            G = R_stamp(G,1/(float(component[3])*unit_conversion(component[4][0])),\
                        int(component[1])-1,int(component[2])-1)
        elif 'L' == component[0][0]:
            G,C = L_stamp(G,C,int(component[1])-1,int(component[2])-1,node_max,\
                          float(component[3])*unit_conversion(component[4][0]))
            node_max += 1
        else:
            print('Unknown item: '+component[0])
    return G,C,source

#################################################################################
#################################################################################
## pj1 : construct MNA matrix and plot bode plot
#################################################################################
#################################################################################

###########################
# MNA matrix construction #
###########################
def project1():
    # read the netlist as lines
    netlist = open('netlist_cp1.txt','r')
    total_list = netlist.readlines()
    # node = np.ndarray((len(total_list),2),dtype = int)
    
    # find the number of nodes and determine the number of extra current branches
    node_max, extra_branch = node_and_branch_number(total_list)
    
    # construct the MNA matrices: G (real), C (imaginary) and source (complex)
    G = np.ndarray((node_max+extra_branch,node_max+extra_branch), dtype = float)
    G[:,:] = 0
    C = np.ndarray((node_max+extra_branch,node_max+extra_branch), dtype = float)
    C[:,:] = 0
    source = np.ndarray(node_max+extra_branch,dtype = complex)
    source[:] = complex(0,0)
    
    # go through the netlist and stamp the G, C & source matrices 
    G, C, source = read_netlist(netlist,total_list,node_max,G,C,source)
    
    # ###########################################
    # matrix inversion/decomposition/factorization and solving
    # ###########################################
    
    # initialize output
    Vout = []
    
    # delete the reference node: node_max
    G[:,node_max-1:node_max+extra_branch-1] = G[:,node_max:node_max+extra_branch]
    G[node_max-1:node_max+extra_branch-1,:] = G[node_max:node_max+extra_branch,:]
    C[:,node_max-1:node_max+extra_branch-1] = C[:,node_max:node_max+extra_branch]
    C[node_max-1:node_max+extra_branch-1,:] = C[node_max:node_max+extra_branch,:]
    source[node_max-1:node_max + extra_branch-1] = source[node_max:node_max+extra_branch]
    
    # solve the MNA equation at each frequency
    for freq in range(13):    
        A = np.ndarray((node_max+extra_branch-1,node_max+extra_branch-1), dtype = complex)
        w = 2*math.pi*(10**freq)
        # for ii in range(10):
        #     for jj in range(10):
        A = G[0:node_max+extra_branch-1,0:node_max+extra_branch-1] + \
            1j*w*C[0:node_max+extra_branch-1,0:node_max+extra_branch-1]
        
        
        lu, piv = linalg.lu_factor(A)
        # L has diagnal elements 1 and U diagonal elements have values
        ##############
        # forward and back ward sub.
        ##############
        x = linalg.lu_solve((lu,piv),source[0:node_max+extra_branch-1])     
        # print('Hz:',freq)
        # print(x)
        # collect output
        Vout.append(x[5])
    
    #############################
    # plot Bode plot using result
    #############################
    
    plt.plot(np.abs(Vout))
    plt.xlabel('Frequency in log10')
    plt.ylabel('|H(w)| in volt')
    plt.show()
    
    plt.plot(np.angle(Vout))
    plt.xlabel('Frequency in log10')
    plt.ylabel('Phase of H(w) in rad')
    plt.show()
    return

project1()
#################################################################################
#################################################################################
## end of pj1 : construct MNA matrix and plot bode plot
#################################################################################
#################################################################################

# Possible sparse matrix code

# minimum operations
# minimum fills




