# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 01:31:59 2019

@author: batuhan
"""

import numpy as np
#import matplotlib.pyplot as plt
import numpy.linalg as lin

r=0.01     # radius ,m
area=np.pi*(r**2) # m^2
E=2.0e8   # kN/m^2
Iy=(np.pi/4.0)*r**4 # m^4   
Iz=(np.pi/4.0)*r**4
It=(np.pi/2.0)*r**4 # m^4
G=E/2     # kN/m^2

XYZ_values=np.array([0.0,0.0,0.0, 0.0,3.0,0.0, 3.0,3.0,0.0, 6.0,3.0,0.0, 6.0,0.0,0.0, 3.0,3.0,-3.0, 3.0,0.0,-3.0],dtype=float) # (in m)
NC_values=np.array([1,2, 2,3, 3,4, 3,6, 4,5, 6,7],dtype=float)
#MG_values=np.array([36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12, 36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12, 36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12, 36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12, 36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12, 36.0e-4,24.0e-4,60.0e-4,20.0e7,79.3e-3,0.12],dtype=float)

BC_values=np.array([1,1,1,1,1,1,1, 5,1,1,1,1,1,1, 7,1,1,1,1,1,1],dtype=int)
FEXT_values=np.array([3,0.0,-5.0,0.0,0.0,0.0,0.0])  # (in kN)
AN_values=np.array([-1.0,0.0,0.0, 0.0,4.0,0.0, 3.0,4.0,0.0,  7.0,3.0,0.0, 4.0,3.0,-3.0],dtype=float)                                                      
NC_AN_values=np.array([1,2,1, 2,3,2, 3,4,3, 3,6,3, 4,5,4, 6,7,5],dtype=int)

n=7              # joint number
m=6              # member number


# FUNCTIONS
# defining createMatrix function to create matrices by using row number,column number and values
def createMatrix(row,column,values):

    matrix=np.zeros([row, column], dtype =float)
    counter = 0                      # counter for element number in values
    for i in range(0,row,1):
        for j in range(0,column,1): 
            matrix[i][j] = values[counter]  
            counter = counter + 1 
    return matrix

def find_V1(matrix):
   
    x=matrix[:,0]
    y=matrix[:,1]
    z=matrix[:,2]
    
    V1=np.zeros((row_NC,3),dtype=float)
        
    for i in range(0,row_NC,1):
        
        index1=int(NC[i,0]-1)
        index2=int(NC[i,1]-1)
        start_x = x[index1] 
        end_x = x[index2]
        start_y = y[index1]
        end_y = y[index2]
        start_z = z[index1]
        end_z = z[index2]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        delta_z = end_z - start_z
        len_V1 =np.sqrt(delta_x*delta_x+delta_y*delta_y+delta_z*delta_z)
        
        V1[i,0]=delta_x/len_V1
        V1[i,1]=delta_y/len_V1
        V1[i,2]=delta_z/len_V1
        
    return V1

def find_V2(index_matrix,value_matrix1,value_matrix2):
    
    x1=value_matrix1[:,0]
    y1=value_matrix1[:,1]
    z1=value_matrix1[:,2]
    
    x2=value_matrix2[:,0]
    y2=value_matrix2[:,1]
    z2=value_matrix2[:,2]
    
    V2=np.zeros((row_NC,3),dtype=float)
    
    
    for i in range(0,row_NC,1):
        
        index1=int(NC_AN[i,0]-1)
        index2=int(NC_AN[i,2]-1)
             
        start_x = x1[index1] 
        end_x = x2[index2]
        start_y = y1[index1]
        end_y = y2[index2]
        start_z = z1[index1]
        end_z = z2[index2]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        delta_z = end_z - start_z
        len_V2 =np.sqrt(delta_x*delta_x+delta_y*delta_y+delta_z*delta_z)
        V2[i,2]=delta_z/len_V2
        V2[i,0]=delta_x/len_V2
        V2[i,1]=delta_y/len_V2
                                
    return V2   

def find_V3(u,v):  
    V3=np.cross(u,v)
           
    return V3

def assemble_k_to_K(NC,DOF,V1,V2,V3,L_matrix,K):    # V for calculating T matrix, L_matrix for calculating k_prime
   
    T=np.zeros((12,12),dtype=float)
    for t in range(0,6,1):
        
        bn_id=int(NC[t,0]-1)
        en_id=int(NC[t,1]-1)
        
        V = np.array((V1[t],V2[t] ,V3[t] ))    
        for i in range(0,10,3):
            counter_x=0
            for k in range(0,3,1):
                counter_y=0
                for j in range(i,i+3,1):
                    
                    row_index=i+k
                    column_index=j
                    T[row_index,column_index]=V[counter_x,counter_y]
                    counter_y=counter_y+1
                counter_x=counter_x+1   
       
        T_transpose=np.transpose(T)
        
        L=L_matrix[t][0]
        
        k_prime=[[((E*area)/L),0,0,0,0,0,(((-1)*E*area)/L),0,0,0,0,0],
                        [0,((12*E*Iz)/(L**3)),0,0,0,((6*E*Iz)/(L**2)),0,(((-1)*12*E*Iz)/(L**3)),0,0,0,((6*E*Iz)/(L**2))],
                        [0,0,((12*E*Iy)/(L**3)),0,(((-1)*6*E*Iy)/(L**2)),0,0,0,(((-1)*12*E*Iy)/(L**3)),0,(((-1)*6*E*Iy)/(L**2)),0],
                        [0,0,0,((G*It)/L),0,0,0,0,0,(((-1)*G*It)/L),0,0],
                        [0,0,(((-1)*6*E*Iy)/(L**2)),0,((4*E*Iy)/L),0,0,0,((6*E*Iy)/(L**2)),0,((2*E*Iy)/L),0],
                        [0,((6*E*Iz)/(L**2)),0,0,0,((4*E*Iz)/L),0,(((-1)*6*E*Iz)/(L**2)),0,0,0,((2*E*Iz)/L)],
                        [(((-1)*E*area)/L),0,0,0,0,0,((E*area)/L),0,0,0,0,0],
                        [0,(((-1)*12*E*Iz)/(L**3)),0,0,0,(((-1)*6*E*Iz)/(L**2)),0,((12*E*Iz)/(L**3)),0,0,0,(((-1)*6*E*Iz)/(L**2))],
                        [0,0,(((-1)*12*E*Iy)/(L**3)),0,((6*E*Iy)/(L**2)),0,0,0,((12*E*Iy)/(L**3)),0,((6*E*Iy)/(L**2)),0],
                        [0,0,0,(((-1)*G*It)/L),0,0,0,0,0,((G*It)/L),0,0],
                        [0,0,(((-1)*6*E*Iy)/(L**2)),0,((2*E*Iy)/L),0,0,0,((6*E*Iy)/(L**2)),0,((4*E*Iy)/L),0],
                        [0,((6*E*Iz)/(L**2)),0,0,0,((2*E*Iz)/L),0,(((-1)*6*E*Iz)/(L**2)),0,0,0,((4*E*Iz)/L)]]

        #k=np.dot(T_transpose,np.dot(k_prime,T))   # CALCULATING k,
        k_mul=np.dot(T_transpose,k_prime)
        k=np.dot(k_mul,T)
        size_array_k=k.shape
        row_k=size_array_k[0]
        col_k=size_array_k[1]
      
        Ce=[DOF[bn_id,0],DOF[bn_id,1],DOF[bn_id,2],DOF[bn_id,3],DOF[bn_id,4],DOF[bn_id,5] ,DOF[en_id,0] ,DOF[en_id,1],DOF[en_id,2],DOF[en_id,3],DOF[en_id,4],DOF[en_id,5]]  
        # Ce for find k values that will be located in K matrix
        print(Ce)
        for i in range(0,row_k,1): #i for calculate row of k
            for j in range(0,col_k,1): #j for calculate row of k
                 K_index_1 = int(Ce[i]-1.0) 
                 K_index_2 = int(Ce[j]-1.0)
         
              
                 K_value = K[K_index_1,K_index_2] 
                 
                 # adding previous K value to new one recursively 
                 K[K_index_1][K_index_2] = K_value + k[i][j]          
                 
    print("K",K)
    return K
    
    
# Creating Matrices
XYZ=createMatrix(n,3,XYZ_values)

NC=createMatrix(m,2,NC_values)
BC=createMatrix(3,7, BC_values)
FEXT=createMatrix(1,7, FEXT_values)
AN=createMatrix(5,3,AN_values)
NC_AN=createMatrix(m,3,NC_AN_values)
print("NC_AN",NC_AN)
print("AN",AN)
print("NC:",NC)
# Calculating row and column of matrices
size_array_XYZ=XYZ.shape
row_XYZ=size_array_XYZ[0]
col_XYZ=size_array_XYZ[1]

size_array_NC=NC.shape
row_NC=size_array_NC[0]
col_NC=size_array_NC[1]

size_array_BC=BC.shape
row_BC=size_array_BC[0]
col_BC=size_array_BC[1]

size_array_FEXT=FEXT.shape
row_FEXT=size_array_FEXT[0]
col_FEXT=size_array_FEXT[1]

size_array_AN=AN.shape
row_AN=size_array_AN[0]
col_AN=size_array_AN[1]

size_array_NC_AN=NC_AN.shape
row_NC_AN=size_array_NC_AN[0]
col_NC_AN=size_array_NC_AN[1]




#--------------------------------- DOF creation ---------------------------------
DOF_values=np.zeros(n*6, dtype=float)   # dof_values is empty array which size is 12


DOF=createMatrix(n,6,DOF_values)      # Creating DOF matrix which is size (nx6)


size_array_BC=BC.shape
row_BC=size_array_BC[0]
col_BC=size_array_BC[1]


k=1  # defining k as counter
displacement_counter = 0

for i in range(0,n,1):              # loop over nodes
    BC_state="not_exist" 
    for j in range(0,row_BC,1):     # loop over rows of BC array
        if BC[j,0]==i+1:            # Check whether there exists BC on the node
            BC_state="exist" 
            x_value=j               # defining x_value as in which row BC on the node exists          
           
        
    
    if BC_state=="not_exist":       # if there is no BC on the node
        DOF[i,0]=k
        DOF[i,1]=k+1
        DOF[i,2]=k+2
        DOF[i,3]=k+3
        DOF[i,4]=k+4
        DOF[i,5]=k+5
        
        k=k+6
        displacement_counter = displacement_counter +6 
    
    elif  BC_state=="exist" and BC[x_value,1]==0:     # elif there is BC on the node AND X dir. is NOT constrained
        DOF[i,0]=k 
        k=k+1 
        displacement_counter = displacement_counter +1 
    elif  BC_state == "exist" and BC[x_value,2]==0:   # elif there is BC on the node AND Y dir. is NOT constrained
        DOF[i,1]=k 
        k=k+1 
        displacement_counter = displacement_counter +1 
    elif  BC_state=="exist" and BC[x_value,3]==0:     # elif there is BC on the node AND Z dir. is NOT constrained
        DOF[i,2]=k 
        k=k+1 
        displacement_counter = displacement_counter +1
    elif  BC_state=="exist" and BC[x_value,4]==0:     # elif there is BC on the node AND Qx . is NOT constrained
        DOF[i,3]=k 
        k=k+1 
        displacement_counter = displacement_counter +1
    elif  BC_state=="exist" and BC[x_value,5]==0:     # elif there is BC on the node AND Qy . is NOT constrained
        DOF[i,4]=k 
        k=k+1 
        displacement_counter = displacement_counter +1
    elif  BC_state=="exist" and BC[x_value,6]==0:     # elif there is BC on the node AND Qz . is NOT constrained
        DOF[i,5]=k 
        k=k+1 
        displacement_counter = displacement_counter +1
        

# filling the remaining 0 entries of the DOF matrix

for i in range(0,n,1):         # loop over nodes
    for j in range(0,6,1):     # Loop over X,Y and Z directions
        if DOF[i][j]==0:
            DOF[i][j]=k
            k=k+1

size_array_DOF=DOF.shape
row_DOF=size_array_DOF[0]
col_DOF=size_array_DOF[1]

#--------------------------------- end of DOF creation ---------------------------------

# Finding length matrix
x=XYZ[:,0]
y=XYZ[:,1]
z=XYZ[:,2]
        
L=np.zeros((row_NC,1),dtype=float)

    
for i in range(0,row_NC,1):
    index1=int(NC[i,0]-1)
    index2=int(NC[i,1]-1)
        
    start_x=x[index1]
    end_x=x[index2]
    
    start_y=y[index1]
    end_y=y[index2]
        
    start_z=z[index1]
    end_z=z[index2]
        
    delta_x=end_x-start_x
    delta_y=end_y-start_y
    delta_z=end_z-start_z
    length =np.sqrt(delta_x*delta_x+delta_y*delta_y+delta_z*delta_z)
    L[i]=length
    


q=row_DOF*col_DOF


K_values=np.zeros((q*q), dtype=float)

K=createMatrix(q,q,K_values)

V1=find_V1(XYZ)
V2=find_V2(NC_AN,XYZ,AN)
V3=find_V3(V1,V2)
K=assemble_k_to_K(NC,DOF,V1,V2,V3,L,K)

print("V1",V1)
print("V2",V2)
print("V3",V3)

# calculating displacement matrix
def calculate_Df(Kff,Fext,Kfp,Dp):
   
    division_matrix = np.subtract(Fext,(Kfp.dot(Dp)))
    inv_Kff=lin.inv(Kff)
    matrix = np.dot(inv_Kff,division_matrix)
    print("Kfp=")
    print(Kfp)
    return matrix

#calculating reaction matrix
def calculate_R(Kpf,Df,Kpp,Dp):
    matrix = np.dot(Kpf,Df) + np.dot(Kpp,Dp)
    return matrix



# creating all zero Q_values array with q elements
Q_values=np.empty([q], dtype=float) 

for i in range(0,q,1):
    Q_values[i] = 0

Q=createMatrix(q,1,Q_values)

size_array_FEXT=FEXT.shape
row_FEXT=size_array_FEXT[0]
col_FEXT=size_array_FEXT[1]


for i in range(0,row_FEXT,1):
    joint_id = int(FEXT[i,0]-1)  # Joint number on which an external force is applied
    Cn = DOF[joint_id,:] # dof numbers of the node
    first_index = int(Cn[0]-1) 
    second_index = int(Cn[1]-1)
    third_index = int(Cn[2]-1)
    fourth_index=int(Cn[3]-1)
    fifth_index=int(Cn[4]-1)
    sixth_index=int(Cn[5]-1)
    Q[first_index] = Q[first_index] + FEXT[i,1]    # assembly of Fx to Q
    Q[second_index] = Q[second_index] + FEXT[i,2]  # assembly of Fy to Q
    Q[third_index] = Q[third_index] + FEXT[i,3]    # assembly of Fz to Q
    Q[fourth_index] = Q[fourth_index]+FEXT[i,4]    # assembly of Mx to Q
    Q[fifth_index] = Q[fifth_index]+FEXT[i,5]      # assembly of My to Q
    Q[sixth_index] = Q[sixth_index]+FEXT[i,6]      # assembly of Mz to Q
    
rf = displacement_counter   # rf: number of rows of Kff,displacement counter has defined in line 221
Kff = K[0:rf,0:rf]    
Kfp = K[0:rf,rf:]
Kpf = K[rf:,0:rf]
Kpp = K[rf:,rf:]


size_array_Cn=Cn.shape

row_Cn=1
col_Cn=size_array_Cn[0]   


Fext = Q[0:rf]

# PRESCRIBED DISPLACEMENTS : ALL OF THEM ARE ZERO
Dp_values=np.empty([q-rf], dtype=float)

for i in range(0,q-rf,1): 
    Dp_values[i] = 0

Dp=createMatrix(q-rf,1,Dp_values) 
Df=calculate_Df(Kff,Fext,Kfp,Dp) 

print("Df=")
print (Df)


#print(NC)

len_Df=len(Df)
len_Dp = len(Dp)

D=np.empty([(len_Df+len_Dp),1], dtype=float)

counter_D_matrix=0
for i in range(0,len_Df,1):
    D[counter_D_matrix]=Df[i]
    counter_D_matrix = counter_D_matrix+1
    
for j in range(0,len_Dp,1):
    D[counter_D_matrix]=Dp[j]
    counter_D_matrix=counter_D_matrix+1

print("D=")
print (D)

R=calculate_R(Kpf,Df,Kpp,Dp)

print("R=")
print (R)
#print(DOF)

''' # 3D PLOT PART----------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

temp1=np.zeros(col_DOF,dtype=int)
temp2=np.zeros(col_DOF,dtype=int)

joint1=np.zeros(col_DOF,dtype=float)
joint2=np.zeros(col_DOF,dtype=float)

t=len(joint1)

counter_del=0

for i in range(0,row_NC,1):
    for x in np.arange(0,L[i]+0.1,0.1):
        counter_del=counter_del+1


del_x=np.zeros((counter_del,1),dtype=float)
del_y=np.zeros((counter_del,1),dtype=float)
del_z=np.zeros((counter_del,1),dtype=float)


        
counter_DEL=0
for i in range(0,row_NC,1):
    index1=(int)(NC[i][0]-1)
    index2=(int)(NC[i][1]-1)
    temp1=DOF[index1]
    temp2=DOF[index2]
    
    x_start=XYZ[i][0]
    x_end=XYZ[i+1][0]
    
    y_start=XYZ[i][1]
    y_end=XYZ[i][1]
    
    z_start=XYZ[i][2]
    z_end=XYZ[i][2]
    
    delta_x=x_end-x_start
    delta_y=y_end-y_start
    delta_z=z_end-z_start
    
    
    
    for j in range(0,col_DOF,1):
        joint1[j] = D[(int)(temp1[j])-1]
        joint2[j] = D[(int)(temp2[j])-1]
    
        
    v1y=joint1[1]
    c4y=v1y
    Q1y=joint1[4]
    c3y=Q1y        
    v2y=joint2[1]
    Q2y=joint2[4]
    Len=L[i]
     
    c1y=(12/(Len**3))*(((Len*Q1y)/2)+v1y-v2y-(Len/2)*Q2y)
    c2y=(1/Len)*(Q2y-Q1y-((c1y*(Len**2))/2))
    #-----------------------------------------------#
    v1z=joint1[2]
    c4z=v1z
    Q1z=joint1[5]
    c3z=Q1z        
    v2z=joint2[2]
    Q2z=joint2[5]
    
    c1z=(12/(Len**3))*(((Len*Q1z)/2)+v1z-v2z-(Len/2)*Q2z)
    c2z=(1/Len)*(Q2z-Q1z-((c1z*(Len**2))/2))
    print(c1y)
    x0=0
    y0=0
    z0=0    
    for x in np.arange(0,L[i]+0.1,0.1):
        if delta_x==0:
            x0=x_start
        else:
            x0=x0+x
        if delta_y==0:
            y0=y_start
        else:
            y0=y0+x
        if delta_z==0:
            z0=z_start
        else:
            z0=z0+x
            
            
        v=c1y*((x**3)/6)+c2y*((x**2)/2)+c3y*x+c4y # for y
        w=c1z*((x**3)/6)+c2z*((x**2)/2)+c3z*x+c4z # for z
            
        y_deformed=y0+v
        z_deformed=z0+w
        x_deformed=x0
        
        del_x[counter_DEL]=x_deformed-x0
        del_y[counter_DEL]=y_deformed-y0
        del_z[counter_DEL]=z_deformed-z0
        counter_DEL=counter_DEL+1

    plt.plot(del_z,del_y)
    plt.show()
        #ax =plt.axes(projection='3d')
        #ax.plot3D(del_x,del_y,del_z,'blue')
        #fig = plt.figure() '''
        




















        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
  
        
    
        
        
        
        
        
        








           
        
        
        
        
            
            
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
















































    
