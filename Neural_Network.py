'''
Created on Feb 27, 2017

@author: opensam
'''
## Hats off to https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ for a great explanation
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def net_and_output(inp,weights,net,out,units):
    for h in range(0,units):
        n = np.dot(inp,weights[h,].T)
        net.append(n)
        out.append(sigmoid(n))
        #print(sigmoid(n))
        
def get_error(outo,num_out):
    err = 0
    for i in range(0,num_out):
        err = err + (np.power(outo[i]-o[i],2))
        
    return err/2


i = np.array([.05,.10,1])
o = np.array([.01,.99])
eta = 0.5


wh = np.array([[.15,.20,.35],[.25,.30,.35]])
wo = np.array([[.40,.45,.60],[.50,.55,.60]])
for t in range(0,10000):
    # Copy it. It would be required later 
    temp_wo=np.copy(wo)
    num_hidden_units=len(wh)
    num_out=len(i)-1
    erro_wrt_out=[]
########## Forward Pass #########
    neth = []
    outh = []
    units = num_hidden_units
    net_and_output(i,wh,neth,outh,units)
    outh.append(1)
    #print(outh)

    neto = []
    outo = []
    units = num_out
    net_and_output(outh,wo,neto,outo,units)
    #print(outo)

###### Forward Pass finished #######
    error = get_error(outo,num_out)
    print(error)
###### Backward Pass begins #######

###### For output layer weights ######
    for j in range(0,num_out):
        eo = -(o[j] - outo[j])
        on = outo[j]*(1-outo[j])
        for l in range(0,num_hidden_units):
            nw = outh[l]    
            wo[j,l] = wo[j,l] - eta*eo*on*nw
   
## For hidden layer ##
    err_sum=0
    for j in range(0,num_out):
        prev = (outh[j]*(1-outh[j]))
        eo = 0
        for k in range(0,num_out):
            eo = eo + (-(o[k]-outo[k]))*(outo[k]*(1-outo[k]))*temp_wo[k,j]
        for l in range(0,num_hidden_units):
            wh[j,l] = wh[j,l] - eta*eo*prev*i[j]
    

print(outo[0])
print(outo[1])
    

    