# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

    
# Initialize weights
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def forward_ae(x,w1,w2):
    z = np.dot(w1,x)
    a1 = act_sigmoid(z)



    return(a)    

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# Calculate Pseudo-inverse
def pinv_ae(x,w1,C):
    z = np.dot(w1,x)
    a1 = act_sigmoid(z)
    I = np.identity(x.shape[0])

    w2 = np.dot(x, np.dot(a1.T, np.dot(a1, (a1.T + I/C) )**(-1) ))
    return (w2)

# STEP 2: Feed-Backward for AE
def gradW1(a,w2):   
    z = np.dot(w2,a)
    e = 1/ (2*len(a))                                           ## ?????

    delta2 = e * deriva_sigmoid(z2)
    gW1 = np.dot(w2.T, delta2) * np.dot(deriva_sigmoid(z), a.T) ## ?????
    return(gW1,Cost)        


# Update AE's weight via RMSprop
def updW1_rmsprop(w,v,gW,mu):
    eps = 10**-8
    b = 0.9

    for i in range(len(w)):
        v[i] = b*v[i] + (1-b)*(gW[i])**2
        gRMS_a = (mu/(np.sqrt(v[i] + eps)))
        gRMS = gRMS_a*gW[i]
        w[i] = w[i] - gRMS
    
    return(w,v)

# Update Softmax's weight via RMSprop
def updW_sft_rmsprop(w,v,gw,mu):
    eps = 10**-10
    b = 0.9
    v = b*v + (1-b)*(gw)**2
    gRMS_a = (mu/(np.sqrt(v + eps)))
    gRMS = gRMS_a*gw
    w = w - gRMS
    return(w,v)


# Softmax's gradient
def gradW_softmax(x,y,a):
    ya = y*np.log(a)
    cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))
    return(gW,cost)

# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# Métrica
def metricas(x,y):
    confussion_matrix = confusion_matrix(x,y)
        
    f_score = []
    
    for index, caracteristica in enumerate(confussion_matrix):
        
        TP = caracteristica[index]
        FP = confussion_matrix.sum(axis=0)[index] - TP
        FN = confussion_matrix.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append(2 * (precision * recall) / (precision + recall))
        
    metrics = pd.DataFrame(f_score)
    metrics.to_csv("metrica_dl.csv", index=False, header=False)
    return(f_score)
    
#Confusion matrix
def confusion_matrix(y,z):
    cm = np.zeros((z.shape[0], y.shape[0]))
    
    for real, predicted in zip(z.T, y.T):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
          
    return(cm)

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt('cnf_sae.csv',delimiter=',')    
    par_sae=[]
    par_sae.append(np.int16(par[0])) # Batch size
    par_sae.append(np.int16(par[1])) # MaxIter
    par_sae.append(np.float(par[2])) # Learn rate    
    par_sae.append(np.float(par[3])) # Penalities    
    
    for i in range(4,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt('cnf_softmax.csv',delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning     
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x = pd.read_csv(fname, header = None)
    x = np.array(x)  
    return(x)

# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):   
    np.savetxt('costo_softmax.csv', cost, delimiter=",")
    W.append(Ws)
    np.savez('w_dl.npz', W=W)
    

#load weight of the DL in numpy format
def load_w_dl():
    W = np.load('w_dl.npz', allow_pickle=True)
    return(W)   
    
