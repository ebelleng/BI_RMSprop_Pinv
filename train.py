#Training SAE via RMSprop+Pseudo-inversa

import pandas     as pd
import numpy      as np
import my_utility as ut

#gets miniBatch
def get_miniBatch(i,x,bsize):
    z=x[:,i*bsize:(i+1)*bsize]
    return(z)

# Training miniBatch for softmax
def train_sft_batch(x,y,W,V,numBatch,BatchSize,mu):
    costo = []    
    for i in range(numBatch):   
        xe,ye = get_miniBatch(i,x,BatchSize)        
        a  = ut.forward_ae(xe,W[0], W[1])
        gW   = ut.gradW1(a,W)
        W,V  = ut.up(W,V,gW,mu);  #eReMeSPPROP      
    return(W,V,costo)

# Softmax's training via RMSprop
def train_softmax(x,y,par1,par2):
    W,V        = ut.ini_WV(y.shape[0],x.shape[0])    
    numBatch   = np.int16(np.floor(x.shape[1]/par2[0]))    
    Costo = []
    for Iter in range(1,par1[0]):                
        xe,ye = reordena_rand(x,y)         
        W,V,c = train_sft_batch(xe,ye,W,V,numBatch,par2[0],par1[1])        
        Costo.append(np.mean(c))         
    return(W,Costo)    
 
# AE's Training with miniBatch
def train_ae_batch(x,w1,v,w2,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))    
    cost= [] 
    for i in range(numBatch):                
        xe    = get_miniBatch(...)
        #complete code               
    return(w1,v,cost)

# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x,hn,param):        
    w1,v = ut.ini_WV(param[hn],x.shape[0])            
    w2   = ut.pinv_ae(x,w1,param[3])
    cost = []
    for Iter in range(1,param[1]):        
        xe     = x[:,np.random.permutation(x.shape[1])]                
        w1,v,c = train_ae_batch(xe,w1,v,w2,param)                      
        cost.append(np.mean(c))                
    return(w2.T) 

#SAE's Training 
def train_sae(x,param):
    W={}
    for hn in range(4,len(param)):
        w1       = train_ae(x,hn,param)        
        #complete code       
    return(W,x) 

# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()    
    x,y         = ut.load_data_csv('train.csv')    
    W,Xr        = train_sae(xe,p_sae)         
    Ws, cost    = train_softmax(Xr,ye,p_sft,p_sae)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

