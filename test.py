import pandas as pd
import numpy as np
import my_utility as ut



# Beginning ...
def main():		
	x,y = ut.load_data_csv('test.csv')		
	W   = ut.load_w_dl()
	z   = ut.forward_dl(x,W)      		
	ut.metricas(y,z) 	
	

if __name__ == '__main__':   
	 main()

