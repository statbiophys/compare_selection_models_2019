import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import random as rn

#load pvae data
folder='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/basic/'
datasets=['H11_B0','H18_B0','H20_B0','H21_B0','H22_B0','H23_B0_a','H23_B0_b','H26_B0',
         'H3_B0','H41_B0','H7_B0','H8_B0_b']
datasets=[d+'.head' for d in datasets]
pvaes=[]
for i in range(len(datasets)):
    pvaes.append(pd.read_csv(folder+datasets[i]+'/test.pvae.csv').values[:,0])
    pd.read_csv(folder+datasets[0]+'/test.pvae.csv').values[:,0]
    
#load pvae models
sonia_vae=pd.read_csv('sampled_data/generated_sonia_trimmed.pvae.csv').values[:,-1]
olga=pd.read_csv(folder+'olga-generated.pvae.csv').values[:,-1]
other_ppost=pd.read_csv(folder+'vae-generated.pvae.csv').values[:,-1]

#plot pvae
plt.figure(figsize=(8,8),dpi=300)
n_bins=35
binning_=np.linspace(-50,-10,n_bins)
binning_2=np.linspace(-50,-10,50)
plt.locator_params(axis='x',nbins=5)

plt.locator_params(axis='y',nbins=4)
k,l=np.histogram(pvaes[0],binning_,density=True)
plt.plot(l[:-1],k,label='data',c='k',alpha=0.3)
for i in range(len(datasets)-1):
    k,l=np.histogram(pvaes[i+1],binning_,density=True)
    plt.plot(l[:-1],k,c='k',alpha=0.3,linewidth=2)

k,l=np.histogram(sonia_vae,binning_2,normed=True)
plt.plot(l[:-1],k,label='SONIA',c='C0',alpha=1,linewidth=3) 

k,l=np.histogram(other_ppost,binning_,density=True)
plt.plot(l[:-1],k,label='basic',c='C1',alpha=1,linewidth=3) 
k,l=np.histogram(other_ppost,binning_,density=True)
plt.plot(l[:-1],k,alpha=0,linewidth=3)
k,l=np.histogram(olga,binning_,density=True)
plt.plot(l[:-1],k,label='OLGA.Q',c='C3',alpha=1,linewidth=3) 
plt.xlabel('ln $P_{VAE}$',fontsize=30)
plt.grid()
plt.ylabel('frequency',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.savefig("pvae.svg")
