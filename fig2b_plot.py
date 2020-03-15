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
    pvaes.append(pd.read_csv(folder+datasets[i]+'/test.pvae.csv').values[:,0]/np.log(10))    
#load pvae models
sonia_vae=pd.read_csv('sampled_data/generated_sonia_lengthpos_adaptive.pvae.csv').values[:,-1]/np.log(10)
sonia_vae_lr=pd.read_csv('sampled_data/generated_sonia_leftright_adaptive.pvae.csv').values[:,-1]/np.log(10)
olga=pd.read_csv('sampled_data/generated_sonia_VJL_adaptive.pvae.csv').values[:,-1]/np.log(10)
other_ppost=pd.read_csv(folder+'vae-generated.pvae.csv').values[:,-1]/np.log(10)
other_ppost_count_match=pd.read_csv('input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/count_match/vae-generated.pvae_basic.csv').values[:,-1]/np.log(10)

#plot pvae
plt.figure(figsize=(8,8),dpi=300)
n_bins=35
binning_=np.linspace(-20,-5,n_bins)
plt.locator_params(axis='x',nbins=5)

plt.locator_params(axis='y',nbins=4)
k,l=np.histogram(pvaes[0],binning_,density=True)
plt.plot(l[:-1],k,label='data',c='k',alpha=0.3)
for i in range(len(datasets)-1):
    k,l=np.histogram(pvaes[i+1],binning_,density=True)
    plt.plot(l[:-1],k,c='k',alpha=0.3,linewidth=2)

#k,l=np.histogram(sonia_vae,binning_,normed=True)
#plt.plot(l[:-1],k,label='SONIA Length-Position',alpha=1,linewidth=3) 
k,l=np.histogram(sonia_vae_lr,binning_,normed=True)
plt.plot(l[:-1],k,label='SONIA Left+Right',alpha=1,linewidth=3) 
k,l=np.histogram(other_ppost,binning_,density=True)
plt.plot(l[:-1],k,label='VAE basic',alpha=1,linewidth=3) 
#k,l=np.histogram(other_ppost_count_match,binning_,density=True)
#plt.plot(l[:-1],k,label='count_match',alpha=1,linewidth=3) 
#k,l=np.histogram(olga,binning_,density=True)
#plt.plot(l[:-1],k,label='OLGA.Q',alpha=1,linewidth=3) 
plt.xlabel(' $\log_{10} P_{VAE}$',fontsize=30)
plt.grid()
plt.ylabel('frequency',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("pvae.png")
