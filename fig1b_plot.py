import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sonia_minimal.evaluate_model import EvaluateModel
from sonia_minimal.sonia_length_pos import SoniaLengthPos
from scipy import stats
import matplotlib.pyplot as plt
import re
import keras.backend as K
import tensorflow as tf
import random as rn

# set seeds for reproducibility
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def rewrite_gene(seq):
    # map from adaptive naming to olga naming
    x=seq.split('-')
    return 'TRB'+seq[4]+str(int(x[0][-2:]))+'-'+str(int(x[1]))

#load model
qm = SoniaLengthPos(load_model='sampled_data/deneuter_model',custom_pgen_model='universal_model')
ev=EvaluateModel(sonia_model=qm,olga_model='universal_model')

# we processed then the data to compute pvae using olga2adaptive and adaptive2olga 
# (104 seqs out of 2e4 rejected by the olga2adaptive function)
_,_,ppost_sonia=ev.evaluate_seqs(list(pd.read_csv('sampled_data/generated_sonia_trimmed.csv').values.astype(np.str)[:int(1e4)]))

#evalute ppost generated sequences from models
data='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/basic/vae-generated.csv'
seqs=[[x[0],rewrite_gene(x[1]),rewrite_gene(x[2])] for x in pd.read_csv(data).values]
_,_,ppost_basic=ev.evaluate_seqs(seqs)
data='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/count_match/vae-generated.csv'
seqs=[[x[0],rewrite_gene(x[1]),rewrite_gene(x[2])] for x in pd.read_csv(data).values]
_,_,ppost_countmatch=ev.evaluate_seqs(seqs)
data='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/olga-generated.csv'
seqs=[[x[0],rewrite_gene(x[1]),rewrite_gene(x[2])] for x in pd.read_csv(data).values]
_,_,ppost_olgaQ=ev.evaluate_seqs(seqs)

#evalute ppost data
folder='input/_output_deneuter-2019-02-07/'
datasets=['H11_B0','H18_B0','H20_B0','H21_B0','H22_B0','H23_B0_a','H23_B0_b','H26_B0',
         'H3_B0','H41_B0','H7_B0','H8_B0_b']
data_name=[d+'.head.csv' for d in datasets]
pposts=[]
for i in range(len(datasets)):
    print i,
    seqs=[[x[0],rewrite_gene(x[1]),rewrite_gene(x[2])] for x in pd.read_csv(folder+datasets[i]+'/'+data_name[i]).values]
    pposts.append(ev.evaluate_seqs(seqs)[2])
    
    
plt.figure(figsize=(8,8),dpi=300)
n_bins=35
binning_=np.linspace(-50,-10,n_bins)
binning_2=np.linspace(-50,-10,50)
plt.locator_params(axis='x',nbins=5)
plt.locator_params(axis='y',nbins=4)
k,l=np.histogram(np.nan_to_num(np.log(pposts[0])),binning_,density=True)
plt.plot(l[:-1],k,label='data',c='k',alpha=0.3)
for i in range(len(datasets)-1):
    k,l=np.histogram(np.nan_to_num(np.log(pposts[i+1])),binning_,density=True)
    plt.plot(l[:-1],k,c='k',alpha=0.3,linewidth=2)
k,l=np.histogram(np.nan_to_num(np.log(ppost_sonia)),binning_2,density=True)
plt.plot(l[:-1],k,label='SONIA',linewidth=3)
k,l=np.histogram(np.nan_to_num(np.log(ppost_basic)),binning_,density=True)
plt.plot(l[:-1],k,label='basic',linewidth=3)
k,l=np.histogram(np.nan_to_num(np.log(ppost_countmatch)),binning_,density=True)
plt.plot(l[:-1],k,label='count_match',linewidth=3)
k,l=np.histogram(np.nan_to_num(np.log(ppost_olgaQ)),binning_,density=True)
plt.plot(l[:-1],k,label='OLGA.Q',linewidth=3)
plt.xlabel('ln $ P_{post}^{SONIA}$',fontsize=30)
plt.ylabel('frequency',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.grid()
plt.savefig("ppost.svg")
plt.show()
