import numpy as np 
import pandas as pd
from sonia_minimal.sonia_length_pos import SoniaLengthPos
from sonia_minimal.evaluate_model import EvaluateModel
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
    
# load data from vampire-analysis-1
path = 'input/_output_pipe_freq/merged.agg.csv.bz2'
df=pd.read_csv(path)

# load data from vampire-analysis-1
path = 'input/_output_pipe_freq/merged.agg.csv.bz2'
df=pd.read_csv(path)

# choose basic because of best performance
df=df.loc[df.model=='basic'] 

# choose 666 because of their best performance
df['n_subjects']=[re.sub('count_in_', '', x) for x in df.column.values]
df['n_subjects']=np.array([re.sub('count', '666', x) for x in df.n_subjects.values]).astype(np.int)
df=df.loc[df.n_subjects==666]
df=df.loc[df.split=='test']

#process data..
df['log_frequency']=np.log(df['frequency'])
df['Pgen']=df.Pgen+np.min(df.Pgen)/2.
df['Ppost']=df.Ppost+np.min(df.Ppost)/2.
df['Pvae']=np.exp(df.log_Pvae)
df['log_Pgen']=np.log(df.Pgen)
df['log_Ppost']=np.log(df.Ppost)
df['normed_Pvae']=df.Pvae/np.sum(df.Pvae)
df['normed_Pgen']=df.Pgen/np.sum(df.Pgen)
df['normed_Ppost']=df.Ppost/np.sum(df.Ppost)
df['log_normed_Pvae']=np.log(df.Pvae/np.sum(df.Pvae))
df['log_normed_Pgen']=np.log(df.Pgen/np.sum(df.Pgen))
df['log_normed_Ppost']=np.log(df.Ppost/np.sum(df.Ppost))

# define sequences to evaluate
test_data=df[['amino_acid','v_gene','j_gene']].values
joined_test_data=["".join(seq) for seq in test_data]

#define model
qm = SoniaLengthPos(load_model='sampled_data/emerson_frequency',custom_pgen_model='universal_model')
qm.add_generated_seqs(int(1e5)) # for partition function estimation
ev=EvaluateModel(sonia_model=qm,olga_model='universal_model')

#evaluate ppost
energy,pgen,ppost=ev.evaluate_seqs([[i[0],rewrite_gene(i[1]),rewrite_gene(i[2])] for i in test_data])

#rejct sequences with pgen=0. They have wrong V assignment (pseudogene thus not productive)
print test_data[np.array(pgen)==0][:,1], 'are pseudogenes, we omit these 3 tcrs'
pgen=np.array(pgen)
selection=pgen!=0
log_Psonia=np.log(ppost[selection])
log_normed_Psonia= np.log(np.exp(log_Psonia)/np.sum(np.exp(log_Psonia)))

#compute correlation
r_value = stats.linregress(df.log_frequency.values[selection],df.log_normed_Ppost.values[selection])[2]
print 'R^2 OLGA.Q',r_value**2
r_value= stats.linregress(df.log_frequency.values[selection],df.log_normed_Pvae.values[selection])[2]
print 'R^2 VAE ',r_value**2
r_value = stats.linregress(df.log_frequency.values[selection],log_normed_Psonia)[2]
print 'R^2 SONIA ',r_value**2

#plot ppost
plt.figure(figsize=(8,8),dpi=100)
plt.scatter(df.log_frequency.values[selection],df.log_normed_Pvae.values[selection],alpha=0.1,c='C1',s=20,rasterized=True)
plt.scatter(df.log_frequency.values[selection],log_normed_Psonia,alpha=0.1,c='C0',s=20,rasterized=True)
plt.scatter([0,0],[0,0],c='C0',label='SONIA',rasterized=True)
plt.scatter([0,0],[0,0],c='C1',label='basic',rasterized=True)

plt.xlabel('ln frequency',fontsize=30)
plt.ylabel('ln normalized probability',fontsize=30)
plt.ylim([-20,-5])
plt.xlim([-16,-10])
plt.locator_params(nbins=4)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
plt.legend(fontsize=20)
plt.savefig("frequency.svg")