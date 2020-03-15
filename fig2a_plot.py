import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
from SONIA.sonia_length_pos import SoniaLengthPos
from SONIA.sonia_vjl import SoniaVJL
from SONIA.sonia_leftpos_rightpos import SoniaLeftposRightpos
from SONIA.evaluate_model import EvaluateModel

from scipy import stats
import matplotlib.pyplot as plt
import re
import keras.backend as K
import tensorflow as tf
import random as rn
import os
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import process_adaptive as pa
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
#qm = SoniaLengthPos(load_dir='selection_models/deneuter_model_lengthpos',custom_pgen_model='deneuter_model')
qmlr = SoniaLeftposRightpos(load_dir='selection_models/deneuter_model_leftright',custom_pgen_model='deneuter_model')
#qmvjl = SoniaVJL(load_dir='selection_models/deneuter_model_VJL',custom_pgen_model='deneuter_model')

# gen model
main_folder='deneuter_model'
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

genomic_data = olga_load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = olga_load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = generation_probability.GenerationProbabilityVDJ(generative_model, genomic_data)

ev=EvaluateModel(sonia_model=qmlr,custom_olga_model=pgen_model)

# we processed then the data to compute pvae using olga2adaptive and adaptive2olga 
# (2 seqs out of 2e4 rejected by the olga2adaptive function)
#sonia_seqs=pd.read_csv('sampled_data/generated_sonia_lengthpos.csv')
#_,_,ppost_sonia=ev.evaluate_seqs(list(sonia_seqs.values))
sonia_seqs=pd.read_csv('sampled_data/generated_sonia_leftright.csv')
_,_,ppost_sonialr=ev.evaluate_seqs(list(sonia_seqs.values))
#sonia_seqs=pd.read_csv('sampled_data/generated_sonia_VJL.csv')
#_,_,ppost_olgaQ=ev.evaluate_seqs(list(sonia_seqs.values))

#evalute ppost generated sequences from models
data='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/basic/vae-generated.csv'
seqs=pa.adaptive2olga(pd.read_csv(data))
_,_,ppost_basic=ev.evaluate_seqs(list(seqs.values))
data='input/_output_deneuter-2019-02-07/deneuter-2019-02-07.train/0.75/count_match/vae-generated.csv'
seqs=pa.adaptive2olga(pd.read_csv(data))
_,_,ppost_countmatch=ev.evaluate_seqs(list(seqs.values))

#evalute ppost data
folder='input/_output_deneuter-2019-02-07/'
datasets=['H11_B0','H18_B0','H20_B0','H21_B0','H22_B0','H23_B0_a','H23_B0_b','H26_B0',
         'H3_B0','H41_B0','H7_B0','H8_B0_b']
data_name=[d+'.head.csv' for d in datasets]
pposts=[]
for i in range(len(datasets)):
    print i,
    seqs=list(pa.adaptive2olga(pd.read_csv(folder+datasets[i]+'/'+data_name[i])).values)
    pposts.append(ev.evaluate_seqs(seqs)[2])
    
    
plt.figure(figsize=(8,8),dpi=300)
n_bins=35
binning_=np.linspace(-20,-5,n_bins)
plt.locator_params(axis='x',nbins=5)
plt.locator_params(axis='y',nbins=4)
k,l=np.histogram(np.nan_to_num(np.log(pposts[0])/np.log(10)),binning_,density=True)
plt.plot(l[:-1],k,label='data',c='k',alpha=0.3)
for i in range(len(datasets)-1):
    k,l=np.histogram(np.nan_to_num(np.log(pposts[i+1])/np.log(10)),binning_,density=True)
    plt.plot(l[:-1],k,c='k',alpha=0.3,linewidth=2)
#k,l=np.histogram(np.nan_to_num(np.log(ppost_sonia)/np.log(10)),binning_,density=True)
#plt.plot(l[:-1],k,label='SONIA Lenght-Position',linewidth=3)
k,l=np.histogram(np.nan_to_num(np.log(ppost_sonialr)/np.log(10)),binning_,density=True)
plt.plot(l[:-1],k,label='SONIA Left+Right',linewidth=3)
k,l=np.histogram(np.nan_to_num(np.log(ppost_basic)/np.log(10)),binning_,density=True)
plt.plot(l[:-1],k,label='VAE basic',linewidth=3)
#k,l=np.histogram(np.nan_to_num(np.log(ppost_countmatch)/np.log(10)),binning_,density=True)
#plt.plot(l[:-1],k,label='count_match',linewidth=3)
#k,l=np.histogram(np.nan_to_num(np.log(ppost_olgaQ)/np.log(10)),binning_,density=True)
#plt.plot(l[:-1],k,label='OLGA.Q',linewidth=3)
plt.xlabel('$\log_{10} P_{post}^{SONIA}$',fontsize=30)
plt.ylabel('frequency',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=22)
plt.grid()
plt.tight_layout()
plt.savefig("ppost.png")