import numpy as np 
import pandas as pd
from SONIA.sonia_length_pos import SoniaLengthPos
from SONIA.sonia_leftpos_rightpos import SoniaLeftposRightpos
from SONIA.evaluate_model import EvaluateModel
from SONIA.sonia_vjl import SoniaVJL
from scipy import stats
import matplotlib.pyplot as plt
import re
import keras.backend as K
import tensorflow as tf
import random as rn
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import os
from SONIA.evaluate_model import compute_all_pgens

# set seeds for reproducibility
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

print 'ANALYSE case 1M sequences'

df=pd.read_csv('vampire-emerson/2019-03-18-freq-1M-train/count/basic/test_data.csv',sep=',')
to_evalutate=list(df[['amino_acid','v_gene','j_gene']].values)

#define model
qm = SoniaLengthPos(load_dir='selection_models/emerson_frequency_lengthpos_1M',custom_pgen_model='universal_model')
qm0 =SoniaLeftposRightpos(load_dir='selection_models/emerson_frequency_leftright_1M',custom_pgen_model='universal_model')
qm1 = SoniaVJL(load_dir='selection_models/emerson_frequency_vjl_1M',custom_pgen_model='universal_model')

# load Evaluate model

main_folder='universal_model'
params_file_name = os.path.join(main_folder,'model_params.txt')
marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

genomic_data = olga_load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = olga_load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = generation_probability.GenerationProbabilityVDJ(generative_model, genomic_data)

ev=EvaluateModel(sonia_model=qm,custom_olga_model=pgen_model)
ev0=EvaluateModel(sonia_model=qm0,custom_olga_model=pgen_model)
ev1=EvaluateModel(sonia_model=qm1,custom_olga_model=pgen_model)

#evaluate ppost/pgen
energy,pgen,ppost=ev.evaluate_seqs(to_evalutate)
_,_,ppost_left=ev0.evaluate_seqs(to_evalutate)
_,_,ppost_vjl=ev1.evaluate_seqs(to_evalutate)


#rejct sequences with pgen=0. They have wrong V assignment (pseudogene thus not productive)
pgen=np.array(pgen)
sel=pgen!=0
r_value = stats.linregress(df.log_freq.values[sel],df.log_pvae.values[sel])[2]
print 'R^2 pvae',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(pgen)[sel]))[2]
print 'R^2 pgen',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost_vjl)[sel]))[2]
print 'R^2 olga.q',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost)[sel]))[2]
print 'R^2 sonia lengthpos',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost_left)[sel]))[2]
print 'R^2 sonia leftright',r_value**2

print 'DKL pvae',np.mean(df.log_freq.values[sel]-df.log_pvae.values[sel])/np.log(2)
print 'DKL pgen',np.mean(df.log_freq.values[sel]-np.log(np.array(pgen)[sel]))/np.log(2)
print 'DKL olga.q',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost_vjl)[sel]))/np.log(2)
print 'DKL psonia lengthpos',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost)[sel]))/np.log(2)
print 'DKL psonia leftright',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost_left)[sel]))/np.log(2)

print 'ANALYSE case 2e5 sequences'

df=pd.read_csv('vampire-emerson/2019-03-15-freq-in-ms/count/basic/test_data.csv',sep=',')
to_evalutate=list(df[['amino_acid','v_gene','j_gene']].values)

#define model
qm = SoniaLengthPos(load_dir='selection_models/emerson_frequency_lengthpos_0_2M',custom_pgen_model='universal_model')
qm0=SoniaLeftposRightpos(load_dir='selection_models/emerson_frequency_leftright_0_2M',custom_pgen_model='universal_model')
qm1 = SoniaVJL(load_dir='selection_models/emerson_frequency_vjl_0_2M',custom_pgen_model='universal_model')

ev=EvaluateModel(sonia_model=qm,custom_olga_model=pgen_model)
ev0=EvaluateModel(sonia_model=qm0,custom_olga_model=pgen_model)
ev1=EvaluateModel(sonia_model=qm1,custom_olga_model=pgen_model)

#evaluate ppost/pgen
energy,pgen,ppost=ev.evaluate_seqs(to_evalutate)
_,_,ppost_left=ev0.evaluate_seqs(to_evalutate)

energy_vjl,pgen_vjl,ppost_vjl=ev1.evaluate_seqs(to_evalutate)
sel=pgen!=0
#rejct sequences with pgen=0. They have wrong V assignment (pseudogene thus not productive)
pgen=np.array(pgen)
r_value = stats.linregress(df.log_freq.values[sel],df.log_pvae.values[sel])[2]
print 'R^2 pvae',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(pgen)[sel]))[2]
print 'R^2 pgen',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost_vjl)[sel]))[2]
print 'R^2 olga.q',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost)[sel]))[2]
print 'R^2 sonia lengthpos',r_value**2
r_value = stats.linregress(df.log_freq.values[sel],np.log(np.array(ppost_left)[sel]))[2]
print 'R^2 sonia leftright',r_value**2

print 'DKL pvae',np.mean(df.log_freq.values[sel]-df.log_pvae.values[sel])/np.log(2)
print 'DKL pgen',np.mean(df.log_freq.values[sel]-np.log(np.array(pgen)[sel]))/np.log(2)
print 'DKL olga.q',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost_vjl)[sel]))/np.log(2)
print 'DKL psonia lengthpos',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost)[sel]))/np.log(2)
print 'DKL psonia leftright',np.mean(df.log_freq.values[sel]-np.log(np.array(ppost_left)[sel]))/np.log(2)

#plot ppost
import matplotlib
def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    bins = [100, 500] # number of bins
    
    # histogram the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)
    hh=hh/hh.max()
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]
    map_reversed = matplotlib.cm.get_cmap('magma_r')
    s=ax.scatter(x2, y2, c=z2, cmap=map_reversed, marker='.',s=300,rasterized=True)
    cb=plt.colorbar(s,ax=ax)
    cb.ax.tick_params(labelsize=20)

    return ax

from matplotlib.lines import Line2D
fix,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8),dpi=200)
ax1.grid()
ax1.set_xlabel('$\log_{10}$ frequency',fontsize=30)
ax1.set_ylabel('$\log_{10}$ probability',fontsize=30)
ax1.set_ylim([-11,-3])
ax1.set_xlim([-7.5,-3.5])
ax1.locator_params(nbins=4)
ax1.plot([-20,0],[-20,0],c='k')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='SONIA',
                          markerfacecolor='w', markersize=0)]
ax1.legend(handles=legend_elements,fontsize=20,handletextpad=-2.0)
ax1.set_xticklabels([-8,-7,-6,-5,-4],fontsize=20)
ax1.set_yticklabels([-12,-10,-8,-6,-4],fontsize=20)

ax2.grid()
ax2.set_xlabel('$\log_{10}$ frequency',fontsize=30)
ax2.set_ylabel('$\log_{10}$ probability',fontsize=30)
ax2.set_ylim([-11,-3])
ax2.set_xlim([-7.5,-3.5])
ax2.locator_params(nbins=4)
ax2.plot([-20,0],[-20,0],c='k')
ax2.set_xticklabels([-8,-7,-6,-5,-4],fontsize=20)
ax2.set_yticklabels([-12,-10,-8,-6,-4],fontsize=20)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='VAE',
                          markerfacecolor='w', markersize=0)]
ax2.legend(handles=legend_elements,fontsize=20,handletextpad=-2.0)
density_scatter(df.log_freq.values[sel]/np.log(10),df.log_pvae.values[sel]/np.log(10),bins = [10,50],ax=ax2)
density_scatter(df.log_freq.values[sel]/np.log(10),np.log10(ppost_left)[sel],bins = [10,50],ax=ax1)
plt.savefig("frequency_0_2M.png")
