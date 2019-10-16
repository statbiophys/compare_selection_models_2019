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

# choose basic because of best performance
df=df.loc[df.model=='basic'] 

#process data
df['n_subjects']=[re.sub('count_in_', '', x) for x in df.column.values]
df['n_subjects']=np.array([re.sub('count', '666', x) for x in df.n_subjects.values]).astype(np.int)

# choose 666 because of best performance of vae
df=df.loc[df.n_subjects==666]
df=df.loc[df.split=='test']

# define test set
test_data=df[['amino_acid','v_gene','j_gene']].values
joined_test_data=["".join(seq) for seq in test_data]

#load sampled data
data=list(pd.read_csv('sampled_data/emerson_universal_sample.csv').values.astype(np.str))
gen=list(pd.read_csv('sampled_data/generated_sequences.csv').values.astype(np.str))

# select 250 000 sequences that are not present in test set (20% used as validation)
data=np.array([d for d in data if not "".join(d) in joined_test_data])[:int(25e4)]
print len(np.intersect1d(["".join(seq) for seq in data],joined_test_data)), 'tcrs in common with test set'

# rewrite to make it compatible with olga
data= [[d[0], rewrite_gene(d[1]),rewrite_gene(d[2])] for d in data] 

#define and train model
qm = SoniaLengthPos(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.reject_bad_features(5)
qm.infer_selection(max_iterations=25)

#save model
qm.save_model('sampled_data/emerson_frequency',attributes_to_save = ['model_params', 'L1_converge_history'])