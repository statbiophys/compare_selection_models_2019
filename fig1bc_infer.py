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
    
#load data
data=list(pd.read_csv('sampled_data/deneuter_data.csv').values.astype(np.str))
data=[[x[0],rewrite_gene(x[1]),rewrite_gene(x[2])] for x in data]
gen=list(pd.read_csv('sampled_data/generated_sequences.csv').values.astype(np.str))

#define and infer model
qm = SoniaLengthPos(data_seqs = data, gen_seqs=gen)
qm.reject_bad_features(5) 
qm.infer_selection(max_iterations=25)

#save model
qm.save_model('sampled_data/deneuter_model',attributes_to_save = ['model_params', 'L1_converge_history','gen_seqs'])

# generate sequences to evaluate
ev=EvaluateModel(sonia_model=qm,olga_model='universal_model')
seqs_sonia=ev.generate_sequences_post(int(2e4))
pd.DataFrame(seqs_sonia,columns=['amino_acid','v_gene','j_gene']).to_csv('sampled_data/generated_sonia.csv',index=False)