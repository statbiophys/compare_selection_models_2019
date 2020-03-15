import numpy as np 
import pandas as pd
from SONIA.sonia_length_pos import SoniaLengthPos
from SONIA.sonia_leftpos_rightpos import SoniaLeftposRightpos
from SONIA.evaluate_model import EvaluateModel
from SONIA.sonia_vjl import SoniaVJL

from scipy import stats
import matplotlib.pyplot as plt
import re
import tensorflow.keras.backend as K
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
 
#load 1M sequences
data=list(pd.read_csv('vampire-emerson/2019-03-18-freq-1M-train/count/basic/training-sequences.olga.tsv',
                      sep='\t',header=None).sample(frac=1).reset_index(drop=True).values)
gen=list(pd.read_csv('sampled_data/generated_sequences.csv').values.astype(np.str))
print len(gen),len(data)

#define and train model
qm = SoniaLengthPos(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)

#save model
qm.save_model('selection_models/emerson_frequency_lengthpos_1M')

qm = SoniaLeftposRightpos(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)

#save model
qm.save_model('selection_models/emerson_frequency_leftright_1M')

qm = SoniaVJL(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection()

qm.save_model('selection_models/emerson_frequency_vjl_1M',attributes_to_save = ['model', 'data_seqs', 'gen_seqs'])

data=list(pd.read_csv('vampire-emerson/2019-03-15-freq-in-ms/count/basic/training-sequences.olga.tsv',
                      sep='\t',header=None).sample(frac=1).reset_index(drop=True).values)

#define and train model
qm = SoniaLengthPos(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)

#save model
qm.save_model('selection_models/emerson_frequency_lengthpos_0_2M')

#define and train model
qm = SoniaLeftposRightpos(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection(epochs=50,batch_size=int(1e4),validation_split=0.01)

#save model
qm.save_model('selection_models/emerson_frequency_leftright_0_2M')

qm = SoniaVJL(data_seqs=data,gen_seqs=gen,custom_pgen_model='universal_model')
qm.infer_selection()

qm.save_model('selection_models/emerson_frequency_vjl_0_2M',attributes_to_save = ['model', 'data_seqs', 'gen_seqs'])
