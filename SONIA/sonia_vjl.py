#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:12:15 2019

@author: administrator
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sonia import Sonia

class SoniaVJL(Sonia):
    
    def __init__(self, data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',
                 load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, L1_hist_file = None, load_seqs = True,
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -5, max_energy_clip = 10, seed = None,custom_pgen_model=None):

        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed)
        self.max_depth = max_depth
        self.max_L = max_L
        self.min_L = 4

        if any([x is not None for x in [load_dir, feature_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, L1_hist_file = L1_hist_file, load_seqs = load_seqs)
        else:
            self.add_features(include_indep_genes = include_indep_genes, include_joint_genes = include_joint_genes, custom_pgen_model = custom_pgen_model)
    
    def add_features(self, include_indep_genes = False, include_joint_genes = True, custom_pgen_model=None):
        """Generates a list of feature_lsts for a length dependent L pos model.
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence
        max_L : int
            Maximum length CDR3 sequence
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.
                
        """
        
        import olga.load_model as olga_load_model
        features = []

        if custom_pgen_model is None:
            main_folder = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type)
        else:
            main_folder = custom_pgen_model

        params_file_name = os.path.join(main_folder,'model_params.txt')
        V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

        genomic_data = olga_load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

        features += [[v, j, 'l'+str(l)] for v in set(['v' + genV[0].split('*')[0].split('V')[-1] for genV in genomic_data.genV]) for j in set(['j' + genJ[0].split('*')[0].split('J')[-1] for genJ in genomic_data.genJ]) for l in range(self.min_L, self.max_L + 1)]
            
        self.update_model(add_features=features)
        
    def find_seq_features(self, seq, features = None):
        """Finds which features match seq
        
        If no features are provided, the length dependent amino acid model
        features will be assumed.
        
        Parameters
        ----------
        seq : list
            CDR3 sequence and any associated genes
        features : ndarray
            Array of feature lists. Each list contains individual subfeatures which
            all must be satisfied.
    
        Returns
        -------
        seq_features : list
            Indices of features seq projects onto.
        
        """
        seq_feature_lsts=[]
        if features is None:
            v_genes = [gene for gene in seq[1:] if 'v' in gene.lower()]
            j_genes = [gene for gene in seq[1:] if 'j' in gene.lower()]
            #Allow for just the gene family match
            v_genes += [gene.split('-')[0] for gene in seq[1:] if 'v' in gene.lower()]
            j_genes += [gene.split('-')[0] for gene in seq[1:] if 'j' in gene.lower()]
            try:
                seq_feature_lsts += [['v' + '-'.join([str(int(y)) for y in v_gene.lower().split('v')[-1].split('-')]), 'j' + '-'.join([str(int(y)) for y in j_gene.lower().split('j')[-1].split('-')]),'l' + str(len(seq[0]))] for v_gene in v_genes for j_gene in j_genes]
            except ValueError:
                pass
            seq_features = list(set([self.feature_dict[tuple(f)] for f in seq_feature_lsts if tuple(f) in self.feature_dict]))
        else:
            seq_features = []
            for feature_index,feature_lst in enumerate(features):
                if self.seq_feature_proj(feature_lst, seq):
                    seq_features += [feature_index]
        return seq_features
    
    def _encode_data(self,seq_features):
        #seq_features=[[feature] for feature in seq_features]
        #print seq_features[0]
        length_input=len(self.features)
        data=np.array(seq_features)
        data_enc = np.zeros((len(data), length_input), dtype=np.int8)
        for i in range(len(data_enc)): data_enc[i][data[i]] = 1
        return data_enc
    
    def infer_selection(self, epochs = 10, batch_size=5000, initialize = True, seed = None):
        """Infer model parameters, i.e. energies for each model feature.

        Parameters
        ----------
        epochs : int
            Maximum number of learning epochs
        intialize : bool
            Resets data shuffle
        batch_size : int
            Size of the batches in the inference
        seed : int
            Sets random seed

        Attributes set
        --------------
        model : keras model
            Parameters of the model
        model_marginals : array
            Marginals over the generated sequences, reweighted by the model.
        L1_converge_history : list
            L1 distance between data_marginals and model_marginals at each
            iteration.

        """

        if seed is not None:
            np.random.seed(seed = seed)
        Q=np.clip(np.nan_to_num(self.data_marginals/self.gen_marginals),0.,1000.)
        Q[Q==0.0]=0.0000000000001
        self.model.set_weights([np.array([[-np.log(i)] for i in Q])])
        self.update_model(auto_update_marginals=True)
