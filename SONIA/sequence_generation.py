import numpy as np
import os
import olga.load_model as olga_load_model
import olga.sequence_generation as seq_gen

class SequenceGeneration(object):

	"""Class used to evaluate sequences with the sonia model

	Attributes
	----------
	sonia_model: object 
		Required. Sonia model: only object accepted.

	custom_olga_model: object 
		Optional: already loaded custom olga sequence_generation object

	custom_genomic_data: object
		Optional: already loaded custom olga genomic_data object

	Methods
	----------

	generate_sequences_pre(num_seqs = 1)
		Generate sequences using olga
	
	generate_sequences_post(num_seqs,upper_bound=10)
		Generate sequences using olga and perform rejection selection.
	
	rejection_sampling(upper_bound=10,energies=None)
		Returns acceptance from rejection sampling of a list of energies.
		By default uses the generated sequences within the sonia model.

	"""

	def __init__(self,sonia_model=None, custom_olga_model=None, custom_genomic_data=None):

		if type(sonia_model)==str or sonia_model is None: 
			print('ERROR: you need to pass a Sonia object')
			return

		self.sonia_model=sonia_model # sonia model passed as an argument

		# define olga sequence_generation model
		if custom_olga_model is not None:
			if custom_genomic_data is None:
				print('ERROR: you need to pass also the custom_genomic_data')
				return
			self.genomic_data = custom_genomic_data
			self.seq_gen_model = custom_olga_model
		else:

			main_folder=os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.sonia_model.chain_type)

			params_file_name = os.path.join(main_folder,'model_params.txt')
			marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
			V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
			J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

			if self.sonia_model.chain_type!='human_T_alpha':
				self.genomic_data = olga_load_model.GenomicDataVDJ()
				self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
				generative_model = olga_load_model.GenerativeModelVDJ()
				generative_model.load_and_process_igor_model(marginals_file_name)
				self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, self.genomic_data)

			else:
				self.genomic_data = olga_load_model.GenomicDataVJ()
				self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
				generative_model = olga_load_model.GenerativeModelVJ()
				generative_model.load_and_process_igor_model(marginals_file_name)   

				self.seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, self.genomic_data)

		# you need Z for rejection selection and generate sequences ppost --> compute only once
		self.energies_gen=self.sonia_model.compute_energy(self.sonia_model.gen_seq_features)
		self.Z=np.sum(np.exp(-self.energies_gen))/len(self.energies_gen) 

	def generate_sequences_pre(self, num_seqs = 1):
		"""Generates MonteCarlo sequences for gen_seqs using OLGA.

		Only generates seqs from a V(D)J model. Requires the OLGA package
		(pip install olga).

		Parameters
		----------
		num_seqs : int or float
			Number of MonteCarlo sequences to generate and add to the specified
			sequence pool.
		
		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model

		"""
		
		#Generate sequences
		seqs_generated=[self.seq_gen_model.gen_rnd_prod_CDR3() for i in range(int(num_seqs))]
		seqs = [[seq[1], self.genomic_data.genV[seq[2]][0].split('*')[0], self.genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in seqs_generated]
		return seqs
	
	def generate_sequences_post(self,num_seqs,upper_bound=10):
		"""Generates MonteCarlo sequences from Sonia through rejection sampling.

		Parameters
		----------
		num_seqs : int or float
			Number of MonteCarlo sequences to generate and add to the specified
			sequence pool.
		upper_bound: int
			accept all above the threshold. Relates to the percentage of 
			sequences that pass selection.

		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model that pass selection.

		"""
		seqs_post=[['a','b','c']] # initialize

		while len(seqs_post)<num_seqs:

			# generate sequences from pre
			seqs=self.generate_sequences_pre(num_seqs = int(1.1*upper_bound*num_seqs))

			# compute features and energies 
			seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]
			energies = self.sonia_model.compute_energy(seq_features)

			#do rejection
			rejection_selection=self.rejection_sampling(upper_bound=upper_bound,energies=energies)
			seqs_post=np.concatenate([seqs_post,np.array(seqs)[rejection_selection]])

		return seqs_post[1:num_seqs+1]

	def rejection_sampling(self,upper_bound=10,energies=None):

		''' Returns acceptance from rejection sampling of a list of seqs.
		By default uses the generated sequences within the sonia model.
		
		Parameters
		----------
		upper_bound : int or float
			accept all above the threshold. Relates to the percentage of 
			sequences that pass selection

		Returns
		-------
		rejection selection: array of bool
			acceptance of each sequence.
		
		'''

		if energies is None:  energies=self.energies_gen
		Q=np.exp(-energies)
		random_samples=np.random.uniform(size=len(energies)) # sample from uniform distribution

		return random_samples < np.clip(Q/self.Z,0,upper_bound)/float(upper_bound)