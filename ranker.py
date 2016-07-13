import os, pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from params import get_params
import random
import time

params     = get_params()
sleep_time = params['sleep_time']
disp_n     = params['disp_n']


class Ranker():

	def __init__(self,params):

		# Read image lists

		self.dataset    = params['dataset']
		self.image_path = params['database_images']
		self.dimension  = params['dimension']
		self.pooling    = params['pooling']
		self.N_QE       = params['N_QE']
		self.stage      = params['stage']

		print "\n\n"
		print "dataset:",    self.dataset
		print "image_path:", self.image_path
		print "dimension:",  self.dimension
		print "pooling:",    self.pooling
		print "N_QE:",       self.N_QE
		print "stage:",      self.stage
		print "\n\n"


		with open(params['frame_list'],'r') as f:
			self.database_list = f.read().splitlines()
		print "len of database:", len(self.database_list)
		print "example:",         self.database_list[0]
		print "\n"

		with open(params['query_list'],'r') as f:
			self.query_names = f.read().splitlines()
		print "len of queries:", len(self.query_names)
		print "example:",        self.query_names[0]
		print "\n"

		# Distance type
		self.dist_type = params['distance']
		print "dist_type:", self.dist_type
		print "\n"

		# Database features ---

		# PCA MODEL - use paris for oxford data and vice versa
		# They are exchanged on purpose. PCA models are trained on Oxford to be applied on Paris and viceversa.
		if self.dataset is 'paris':
			pca_model_pkl = params['pca_model'] + '_oxford.pkl'
			self.pca      = pickle.load(open(pca_model_pkl, 'rb'))
		elif self.dataset is 'oxford':
			pca_model_pkl = params['pca_model'] + '_paris.pkl'
			self.pca      = pickle.load(open(pca_model_pkl, 'rb'))
		print "pca_model_pkl:", pca_model_pkl
		print "\n"

		# Load features
		feats_path    = params['database_feats']
		self.db_feats = pickle.load(open(feats_path,'rb'))
		print "feats_path:", feats_path
		print "\n\n"

		# ########################################################
		print "Applying PCA"

		self.db_feats   = normalize(self.db_feats)
		if self.pooling is 'sum':
			self.db_feats = self.pca.transform(self.db_feats)
			self.db_feats = normalize(self.db_feats)

		print "PCA done!"
		print "\n\n"
		# ########################################################

		# Where to store the rankings
		self.rankings_dir = params['rankings_dir']
		print "rankings_dir:", self.rankings_dir
		print "\n\n"

		# time.sleep(sleep_time)

	def get_distances(self):

		distances = pairwise_distances(self.query_feats, self.db_feats, self.dist_type, n_jobs=-1)

		return distances

	def query_info(self, filename):

		'''
		For oxford and paris, get query frame and box
		'''

		data = np.loadtxt(filename, dtype="str")

		if self.dataset is 'paris':
			query = data[0]
		elif self.dataset is 'oxford':
			query = data[0].split('oxc1_')[1]

		bbx = data[1:].astype(float).astype(int)

		if self.dataset is 'paris':
			query = os.path.join(self.image_path, query.split('_')[1], query + '.jpg')
		elif self.dataset is 'oxford':
			query = os.path.join(self.image_path, query + '.jpg')

		return query, bbx

	def get_query_vectors(self):

		self.query_feats = np.zeros((len(self.query_names), self.dimension))

		i = 0
		for query in self.query_names:
			# query info
			query_file, box       = self.query_info(query)

			print "\nqf: {}, box: {}".format(query_file, box)

			# get feat
			self.query_feats[i, :] = self.db_feats[np.where(np.array(self.database_list) == query_file)]

			# add top elements of the ranking to the query
			if self.stage is 'QE':

				qe_query_file = os.path.join(self.rankings_dir, os.path.basename(query.split('_query')[0]) +'.txt')
				print "qe_query_file:", qe_query_file
				
				with open(qe_query_file, 'r') as f:
					ranking = f.read().splitlines()

				for i_q in range(self.N_QE):
					imfile = ranking[i_q]

					# construct image path
					if self.dataset is 'paris':
						imname = os.path.join(self.image_path, imfile.split('_')[1], imfile + '.jpg')
					elif self.dataset is 'oxford':
						imname = os.path.join(self.image_path, imfile + '.jpg')

					# find feature and add to query
					feat = self.db_feats[np.where(np.array(self.database_list) == imname)].squeeze()

					self.query_feats[i, :] += feat

			# find feature and add to query
			i += 1

		self.query_feats = normalize(self.query_feats)

	def write_rankings(self, final_scores):

		i = 0

		for query in self.query_names:
			scores   = final_scores[i, :]

			ranking  = np.array(self.database_list)[np.argsort(scores)]
			savefile = open(os.path.join(self.rankings_dir, os.path.basename(query.split('_query')[0]) +'.txt'), 'w')

			for res in ranking:
				savefile.write(os.path.basename(res).split('.jpg')[0] + '\n')
			savefile.close()

			i += 1

	def rank(self):
		
		print "Fetching query vectors..."
		
		self.get_query_vectors()

		print "Computing distances..."

		t0           = time.time()
		distances    = self.get_distances()
		final_scores = distances

		print "Done. Time elapsed", time.time() - t0

		print "Writing rankings to disk..."
		t0 					 = time.time()
		self.write_rankings(final_scores)
		print "Done. Time elapsed", time.time() - t0


if __name__ == "__main__":
	''''''

	R = Ranker(params)
	R.rank()
