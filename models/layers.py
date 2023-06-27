import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import csv

"""
	GDN Layers
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, train_neg, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		# stored parameters
		self.features = features.cuda(1)
		self.pos_vector = None
		self.neg_vector = None

		# Functions
		self.softmax = nn.Softmax(dim=-1)
		# self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		# self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos
		self.train_neg = train_neg

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []


		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			self.pos_index = torch.LongTensor(self.train_pos).cuda(1)
			self.neg_index = torch.LongTensor(self.train_neg).cuda(1)
		else:
			self.pos_index = torch.LongTensor(self.train_pos)
			self.neg_index = torch.LongTensor(self.train_neg)

	def forward(self, nodes):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
									set.union(*to_neighs[2], set(nodes)))
		self.unique_nodes = unique_nodes

		# intra-aggregation steps for each relation
		r1_feats = self.intra_agg1.forward(nodes, r1_list)
		r2_feats = self.intra_agg2.forward(nodes, r2_list)
		r3_feats = self.intra_agg3.forward(nodes, r3_list)

		self_feats = self.fetch_feat(nodes)


		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
		combined = F.relu(cat_feats.mm(self.weight.cuda(1)).t())
		return combined

	def fetch_feat(self, nodes):
		index = torch.LongTensor(nodes).cuda(1)
		return self.features(index)



class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.sigmod = nn.Sigmoid()
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))


		self.r = nn.Parameter(torch.FloatTensor( 1,self.feat_dim)).cuda(1)
		init.normal_(self.r)
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, to_neighs_list):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
		:param train_flag: indicates whether in training or testing mode
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		samp_neighs = [set(x) for x in to_neighs_list]
		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda(1)
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)  # mean aggregator
		if self.cuda:
			self.features = self.features.cuda(1)
			self_feats = self.features(torch.LongTensor(nodes).cuda(1))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(1))
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		# mask = neibor_select(self, self_feats, mask, embed_matrix).cuda(1)
		agg_feats = mask.mm(embed_matrix).cuda(0) 
		agg_feats = agg_feats.cuda(1)
		cat_feats = torch.cat((self_feats, agg_feats), dim=1).cuda(1)  # concat with last layer

		to_feats = F.relu(cat_feats.mm(self.weight.cuda(1)))
		return to_feats
def neibor_select(self, self_feats, mask, nei_feats):
	
	# mask = mask.clone()
	scores = []
	mask_list = []
	# torch.save(mask, 'mask.pt')
	len_node = len(mask)
	simi_avg_list = []
	self_feats = self_feats + self.r.repeat(self_feats.shape[0],1)
	for i in range(len_node):
		node_feats = self_feats[i]
		node_nei = mask[i].nonzero()
		nei_feat = nei_feats[node_nei].squeeze(dim=0)
		simi = []
		if (len(nei_feat) == 1):
			simi.append(torch.dot(node_feats, nei_feat[0]))
		else:
			for j in range(len(nei_feat)):
				simi.append(F.cosine_similarity(node_feats, nei_feat[j]))
		simi = torch.tensor(simi)
		simi_avg = simi.mean()
		simi_avg_list.append(simi_avg)
		simi_the = sum(simi_avg_list) / len(simi_avg_list)
		simi_var = simi.var()

		mask_index = [i for i,x in enumerate(simi) if x< simi_the]
		if (len(mask_index)==0):
			continue
		else:
			mask_node = node_nei[mask_index]
			mask[i][mask_node] = 0.6
		return mask

