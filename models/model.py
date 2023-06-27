import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
"""
	GDN Model
"""


class GFANLayer(nn.Module):
	"""
	One GDN layer
	"""

	def __init__(self, num_classes, inter1):
		"""
		Initialize GDN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(GFANLayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(dim=-1)
		# self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		# self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		# the parameter to transform the final embedding
		self.sigmoid = nn.Sigmoid()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds1 = self.inter1(nodes)
		scores = self.weight.mm(embeds1)
		return scores.t()

	def to_prob(self, nodes):
		gnn_logits = self.forward(nodes)
		gnn_scores = self.softmax(gnn_logits)
		return gnn_scores

	def loss(self, nodes, labels, Alpha, Beta, center):
		gnn_scores = self.forward(nodes)
		# GNN loss
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		svdd_loss, margin= self.SVDD_loss(nodes, Alpha, labels, center)
		# svdd_loss,margin = 0,0
		loss = gnn_loss + Beta*svdd_loss
		return loss, svdd_loss, gnn_loss, margin
	def SVDD_loss(self, nodes, Alpha, labels, center):
		embed = self.getembeds(nodes).t()
		pos_nodes = [i for i,x in enumerate(labels) if x == 0]
		neg_nodes = [i for i,x in enumerate(labels) if x == 1]
		embed_pos = embed[pos_nodes]
		embed_neg = embed[neg_nodes]
		svdd_loss_pos = torch.zeros(len(pos_nodes)).cuda(1)
		svdd_loss_neg = torch.zeros(len(neg_nodes)).cuda(1)
		for i in range(len(pos_nodes)):
			svdd_loss_pos[i] = torch.norm(embed_pos[i] - center, 2)
		loss_pos = torch.mean(svdd_loss_pos)
		for i in range(len(neg_nodes)):
			svdd_loss_neg[i] = torch.norm(embed_neg[i] - center, 2)
		loss_neg = torch.mean(svdd_loss_neg)
		margin = torch.sqrt(1/(loss_neg-loss_pos)**2)
		loss = loss_pos + Alpha*margin
		return loss, margin

		
	def getembeds(self,nodes):
		embeds = self.inter1(nodes)
		return embeds
	
	# def svdd_score(self,nodes,center):
	# 	embed = self.getembeds(nodes).t()
	# 	svdd_score = torch.zeros(len(nodes)).cuda(1)
	# 	for i in range(len(nodes)):
	# 		svdd_score[i] = (torch.sum((embed[i] - center) ** 2))
	# 	svdd_score = self.sigmoid(svdd_score)
	# 	return svdd_score
