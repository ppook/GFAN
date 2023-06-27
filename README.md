# GDN: Alleviating Structrual Distribution Shift in Graph Anomaly Detection
Pytorch Implementation of

GFAN: Enhancing Graph-Based Fraud Detection with Semantic Edge Screening and Representation Co-Trainin

# Overview
The fraud detection is to distinguish the fraudsters from the normal users. Fraudsters usually camouflage themselves through “normal behavior”, say, by intentionally establishing many connections to normal users. Moreover, with the widespread use of the Internet, the types of fraudsters
continue to increase, posing an increasing challenge in detecting them. Most existing work upon Graph Neural Networks (GNNs)
blindly smooth neighbor nodes, thus effect is relatively poor and still leads to the dilution of fraudulent features. Besides, there are few methods consider the performance of the model on emerging fraudsters, which indicates that the generalization performance
of the model is relatively poor. For this purpose, in this work we propose GFAN, a novel Graph Feature enhAncement Network to
tackle the above challenges. Specifically, GFAN provides a specific semantic edge screening module to solve the dilution of fraudulent
feature issue. Such screening module trains the edge facts by learning the connection patterns of normal users, and then filters
the heterogeneous edges by computing the edge confidence scores. Along with edge screening module, we devise a representation enhanced co-training module which trains a hypersphere to enhance the discrimination between the representations of normal nodes
and fraud nodes to highlight hidden fraudsters. Meanwhile, we co-train the fraud detector to obtain more precise classification
boundaries, the accuracy and generalization performance of the fraud detector are also improved. Experimental results clearly show that GFAN outperforms other competitive graph-based fraud detectors on public datasets

<h2 align="center">
<figure> <img src="figures/topology.png" height="300"></figure>
</h2>

Illustration of GDN. The feature separation module
separates the node feature into two sets. Two constraints
are leveraged to assist separation. Blank positions in node
representation mean they are zero when calculating losses.

# Dataset
YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Run `python src/data_process.py` to pre-process the data.

# Dependencies
Please set up the environment following Requirements in this [repository](https://github.com/PonderLY/PC-GNN). 
```sh
argparse          1.1.0
networkx          1.11
numpy             1.16.4
scikit_learn      0.21rc2
scipy             1.2.1
torch             1.4.0
```

# Reproduce
```sh
python main.py --config ./config/gdn_yelpchi.yml
```

# Acknowledgement

