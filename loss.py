import torch    
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def init_center(args,input_g, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    if args.cuda_id!='0' :
        c = torch.zeros(args.n_hidden)
    else:
        c = torch.zeros(args.n_hidden, device=torch.device('cuda:1'))

    model.eval()
    with torch.no_grad():
        # if args.module in [ 'GCN', 'GraphSAGE', 'SVDD_Attr','SVDD_Stru']:
        outputs = model.getembeds(input_g)
        # else:
        #     outputs,rec = model(input_g, input_feat)
        # get the inputs of the batch

        n_samples = outputs.shape[1]
        c =torch.sum(outputs, dim=1)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c