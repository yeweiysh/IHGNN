from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
from mlp import MLP

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init, gnn_spmm


class IHGNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 64, 32], num_layers=5, k=30, conv1d_kws=[0, 1], conv1d_activation='ReLU'):
        print('Initializing IHGNN')
        super(IHGNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.num_layers = num_layers
        conv1d_kws[0] = self.latent_dim[0]

        self.dropout = nn.ModuleList()
        #for i in range(self.num_layers):
        #    self.dropout.append(nn.Dropout(p=0.5))
        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, self.latent_dim[0]))
        for i in range(1, self.num_layers):
           self.conv_params.append(nn.Linear(2 * self.latent_dim[0], self.latent_dim[0]))
       
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(MLP(2, num_node_feats, self.latent_dim[0], self.latent_dim[0]))
        for i in range(1, self.num_layers):
            self.mlps.append(MLP(2, 3 * self.latent_dim[0], self.latent_dim[0], self.latent_dim[0]))

        self.dense_dim = self.k * self.num_layers * self.latent_dim[0]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.aggregate_combine(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def aggregate_combine(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            #input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        cur_message_layer = node_feat
        cat_message_layers = []
        ego = self.mlps[0](cur_message_layer)
        cat_message_layers.append(ego)

        for layer in range(1, self.num_layers):
            neig = gnn_spmm(n2n_sp, ego) # Y = A * Y
            agg = torch.cat((ego, neig), 1)
            neig_ego = neig + ego
            agg = torch.cat((agg, neig_ego), 1)
            ego = self.mlps[layer](agg)
            #h = F.relu(ego)
            cat_message_layers.append(ego)

        out = torch.cat(cat_message_layers, 1)

        wl_color = ego[:, -1]
        batch_sort_graphs = torch.zeros(len(graph_sizes), self.k, self.num_layers  * self.latent_dim[0])
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sort_graphs = batch_sort_graphs.cuda()

        batch_sort_graphs = Variable(batch_sort_graphs)
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = wl_color[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sort_graph = out.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.num_layers * self.latent_dim[0])
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sort_graph = torch.cat((sort_graph, to_pad), 0)
            batch_sort_graphs[i] = sort_graph
            accum_count += graph_sizes[i]

        to_conv1d = batch_sort_graphs.view((-1, 1, self.k * self.num_layers * self.latent_dim[0]))
  
        to_dense = to_conv1d.view(len(graph_sizes), -1)
        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
