from typing import Dict, Any

import networkx as nx
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import torch
import torch_geometric.nn as pyg_nn
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import from_networkx, to_networkx, train_test_split_edges
from tqdm.notebook import tqdm

from torch_geometric.nn import GCN, MLP, GCNConv, GATConv
from torchinfo import summary

import copy
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as py_T
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from tqdm.notebook import tqdm

def encode_evidence(evidence,encoder,evidence_list):
    if pd.notna(evidence):
        return encoder.transform([[evidence]])[0]
    else:
        return [0] * len(evidence_list)

def build_graph(data):
    evidence_list = data['DirectEvidence'].dropna().unique()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(evidence_list.reshape(-1, 1))

    data['EncodedEvidence'] = data['DirectEvidence'].apply(lambda x: encode_evidence(x, encoder, evidence_list))

    G = nx.Graph()

    genes = data[['GeneID', 'GeneSymbol']].drop_duplicates()
    G.add_nodes_from(
        (row.GeneID, {'type': 'gene', 'label': row.GeneSymbol})
        for row in genes.itertuples(index=False)
    )

    diseases = data[['DiseaseID', 'DiseaseName']].drop_duplicates()
    G.add_nodes_from(
        (row.DiseaseID, {'type': 'disease', 'label': row.DiseaseName})
        for row in diseases.itertuples(index=False)
    )

    edges = [
        (row.GeneID, row.DiseaseID, {'evidence': torch.tensor(row.EncodedEvidence, dtype=torch.float)})
        for row in data.itertuples(index=False)
    ]
    G.add_edges_from(edges)

    for node, data in G.nodes(data=True):
        is_gene = 1 if data['type'] == 'gene' else 0
        is_disease = 1 - is_gene
        degree = G.degree(node)
        G.nodes[node]['x'] = torch.tensor([is_gene, is_disease, degree], dtype=torch.float)

    return G

def plot_graph(G,pos,labels,colors):
    plt.figure(figsize=(20, 20))
    nx.draw_networkx(nx.draw_networkx(G,pos= pos,
                                      labels= labels,
                                      node_color=colors,
                                      with_labels=True,
                                      edge_color='black',
                                      node_size=1000,
                                      font_size=8,
                                      font_color='black',
                                      font_weight='bold'))
    plt.show()

def get_subgraph(G, sample_size=1000):
    sub_edge = list(G.edges())[:sample_size]
    H = G.edge_subgraph(sub_edge).copy()

    return H

def compute_bar_data(df, key_col, value_col):
    """
    Computes a dictionary for bar charting: how many unique `value_col`s
    are associated with each `key_col`.

    Example: gene -> number of diseases

    Parameters:
    - df: pd.DataFrame containing the dataset
    - key_col: str, column to use as keys (e.g. 'GeneSymbol')
    - value_col: str, column to count (e.g. 'DiseaseName')

    Returns:
    - dict: {key1: count_of_unique_values, key2: count_of_unique_values, ...}
    """
    if key_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns not found in dataset: {key_col}, {value_col}")
    grouped = df.groupby(key_col)[value_col].nunique()
    return grouped.to_dict()


def get_subgraph_by_label(G: nx.Graph, label: str) -> Dict[str, Any]:
    # Find nodes matching the label (case insensitive)
    matched_nodes = [n for n, data in G.nodes(data=True) if data.get('label', '').lower() == label.lower()]

    if not matched_nodes:
        return {"error": f"No node found with label '{label}'"}

    # For simplicity, take the first matched node
    node = matched_nodes[0]

    # Get the node itself + its immediate neighbors (1-hop subgraph)
    nodes_to_include = list(G.neighbors(node)) + [node]
    subgraph = G.subgraph(nodes_to_include).copy()

    # Prepare JSON serializable structure
    nodes = []
    for n, data in subgraph.nodes(data=True):
        nodes.append({
            "id": n,
            "label": data.get('label'),
            "type": data.get('type')
        })

    edges = []
    for u, v, data in subgraph.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "evidence": data.get('evidence').tolist() if data.get('evidence') is not None else None
        })

    return {
        "nodes": nodes,
        "edges": edges
    }

# -------------------------------------------------------
#                   Prediction
# -----------------------------------------------------
class LinkPredModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        dot_product: bool = True,
    ):
        super(LinkPredModel, self).__init__()

        # Convolution layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        if not dot_product:
            self.classifier = torch.nn.Linear(2 * hidden_dim, 1)  # hardcoded output dim

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dot_product = dot_product

    def forward(self, x, edge_index, edge_label_index):

        # Compute embeddings
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        if self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index)

        x_src = x[edge_label_index[0]]  # Take the embeddings of the source nodes
        x_trg = x[edge_label_index[1]]  # Take the embeddings of the targe nodes

        # dot product h^t * h
        if self.dot_product:
            out = torch.sum(x_src * x_trg, dim=1)  # Dot product
        else:
            # Concat and MLP version
            x = torch.cat([x_src, x_trg], dim=1)
            # x = torch.sum([x_src, x_trg], dim = 1)
            out = self.classifier(x).squeeze()

        return out

    def loss(self, preds, link_label):
        return self.loss_fn(preds, link_label.type(preds.dtype))

def merge_edge_labels(data):
    edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    edge_label = torch.cat([data.pos_edge_label, data.neg_edge_label], dim=0)
    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    return data

def train(model, data_train, data_val, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_label_index)
    loss = model.loss(out, data_train.edge_label)
    loss.backward()
    optimizer.step()

    loss_train, acc_train = test(model, data_train)
    loss_val, acc_val = test(model, data_val)
    return loss_train, acc_train, loss_val, acc_val

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    loss = model.loss(out, data.edge_label.type(out.dtype))
    probs = torch.sigmoid(out)
    preds = (probs > 0.5).float()
    correct = (preds == data.edge_label).sum().item()
    acc = correct / data.edge_label.size(0)
    return loss.item(), acc

@torch.no_grad()
def predict_new_edges(model, data, threshold=0.9):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    probs = torch.sigmoid(out)
    # Select edges with probability above threshold (likely new connections)
    mask = probs > threshold
    predicted_edges = data.edge_label_index[:, mask].T.cpu().tolist()
    predicted_scores = probs[mask].cpu().tolist()

    # Prepare output list of dicts (for JSON)
    new_links = []
    for (src, tgt), score in zip(predicted_edges, predicted_scores):
        new_links.append({"source": src, "target": tgt, "score": score})

    return new_links
