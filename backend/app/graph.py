import networkx as nx
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt

def encode_evidence(evidence,encoder,evidence_list):
    if pd.notna(evidence):
        return encoder.transform([[evidence]])[0]
    else:
        return [0] * len(evidence_list)

def build_graph(data):
    evidence_list = data['DirectEvidence'].dropna().unique()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(evidence_list.reshape(-1, 1))

    data['EncodedEvidence'] = data['DirectEvidence'].apply(encode_evidence)

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
