import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def community_layout(g, partition):
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g,partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=2)
    pos_communities = nx.spring_layout(hypergraph,**kwargs)

    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos

def _find_between_community_edges(g, partition):
    edges = dict()
    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]
        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]
    return edges

def _position_nodes(g, partition, **kwargs):
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos



def Pos_gene(C):
    pos = dict()
    for i in range(len(C)):
        if C[i]==0:
            X = np.random.uniform(0,2)
            Y = np.random.uniform(0,2)
            pos[i] = np.array([X,Y])
        if C[i]==1:
            X = np.random.uniform(4,6)
            Y = np.random.uniform(0,2)
            pos[i] = np.array([X,Y])
        if C[i]==2:
            X = np.random.uniform(0,2)
            Y = np.random.uniform(4,6)
            pos[i] = np.array([X,Y])
        if C[i]==3:
            X = np.random.uniform(4,6)
            Y = np.random.uniform(4,6)
            pos[i] = np.array([X,Y])
        if C[i]==4:
            X = np.random.uniform(2,4)
            Y = np.random.uniform(7,8)
            pos[i] = np.array([X,Y])
    return(pos)

