import networkx as nx
import random
import pyscipopt
from pyscipopt import quicksum
import os


def gen_graph(n, g_type='barabasi_albert', edge=4):
    if g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n = n, m = edge)
    for edge in nx.edges(g):
        g[edge[0]][edge[1]]['weight'] = random.uniform(0,1)
    for node in g.nodes():
        g.nodes[node]['weight'] = random.uniform(0,1)
    return g


def getEdgeVar(m, v1, v2, vert):
    u1 = min(v1, v2)
    u2 = max(v1, v2)
    if not ((u1, u2) in vert):
        vert[(u1, u2)] = m.addVar(name='u%d_%d' %(u1, u2),
                                   vtype='B')

    return vert[(u1, u2)]


def getNodeVar(m, v, node):
    if not v in node:
        node[v] = m.addVar(name='v%d' %v,
                            vtype='B')

    return node[v]


def createOptVC(G):
    m = pyscipopt.Model()
    edgeVar = {}
    nodeVar = {}
    for j, (v1, v2) in enumerate(G.edges()):
        node1 = getNodeVar(m, v1, nodeVar)
        node2 = getNodeVar(m, v2, nodeVar)

        m.addCons((node1 + node2) >= 1)

    m.setObjective(quicksum(G.nodes[v]['weight'] * getNodeVar(m, v, nodeVar) for v in G.nodes()), sense = "minimize")
    return m


def generate(filename, seed=1, gtype="barabasi_albert", number_of_nodes=500, param=4):
    G = gen_graph(number_of_nodes, gtype, param)
    P = createOptVC(G)
    P.writeProblem(filename)


def main():
    # 1. Create a folder called 'mvc'
    if not os.path.exists('../mvc/'):
        os.mkdir('../mvc/')

    for i in range(1, 101):
        filename = os.path.join('../mvc/', f'mvc_{i}.lp')
        generate(filename, seed=i+8915, number_of_nodes=9000, param=5)


if __name__ == '__main__':
    main()