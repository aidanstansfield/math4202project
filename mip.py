from gurobipy import *
from problemGen import generateNetwork, genArcs, genNodes, S, E, O
from collections import defaultdict
from fractions import Fraction

numEdges = 19
numNodes = 15
K = 1
maxTime = 50

M = range(0, numEdges)
N = range(0, numNodes)
L = range(0, 2 * numEdges)
T = range(1, maxTime + 1)

mip = Model("Model searchers searching for a randomly distributed immobile" \
    "target on a unit network")

#sparse instance, uniform prob
graph, pUn, edges = generateNetwork(numEdges, numNodes, 0)
print('Prob: ', pUn)
print('Edges: ', edges)

arcs = genArcs(graph)
nodes = genNodes(graph)

X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T for l in L}
Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in range(0, maxTime + 1) for m in M}
alpha = {t: mip.addVar() for t in range(0, maxTime + 1)}

mip.setObjective(quicksum((alpha[t - 1] + alpha[t])/2 for t in T), GRB.MINIMIZE) #2

#num arcs searched can't exceed num searchers
searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T}
#define alpha as in paper
defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p * Y[t, m] for m in 
            M for p in pUn.keys() if edges[m] in pUn[p])) for t in T}
#update search info after every time step
updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
                quicksum(O(arcs[l], edges[m]) * X[t, l] for l in L)) for t in 
                T for m in M}
#conserve arc flow on nodes
consFlow = {(t, n): mip.addConstr(quicksum(E(arcs[l], nodes[n]) * X[t, l] for l in L) == 
                      quicksum(S(arcs[l], nodes[n])* X[t + 1, l] for l in L)) for t in 
                        range(1, maxTime) for n in N}
#initially, no edges have been searched
initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
#limit alpha so target is found by time maxTime
mip.addConstr(alpha[maxTime] <= 0.001)

mip.optimize()
    
