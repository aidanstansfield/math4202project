from gurobipy import *
from problemGen import generateNetwork
from collections import defaultdict
from fractions import Fraction

numEdges = 28
numNodes = 12
K = 1
maxTime = 50

M = range(1, numEdges + 1)
N = range(1, numNodes + 1)
L = range(1, 2 * numEdges + 1)
T = range(1, maxTime + 1)

mip = Model("Model searchers searching for a randomly distributed immobile" \
    "target on a unit network")

graph = generateNetwork(numEdges, numNodes)

count = 1
transition = {}
for node in graph:
    transition[node] = count
    count += 1

mygraph = {}
for node in graph:
    mygraph[transition[node]] = [transition[x] for x in graph[node]]

p = {}
for m in M:
    p[m] = float(Fraction(1, numEdges))
#p = [float(Fraction(1, edges))] * (edges + 1)

arcs = []
edges = []
arcCount = 1
edgeCount = 1
S = defaultdict(int)
E = defaultdict(int)
O = defaultdict(int)
for node in mygraph:
    for endNode in mygraph[node]:
        if (node, endNode) in arcs:
            print("duplicate arc")
            exit(1)
        arcs.append((node, endNode))
        S[node, arcCount] = 1
        E[endNode, arcCount] = 1
        O[edgeCount, arcCount] = 1
        arcCount += 1
        if (node, endNode)[::-1] not in edges:
            edges.append((node, endNode))
            edgeCount += 1

X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T for l in L}
Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in range(0, maxTime + 1) for m in M}
alpha = {t: mip.addVar() for t in range(0, maxTime + 1)}

mip.setObjective(quicksum((alpha[t-1]+alpha[t])/2 for t in T), GRB.MINIMIZE) #2

for t in T:
    mip.addConstr(quicksum(X[t, l] for l in L) <= K) #3
    mip.addConstr(alpha[t] == 1 - quicksum(p[m] * Y[t, m] for m in M)) # 6
    for m in M:
        mip.addConstr(Y[t, m] <= Y[t - 1, m] + quicksum(O[m, l] * X[t, l] for l in L)) #5 

for t in range(1, maxTime):
    for n in N:
        mip.addConstr(quicksum(E[n, l] * X[t, l] for l in L) == 
                      quicksum(S[n, l] * X[t + 1, l] for l in L)) # 4

mip.addConstr(alpha[maxTime] <= 0.001)
for m in M:
    mip.addConstr(Y[0, m] == 0)

mip.optimize()

print("arc count", arcCount)
print("edge count", edgeCount)
print("len s", len(S))
print('len e', len(E))
print("len o", len(O))
    





#def run_instance(graph):
