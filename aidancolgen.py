from gurobipy import *
from problemGen import *
from collections import defaultdict
from matplotlib import pyplot, animation
from mip import visualiseStrategy

UNIFORM = 0
NON_UNIFORM = 1

numEdges = 19
numNodes = 15
K = 1
maxTime = 2 * numEdges
seed = 748345644471475368
probType = UNIFORM
maxPathLength = 2
graph, prob, edges, seed = generateNetwork(numEdges, numNodes, probType, seed)

M = range(0, numEdges)
N = range(0, numNodes)
L = range(0, 2 * numEdges)
T = range(0, maxTime + 1)

# gen set of arcs
arcs = genArcs(graph)
# gen set of nodes
nodes = genNodes(graph)
leafs = genLeaf(graph)
            
successors = {arc: [] for arc in arcs}
for arc in arcs:
    thisArcEnd = arc[1]
    for nextArcEnd in graph[thisArcEnd]:
        successors[arc].append((thisArcEnd, nextArcEnd))
        
paths = [[arc] for arc in arcs]

def addPathsOfLength(l):
    added = 0
    for path in paths:
        if len(path) == l - 1:
            # create new path by appending successors
            for successor in successors[path[-1]]:
                newPath = path.copy()
                newPath.append(successor)
                paths.append(newPath)
                added += 1
    print(f"Added {added} paths of length {l}")
    
def makePathsBigger(paths):
    for path in paths:
        for successor in successors[path[-1]]:
            newPath = path.copy()
            newPath.append(successor)
            paths.append(newPath)
        paths.remove(path)

for pathLength in range(2, maxPathLength + 1):
    addPathsOfLength(pathLength)
"""
for i in range(5):
    makePathsBigger(paths)"""

P = range(len(paths))

S = defaultdict(int)
E = defaultdict(int)
arcS = defaultdict(int)
arcE = defaultdict(int)
O = {}
# define values for functions S, E and O and store in dict.
for l in L:
    for m in M:
        if arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
            O[l, m] = 1
        else:
            O[l, m] = 0
    for n in N:
            if arcs[l][0] == nodes[n]:
                arcS[l, n] = 1
            elif arcs[l][1] == nodes[n]:
                arcE[l, n] = 1
for p in P:
    for n in N:
        if paths[p][0][0] == nodes[n]:
            S[p, n] = 1
        if paths[p][-1][1] == nodes[n]:
            E[p, n] = 1

mip = Model("Model searchers searching for a randomly distributed immobile" \
            "target on a unit network")

# variables
# if arc l is traversed at time t
X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
# if edge m has been traversed by time t
Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}

def pathValid(t, p):
    if maxPathLength == 1:
        return True
    if t <= maxTime - maxPathLength + 1 and len(paths[p]) == maxPathLength:
        return t % maxPathLength == 1
    return t + len(paths[p]) - 1 == maxTime
    
# if path p is traversed starting at time t
Z = {(t, p): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for p in P 
     if pathValid(t, p)}
     #if len(paths[p]) == min(maxTime - t + 2, maxPathLength)}
#     if (t + len(paths[p]) <= maxTime + 1) and (len(paths[p]) == maxPathLength 
#         or t >= maxTime - maxPathLength)}
# the probability that the target has not been found by time t
alpha = {t: mip.addVar() for t in T}

# objective
mip.setObjective(quicksum((alpha[t-1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)

# constraints

# num arcs searched can't exceed num searchers- implicitly defines capacity
searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T[1:]}

# define alpha as in paper
defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(prob[edges[m]] * Y[t, m] for m in 
            M)) for t in T}

# update search info after every time step
updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
                quicksum(O[l, m] * X[t, l] for l in L)) for t in 
                T[1:] for m in M}

arcconsFlow = {(t, n): mip.addConstr(quicksum(arcE[l, n] * X[t, l] for l in L) == 
                          quicksum(arcS[l, n]* X[t + 1, l] for l in L)) for t in 
                            T[1: -1] for n in N}

pathconsFlow = {(t, n): mip.addConstr(E[p1, n] * Z[t, p1] <= 
            quicksum(S[p2, n] * Z[t + len(paths[p1]), p2] for p2 in P if (t + len(paths[p1]), p2) in Z)) 
            for n in N for t in T[1:-1] for p1 in P if (t, p1) in Z and t + len(paths[p1]) < maxTime}                          

# link Z to X
setX = {(t, p): mip.addConstr(Z[t, p] * len(paths[p]) <= 
            quicksum(X[t + i, arcs.index(paths[p][i])] for i in range(len(paths[p])))) 
            for (t, p) in Z}

mustUseZ = mip.addConstr(quicksum(Z[1, p] for p in P if (1, p) in Z) == K)
onlyKSearchers = {t: mip.addConstr(quicksum(Z[t, p] for p in P if (t, p) in Z)
                        <= K - quicksum(Z[t2, p] for t2 in 
                        range(t - maxPathLength + 1, t) for p in P if (t2, p) 
                        in Z and t2 + len(paths[p]) > t)) for t in T[1:]}

# initially, no edges have been searched
initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
# limit y so that every edge is searched by T
{m: mip.addConstr(Y[maxTime, m] == 1) for m in M}

if probType == UNIFORM and len(leafs) != 0:
        mip.addConstr(quicksum(X[1, arcs.index(leaf)] for leaf in leafs) >= 1)
           

# Changinge Branch priority based on aggregation
"""
XT = mip.addVar(vtype=GRB.INTEGER)
mip.addConstr(XT==quicksum(X.values()))
XT.BranchPriority = 10
mip.setParam('GURO_PAR_MINBPFORBID', 1)"""
#mip.setParam('Method',2)
mip.optimize()

state = {
        "X": X,
        "Y": Y,
        "T": T,
        "L": L,
        "maxTime": maxTime
    }
visualiseStrategy(state, arcs, graph)