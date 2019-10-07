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
graph, prob, edges, seed = generateNetwork(numEdges, numNodes, UNIFORM, seed)


M = range(0, numEdges)
N = range(0, numNodes)
L = range(0, 2 * numEdges)
T = range(0, maxTime + 1)

# gen set of arcs
arcs = genArcs(graph)
# gen set of nodes
nodes = genNodes(graph)
            
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

for pathLength in range(2, 3):
    addPathsOfLength(pathLength)
"""
for i in range(5):
    makePathsBigger(paths)"""

P = range(len(paths))

S = defaultdict(int)
E = defaultdict(int)
O = {}
# define values for functions S, E and O and store in dict.
for l in L:
    for m in M:
        if arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
            O[l, m] = 1
        else:
            O[l, m] = 0
    """for n in N:
        if arcs[l][0] == nodes[n]:
            S[l, n] = 1
            E[l, n] = 0
        elif arcs[l][1] == nodes[n]:
            E[l, n] = 1
            S[l, n] = 0
        else:
            S[l, n] = 0
            E[l, n] = 0"""
for p in P:
    for n in N:
        if paths[p][0][0] == nodes[n]:
            S[p, n] = 1
            #E[p, n] = 0
        if paths[p][-1][1] == nodes[n]:
            #S[p, n] = 0
            E[p, n] = 1
        """else:
            S[p, n] = 0
            E[p, n] = 0"""

mip = Model("Model searchers searching for a randomly distributed immobile" \
            "target on a unit network")

# variables
# if arc l is traversed at time t
X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
# if edge m has been traversed by time t
Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
# if path p is traversed starting at time t
Z = {(t, p): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for p in P if (t + len(paths[p]) <= maxTime)}
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

"""
# conserve arc flow on nodes
consFlow = {(t, n): mip.addConstr(quicksum(E[l, n] * X[t, l] for l in L) == 
                      quicksum(S[l, n]* X[t + 1, l] for l in L)) for t in 
                        T[1: -1] for n in N}
"""
# conserve path flow on nodes
consFlow = {(t, n): mip.addConstr(quicksum(E[p, n] * Z[t, p] for p in P if ((t, p) in Z and (t + len(paths[p]) < maxTime))) <= 
            quicksum(S[p, n] * Z[t + len(paths[p]), p] for p in P if (t + len(paths[p]), p) in Z)) 
            for t in T[1:-1] for n in N}

# link Z to X
setX = {(t, p): mip.addConstr(Z[t, p] * len(paths[p]) <= 
            quicksum(X[t + i, arcs.index(paths[p][i])] for i in range(len(paths[p])))) 
            for (t, p) in Z}

mustUseZ = mip.addConstr(quicksum(Z[1,p] for p in P) == K)

# initially, no edges have been searched
initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
# limit y so that every edge is searched by T
{m: mip.addConstr(Y[maxTime, m] == 1) for m in M}


# Changinge Branch priority based on aggregation
#XT = mip.addVar(vtype=GRB.INTEGER)
#mip.addConstr(XT==quicksum(X.values()))
#XT.BranchPriority = 10
#mip.setParam('GURO_PAR_MINBPFORBID', 1)
mip.setParam('Method',2)
mip.optimize()

state = {
        "X": X,
        "Y": Y,
        "T": T,
        "L": L,
        "maxTime": maxTime
    }
visualiseStrategy(state, arcs, graph)