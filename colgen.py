from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt

def updateSearch(path, unexplored, O):
    temp = []
    for e in unexplored:
        if e not in path and (e[1], e[0]) not in path:
            temp.append(e)
    return temp

def genSuccessors(graph, arcs):
    successors = {}
    for a in arcs:
        suclist = []
        for x in arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

def getConnections(path, succ, K):
    options = []
    for a in path:
        options.append(succ[a])
    pathsucc = list(itertools.product(*options))
    return pathsucc

def getProb(p, path, hist, O):
    prob = 0
    for a in path:
        for e in hist:
            if O[a, e]:
                prob += p[e]
    return prob


def genPred(start, path, succ, initUnexp, O, numEdges):
    connected = getConnections(path, succ, K)
    unexplored = updateSearch(path, initUnexp, O)
    if len(unexplored) == numEdges:
        pred = None

def genFeas(graph, prob, K, Arcs, Edges, Nodes, maxTime):
    S = {}
    E = {}
    O = {}
    #define values for functions S, E and O and store in dict.
    for a in Arcs:
        for e in Edges:
            if ((a[0] == e[0]) and (a[1] == e[1])) or ((a[0] == e[1]) and (a[1] == e[0])):
                O[a, e] = 1
            else:
                O[a, e] = 0
        for n in Nodes:
            if a[0] == n:
                S[a, n] = 1
                E[a, n] = 0
            elif a[1] == n:
                E[a, n] = 1
                S[a, n] = 0
            else:
                S[a, n] = 0
                E[a, n] = 0
    succ = genSuccessors(graph, Arcs)
    leafArcs = genLeaf(graph)
    initUnexp= [e for e in Edges]
#    print('INITUNEXP: ', initUnexp)
    if len(leafArcs) == K:
        print('LA = K')
        start = [leafArcs]
    elif len(leafArcs) > K:
        print('LA > K')
        start = list(itertools.combinations(leafArcs, K))
    else:
        print('LA < K')
        candArcs = []
        for a in Arcs:
            if a in leafArcs:
                pass
            else:
                candArcs.append(a)
        candArcs.append(leafArcs)
#        print('cand', candArcs)
#        for n in range(len(Arcs)):
#            print(n)
#            if Arcs[n] in leafArcs:
#                candArcs.pop(n)
        start = list(itertools.combinations(candArcs, K - len(leafArcs) + 1))
#        print('BEFORE: ', start)
        temp = []
        for s in range(len(start)):
            slist = []
            hasLeaf = 0
            for x in start[s]:
                if type(x) is list:
                    for a in x:
                        slist.append(a)
                    hasLeaf = 1
                else:
                    slist.append(x)
            if hasLeaf:
                temp.append(slist)
        start = temp
    H = {s: updateSearch(start[s], Edges, O) for s in range(len(start))}
    print(start)
    canReach = {s: [start[s]] + [c for c in itertools.combinations(Arcs, K)
                    if all(
                        [getProb(p, c, H[s], O) > 0,
                         len(H[s]) - len(updateSearch(c, H[s], O)) == K,
                         c in getConnections(start[s], succ, K)
                         ]
                            )
                    
                ]
                for s in range(len(start))}
    
    return canReach
                
                
g, p, Edges, _ = generateNetwork(19, 15, 0)
Arcs = genArcs(g)
Nodes = genNodes(g)
#hist = Edges
#print(hist)
#K = 3
#succ = genSuccessors(g, Arcs)
#a1 = Arcs[3]
#a2 = Arcs[25]
#a3 = Arcs[17]
#maxTime = math.ceil(2 * len(Edges)/K + 5)
canReach = genFeas(g, p, K, Arcs, Edges, Nodes, maxTime)
print(canReach)
#print(Arcs)
#path = [a1, a2, a3]
#con = genConnections(path, succ, K)
#prob = getProb(p, path, hist, O)
#print('P: ', path)
#print(con)
#print(prob)

