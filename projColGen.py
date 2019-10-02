from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time

a = 19
b = 15
probType = 0
graph, prob, Edges, seed = generateNetwork(a, b, probType)
Arcs = genArcs(graph)
Nodes = genNodes(graph)
K = 3
maxTime = 21
leafArcs = genLeaf(graph)

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

def genSuccessors(graph, arcs):
    successors = {}
    for a in arcs:
        suclist = []
        for x in arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

arcCon = genSuccessors(graph, Arcs)


def getConnections(r):
    options = []
    for a in r:
        options.append(arcCon[a])
    cons = list(itertools.product(*options))
    return cons

def canConnect(r1, r2):
    if r2 in getConnections(r1):
        return True
    return False

#if len(leafArcs) == K:
#    print('LA = K')
#    initr = [tuple(leafArcs)]
#elif len(leafArcs) > K:
#    print('LA > K')
#    if K == 1:
#        combs = [tuple(l) for l in leafArcs]
#    else:
#        combs = list(itertools.combinations(leafArcs, K))
#else:
#    print('LA < K')
#    if K- len(leafArcs) == 1:
#        cands = [a for a in Arcs if a not in leafArcs]
#        print(cands)
#        combs = []
#        for c in cands:
#            temp = leafArcs + [c]
#            combs.append(tuple(temp))
#    else:
#        cands = list(itertools.combinations([a for a in Arcs if a not in leafArcs], K - len(leafArcs)))
#        print(cands)
#        combs = []
#        for c in cands:
#            temp = leafArcs + [x for x in c]
#            combs.append(tuple(temp))
#    initr = combs

def isLegal(path):
    if len(path) > maxTime:
        return (False, 1000000)
    if len(path) > 1:
        for (a1, a2) in zip(path, path[1: ]):
            if a2 not in arcCon[a1]:
                return (False, 100000)
    cost = 0
    t = 0
    for a in path:
        cost += sum(prob[e] for e in Edges if O[a, e]) * t
        t += 1
    return (True, cost)
    

#print(initr)
#candPaths = [[[r] for r in list(itertools.combinations(Arcs, K))]
candPaths = [[[a] for a in Arcs]]
print(candPaths)
#print(candPaths)
legalPaths = {(a, ): isLegal([a])[1] for a in Arcs}
t = 0
while t < maxTime/K and len(candPaths[t]) > 0:
    print('t = ', t, len(candPaths[t]), len(legalPaths))
    candPaths.append([])
    for p in candPaths[t]:
        for r in arcCon[p[-1]]:
            newPath = p + [r]
            candPaths[t + 1].append(newPath)
            legal, cost = isLegal(newPath)
            if legal:
                legalPaths[tuple(newPath)] = cost
#        for r in arcCon[tuple(a)]:
#            newPath = a + [r]
#            candPaths[t + 1].append(newPath)
#            legal, cost = isLegal(newPath)
#            if legal:
#                legalPaths[tuple(newPath)] = cost
    t += 1
    
print('Num Starting Paths: ', len(legalPaths))
