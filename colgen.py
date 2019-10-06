from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time
import math
import random

#some lower bound
EPS = 0.001

a = 19
b = 15
probType = 1
graph, prob, Edges, seed = generateNetwork(a, b, probType)
print('genArcs')
Arcs = genArcs(graph)
print('genNodes')
Nodes = genNodes(graph)
numSearchers = 3
K = range(numSearchers)
maxTime = 50
T = range(1, maxTime + 1)

#solve the mip on the same network
#mipVal, _, mipTime = MIP(probType, numSearchers, a, b, maxTime, 
#                         graph = graph, p = prob, edges = Edges, graphSeed = seed)

print('init')
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
print('dist')

#min distance between each pair of arcs (2 + dist between end node of first arc
#and start node of second arc)
dist = {(a1, a2): abs(indexToXY(a1[1])[0] - indexToXY(a2[0])[0]) + 
                    abs(indexToXY(a1[1])[1] - indexToXY(a2[0])[1]) + 2
                    for a1 in Arcs for a2 in Arcs}

#arcs that can succeed an arc in a graph
def genSuccessors(graph, arcs):
    successors = {}
    for a in arcs:
        suclist = []
        for x in arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors
#print('arcCon')
arcCon = genSuccessors(graph, Arcs)

def canConnect(p1, p2):
    for i in K:
        if p2[i] not in arcCon[p1[i]]:
            return False
    return True

def getEdge(a):
    for e in Edges:
        if O[a, e]:
            return e

def pickCorners(Arcs, leafArcs):
    #corners should have 2 neighbours
    corners = []
    maxDense = 4
    for a in Arcs:
        if a in leafArcs:
            continue
        if len(arcCon[a]) == 4:
            corners.append(a)
    return corners

initH = ()
leafArcs = genLeaf(graph)
if len(leafArcs) == numSearchers:
    pathCombs = [tuple(leafArcs)]
elif len(leafArcs) < numSearchers:
    print('start gen corners')
    corners = pickCorners(Arcs, leafArcs)
    print(corners)
    print('start gen extras')
    if numSearchers - len(leafArcs) == 1:
        extras = [(c,) for c in corners]
    else:
        extras = list(itertools.combinations(corners, numSearchers - len(leafArcs)))
#    print(extras)
#    print('start gen combs')
    pathCombs = [tuple(a for a in leafArcs) + e for e in extras]
else:
    pathCombs = list(itertools.combinations(leafArcs, numSearchers))
    
def updateHist(hist, searched):
#    print('Searched: ', searched)
    temp = list(hist)
    for s in searched: 
        if getEdge(s) not in temp:
            temp.append(getEdge(s))
    return tuple(temp)



#
def Cost(p, t):
    if t == 0 or t == maxTime:
        factor = 1/2
    else:
        factor = 1
#    newH = updateHist(p[1], p[0])
    alpha = 1
    for e in p[1]:
        alpha -= prob[e]
    return factor * alpha

pathCands = [(c, updateHist(initH, c)) for c in pathCombs]
#    
##initPaths = {(1, c): 
##                [(c, Cost(c, 1))] for c in pathCands}
initPaths = {1: [(c, Cost(c, 1)) for c in pathCands]}

succ = {(1, startP[0][0]): list(itertools.product(*[[s for s in arcCon[a]] for a in 
        startP[0][0]])) for startP in initPaths[1]}
pred = {}
t = 1

while t <= maxTime:
    print('t = ', t)
    initPaths[t + 1] = []
    for i in initPaths[t]:
        p = i[0][0]
        h = i[0][1]
        if t >= 2:
            pred[t, p] = [q[0][0] for q in initPaths[t - 1] if succ[t - 1, q[0][0]] == p]
        for q in succ[t, p]:
#            print('TEST: ', [[x for x in arcCon[a]] for a in q])
            succ[t + 1, q] = list(itertools.product(*[[x for x in arcCon[a]] for a in q]))
#            print(succ[t + 1, q])
#            for s in succ[t + 1, q]:
#                newPath = (s, updateHist(h, s))
#                initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
    for i in initPaths[t]:
        initPaths[t + 1] = []
        for q in succ[t, i[0][0]]:
#            print(len(updateHist(h, q)))
            newPath = (q, updateHist(h, q))
            initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
#    print(succ.keys())
    t += 1
#print(initPaths)
P = {(t, s, x[0]): [] for t in T for s in pathCands for x in initPaths[maxTime]}
go = 1
n = 0
for end in initPaths[maxTime]:
    if end[1] < EPS:
        n += 1
used = [p for p in pathCombs]
while n == 0:
    go += 1
    print(go)
    arcChoices = []
    pathCombs = []
    while len(pathCombs) == 0:
        for i in range(10):
            newArc = random.choice(Arcs)
            while newArc in arcChoices:
                newArc = random.choice(Arcs)
            arcChoices.append(newArc)
        for i in range(len(leafArcs)):
            arcChoices.append(leafArcs[i])
        pathChoices = list(itertools.combinations(arcChoices, numSearchers))
        for c in pathChoices:
            if c not in used:
                pathCombs.append(c)
    pathCands = [(c, updateHist(initH, c)) for c in pathCombs]
    initPaths = {1: [(c, Cost(c, 1)) for c in pathCands]}
    succ = {(1, startP[0][0]): list(itertools.product(*[[s for s in arcCon[a]] for a in 
        startP[0][0]])) for startP in initPaths[1]}
    pred = {}
    t = 1

    while t <= maxTime:
#        print('t = ', t)
        initPaths[t + 1] = []
        for i in initPaths[t]:
            p = i[0][0]
            h = i[0][1]
            if t >= 2:
                pred[t, p] = [q[0][0] for q in initPaths[t - 1] if succ[t - 1, q[0][0]] == p]
            for q in succ[t, p]:
    #            print('TEST: ', [[x for x in arcCon[a]] for a in q])
                succ[t + 1, q] = list(itertools.product(*[[x for x in arcCon[a]] for a in q]))
    #            print(succ[t + 1, q])
    #            for s in succ[t + 1, q]:
    #                newPath = (s, updateHist(h, s))
    #                initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
        for i in initPaths[t]:
            initPaths[t + 1] = []
            for q in succ[t, i[0][0]]:
    #            print(len(updateHist(h, q)))
                newPath = (q, updateHist(h, q))
                initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
    #    print(succ.keys())
        t += 1
    for end in initPaths[maxTime]:
        if end[1] < EPS:
            n += 1
print('WE DID IT!!!')
print(pathCombs)
print(initPaths[maxTime])