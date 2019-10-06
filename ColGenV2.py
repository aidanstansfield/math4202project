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

initH = ()

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
        
leafArcs = genLeaf(graph)
arcChoices = [l for l in leafArcs]
for i in range(numSearchers - len(leafArcs) + 1):
    newArc = random.choice(Arcs)
    while newArc in arcChoices:
        newArc = random.choice(Arcs)
    arcChoices.append(newArc)

pathCombs = list(itertools.combinations(arcChoices, numSearchers))

def updateHist(hist, searched):
#    print('Searched: ', searched)
    temp = list(hist)
    for s in searched: 
        if getEdge(s) not in temp:
            temp.append(getEdge(s))
    return tuple(temp)

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

initPaths = {1: [(c, Cost(c, 1)) for c in pathCands]}

succ = {(1, startP[0][0]): list(itertools.product(*[[s for s in arcCon[a]] for a in 
        startP[0][0]])) for startP in initPaths[1]}
pred = {}
t = 1
print(len(initPaths[t]))
found = 0
while t <= maxTime:
    print('t = ', t)
    initPaths[t + 1] = []
#    for i in initPaths[t]:
#        p = i[0][0]
#        h = i[0][1]
#        if t >= 2:
#            pred[t, p] = [q[0][0] for q in initPaths[t - 1] if succ[t - 1, q[0][0]] == p]
#        for q in succ[t, p]:
##            print('TEST: ', [[x for x in arcCon[a]] for a in q])
#            succ[t + 1, q] = list(itertools.product(*[[x for x in arcCon[a]] for a in q]))
#            print(succ[t + 1, q])
#            for s in succ[t + 1, q]:
#                newPath = (s, updateHist(h, s))
#                initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
    for i in initPaths[t]:
        if t >= 2:
#            print('hi')
            pred[t, i[0][0]] = [q[0][0] for q in initPaths[t - 1] if
                 succ[t - 1, q[0][0]] == i[0][0]]
#        print(len(succ[t, i[0][0]]))
#        msvcrt.getch()
        for q in succ[t, i[0][0]]:
#            print('do thing')
                
#            print(len(updateHist(h, q)))
            newH = updateHist(i[0][1], q)
            if len(newH) - len(i[0][1]) <= 2 and t < 5:
#                print('no')
                continue
            elif len(newH) - len(i[0][1]) == 0 and t >= 5:
                continue
            sCands = list(itertools.product(*[[x for x in arcCon[a]]
            for a in q]))
            for s in sCands:
                count = 0
                for a in s:
                    if a in i[0][1]:
                        count += 1
                if count >= 2:
                    sCands.remove(s)
            succ[t + 1, q] = sCands
#            print(succ[t + 1, q])
#            msvcrt.getch()
            
            newPath = (q, newH)
            cost = Cost(newPath, t + 1)
            if cost < EPS:
                found = 1
            initPaths[t + 1].append((newPath, Cost(newPath, t + 1)))
    if found == 1:
        print('Breaking')
        tFin = t + 1
        break
#    print(succ.keys())
    print(len(initPaths[t + 1]))
    t += 1
    
validEnds = []
for end in initPaths[tFin]:
    if end[1] < EPS:
        validEnds.append(end)
print('Num Ends: ', len(validEnds))

P = {t: [] for t in range(1, tFin + 1)}
P[tFin] = validEnds

for t in range(1, tFin):
    for p in initPaths[tFin - t]:
        for s in succ[tFin - t, p[0][0]]:
            if s in P[tFin - t + 1]:
                P[tFin - t].append(p)
                break
    print(len(P[tFin - t]))
#Restricted Master Problem
RMP = Model('Search Paths')

#variables
Z = {(p, )}