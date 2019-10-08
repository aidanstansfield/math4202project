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
#T = range(1, maxTime + 1)

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
        
def pickExtras(Arcs, leafArcs):
    cands = []
    dense = 4
    for a in Arcs:
        if a in leafArcs:
            continue
        if len(arcCon[a]) == dense:
            cands.append(a)
    return cands
        
leafArcs = genLeaf(graph)
arcChoices = [l for l in leafArcs]
extras = pickExtras(Arcs, leafArcs)
for i in range(numSearchers - len(leafArcs) + 1):
    newArc = random.choice(extras)
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
    if t == maxTime:
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
succ = defaultdict(list)
for startP in initPaths[1]:
    succ[1, startP[0][0]] = list(itertools.product(*[[s for s in arcCon[a]] for a in 
        startP[0][0]]))
#n = 0
#for p in initPaths[0]:
#    for s in succ[0, p[0][0]]:
#        n += 1
#print('n = ', n)
#msvcrt.getch()
#succ = {(0, startP[0][0]): list(itertools.product(*[[s for s in arcCon[a]] for a in 
#        startP[0][0]])) for startP in initPaths[0]}
pred = defaultdict(list)
t = 1
print(len(initPaths[t]))
found = 0
validEnds = []
while t < maxTime:
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
    lbH1 = math.ceil(numSearchers * 1/3) + 1
    lbT1 = math.ceil(len(Edges)/numSearchers * 1/3)
    lbH2 = math.ceil(numSearchers * 2/3) + 1
    lbT2 = math.ceil(len(Edges)/numSearchers * 2/3)
    while len(initPaths[t + 1]) == 0:
        lbH1 -= 1
        lbH2 -= 1
        for i in initPaths[t]:
    #        print(len(succ[t, i[0][0]]))
    #        msvcrt.getch()
    #        print('P: ', pred[t, i[0][0]])
    #        print(i[0][0])
            for q in succ[t, i[0][0]]:
    #            print('do thing')
                temp = list(itertools.product(*[[x for x in arcCon[a]]
                    for a in q]))
                for s in temp:
                    density = 0
                    count = 0
                    for a in s:
                        if getEdge(a) in i[0][1]:
                            count += 1
                        density += len(arcCon[a])
                    if count >= 2 or density >= 3 * numSearchers + 1:
                        continue
                    succ[t + 1, q].append(s)
    #            print(len(updateHist(h, q)))
                newH = updateHist(i[0][1], q)
                if len(newH) - len(i[0][1]) < lbH1 and \
                    t <= lbT2:
    #                print('no')
                    continue
#                elif len(newH) - len(i[0][1]) <= lbH2 and \
#                    t <= lbT2:
                    continue
                elif len(newH) - len(i[0][1]) == 0 and t >= \
                    math.ceil(len(Edges)/numSearchers * 2/3):
                    continue
    #            elif len(newH) - len(i[0][1]) == 0 and t >5:
    #                continue
                
    #            print(sCands)
    #            if t >= 6:
    #                print('|sc| = ', len(sCands))
    #            print(q)
    #            print(succ[t + 1, q])
    #            msvcrt.getch()
    #            pred[t + 1, q].append(i[0][0])
    #            if t >= 6:
    #                print('|succ| = ', len(succ[t + 1, q]))
    #            print(succ[t + 1, q])
    #            msvcrt.getch()
                
                newPath = (q, newH)
                cost = Cost(newPath, t + 1)
                if cost < EPS:
                    found = 1
                    validEnds.append((newPath, cost))
                initPaths[t + 1].append((newPath, cost))
    #            print('hi')
    #        if t >= 1:
    #            pred[t, i[0][0]] = [q[0][0] for q in initPaths[t - 1] if
    #                 i[0][0] in succ[t - 1, q[0][0]]]
        print(len(initPaths[t + 1]))
    t += 1
    if found == 1:
        print('Breaking')
        tFin = t + 1
        print('tFin: ', tFin, len(initPaths[tFin]))
#        print('|validEnds| = ', validEnds)
#        validEnds = []
#        for i in initPaths[tFin]:
#            if i[1] < EPS:
##                for p in initPaths[tFin - 1]:
##                    if i[0][0] in succ[tFin - 1, p[0][0]]:
##                        pred[tFin, i[0][0]].append(p[0][0])
##                pred[tFin, i[0][0]] = [p[0][0] for p in initPaths[tFin - 1]
##                if i[0][0] in succ[tFin - 1, p[0][0]]]
#                validEnds.append(i)
        break
    #    print(succ.keys())
        
    
#thing = [0 for t in range(tFin)]
#for t in range(tFin):
#    for p in initPaths[t]:
#        for q in initPaths[t + 1]:
#            if q[0][0] in succ[t, p[0][0]]:
#                thing[t] += 1
#print(thing)
#print(tFin)
#msvcrt.getch()
#for (t, _) in pred.keys():
#    print(t)
#validEnds = []
#for end in initPaths[tFin]:
#    if end[1] < EPS:
#        validEnds.append(end)
print('Num Ends: ', len(validEnds), 'tFin = ', tFin)

P = {t: [] for t in range(1, tFin + 1)}
P[tFin] = validEnds

def revHist(h, prevPath):
    temp = list(h)
    for a in prevPath:
        #if double ups for edges in that path
        if getEdge(a) not in temp:
            temp.remove(getEdge(a))
    return tuple(temp)

for t in range(tFin):
    print('t = ', tFin - t)
    for p in initPaths[tFin - t - 1]:
        for q in P[tFin - t]:
            if p not in P[tFin - t - 1] and q[0][0] in succ[tFin - t - 1, p[0][0]]:
                P[tFin - t - 1].append(p)
                break
#    for p in P[tFin - t - 1]:
#        print('|pred| = ', len(pred[tFin - t]))
##        print('COST: ', p[1])
##        print(len(p[0][1]))
##        print(p[0][0])
##        print(len(pred[tFin - t, p[0][0]]))
#        for q in pred[tFin - t, p[0][0]]:
##            print('q = ', q)
##            for x in initPaths[tFin - t - 1]:
##                if q == x[0][0]:
##                    P[tFin - t - 1].append(x)
##                    break
#            prevPath = (q, revHist(p[0][1], q))
#            pOption = (prevPath, Cost(prevPath, tFin - t - 1))
#            P[tFin - t - 1].append(pOption)
#            if pOption in initPaths[tFin - t - 1]:
##            print('newt: ', tFin - t - 1)
#                P[tFin - t - 1].append(pOption)
#            else:
#                print('not an option')
#        for s in succ[tFin - t, p[0][0]]:
#            if s in P[tFin - t + 1]:
#                P[tFin - t].append(p)
#                break
    print(len(P[tFin - t]))

#for t in range(1, tFin):
#    for p in P[t]:
#        count = 0
#        for q in P[t]:
#            if p == q:
#                count += 1
#        if count >= 2:
#            print('OH DEAR')
T = range(1, tFin + 1)
#num = [0 for t in T[:-1]]
#
#for t in T[:-1]:
#    for p in P[t]:
#        for q in P[t + 1]:
#            if q[0][0] in succ[t, p[0][0]]:
#                num[t] += 1
#print(num)
#msvcrt.getch()
#Restricted Master Problem
RMP = Model('Search Paths')

#variables
Z = {(p, t): RMP.addVar(vtype = GRB.BINARY) for t in T for p in P[t]}

#objective
RMP.setObjective(quicksum(p[1] * Z[p, t] for (p, t) in Z), GRB.MINIMIZE)

#constraints
onePathPerTime = {t: RMP.addConstr(quicksum(Z[p, t] for p in P[t]) == 1) for t in T}
continuity = {(p, t): RMP.addConstr(Z[p, t + 1] <= quicksum(Z[q, t] for q in P[t]
                if p[0][0] in succ[t, q[0][0]])) for t in T[:-1] for p in P[t + 1]}
    
RMP.optimize()
RMPval = RMP.objVal + 1


MIP, _, time = MIP(probType, numSearchers, 19, 15, maxTime, graph, Edges, prob)
print('RMP: ', RMPval)
print('MIP: ', MIP.objVal)
displayGraph(graph)