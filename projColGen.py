from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time
import math

#print statements are just debugging things like trying to figure out what's 
#being slow

#some lower bound
EPS = 0.001

a = 200
b = 150
probType = 1
graph, prob, Edges, seed = generateNetwork(a, b, probType)
print('genArcs')
Arcs = genArcs(graph)
print('genNodes')
Nodes = genNodes(graph)
numSearchers = 30
K = range(numSearchers)
maxTime = 70

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
print('arcCon')
arcCon = genSuccessors(graph, Arcs)
#
#def getConnections(path):
#    options = []
#    for r in path:
#        options.append(arcCon[r[-1]])
#    con = list(itertools.product(*options))
#    return con

#return edge containing arc
def getEdge(a):
    for e in Edges:
        if O[a, e]:
            return e
#cost of traelling path p
def Cost(p):
    cost = 0
    for t in range(1, len(p)):
        cost += 1/2 * (2 - sum(prob[e] for e in p[1][:len(p) - t - 1])) - \
            sum(prob[e] for e in p[1][:len(p) - t])
    return cost
    
#whether a path is legal or not and its cost
def isLegal(path):
    if len(path[0]) > maxTime:
        return (False, GRB.INFINITY)
    if len(path[0]) > 1:
        for (a1, a2) in zip(path[0], path[0][1:]):
            if a2 not in arcCon[a1]:
                return (False, GRB.INFINITY)
    return (True, Cost(path))

#update search history by removing searched arcs
def updateHist(hist, searched):
#    print('Searched: ', searched)
    temp = list(hist)
    if searched in temp:
        temp.remove(searched)
    return tuple(temp)

#initial search history- no edges explored
initH = [e for e in Edges]
print('Constr paths')

#paths we consider adding
candPaths = [[([a], updateHist(initH, a)) for a in Arcs]]
#paths we can add to feasible region
legalPaths = {((a,), updateHist(initH, a)): isLegal(((a,), updateHist(initH, a)))[1] 
                for a in Arcs}
t = 0
#generate paths up to length maxTime/numSearchers - 1
while t <= math.ceil(maxTime/numSearchers) and len(candPaths[t]) > 0:
    print('t = ', t, len(legalPaths), len(candPaths[t]))
    candPaths.append([])
    for p in candPaths[t]:
#        print(p)
        for a in arcCon[p[0][-1]]:
            newPath = (p[0] + [a], updateHist(p[1], a))
            candPaths[t + 1].append(newPath)
            legal, pCost = isLegal(newPath)
            if legal:
#                print(newPath[1])
                legalPaths[tuple(newPath[0]), newPath[1]] = pCost
    print(len(candPaths[t]))
    t += 1
    
print('Num Starting Paths: ', len(legalPaths))
#startPaths = {(k, aStart): [p for p in legalPaths if p[0][0] == aStart] 
#            for k in K for aStart in Arcs}
P = [p for p in legalPaths if len(p[0]) in 
     range(math.ceil(maxTime/numSearchers) - 2, math.ceil(maxTime/numSearchers))]
#model
RMP = Model('Search Path')

#variables
#Z[p, k] = 1 if searcher k searches path p
Z = {(p, k): RMP.addVar(vtype = GRB.BINARY) for p in P for k in K}
#U[k, e] = # of edges searched by searcher k that have already been explored by
#prev searchers
U = {(k, e): RMP.addVar(vtype = GRB.INTEGER) for k in K for e in Edges}

#1 if a path contains an edge
def gamma(p, e):
    for a in p:
        if O[a, e]:
            return 1
    return 0

#objective
RMP.setObjective(quicksum(Cost(p)* Z[p, k] for (p, k) in Z) + \
                    quicksum(prob[e] * U[k, e] for (k, e) in U))
    
#constraints
print('def onePathSearched')
onePathSearched = {k: RMP.addConstr(quicksum(Z[p, k] for p in P) == 1)
                    for k in K}
print('def eachEdgeSearched')
eachEdgeSearched = {e: RMP.addConstr(quicksum(Z[p, k] * gamma(p[0], e) for 
                        (p, k) in Z) >= 1) for e in Edges}
print('def defU')
defU = {(k, e): RMP.addConstr(U[k, e] == quicksum(Z[p, kk] * gamma(p[0], e)
            for (p, kk) in Z if kk < k)) for k in K for e in Edges}
RMP.setParam('OutputFlag', 1)

RMP.optimize()

#add column to problem
def addPath(p, k):
    if (p, k) in Z:
        print('***************DUPLICATE********************', p, k)
    else:
        Z[p, k] = RMP.addVar(obj = Cost(p))
        print('ADDING COL')
        for e in Edges:
            RMP.chgCoeff(eachEdgeSearched[e], gamma(p[0], e))
        RMP.chgCoeff(onePathSearched, 1)
        for kk in K:
            if k < kk:
                for e in E:
                    RMP.chgCoeff(defU[kk], gamma(p[0], e))
#legal paths paired with start arc


def genColumns():
    colsAdded = 0
    for (k, aStart) in startPaths:
        #label with keys being (searcher, start arc, end arc) and values being
        #cost of path to get to aEnd and the path we take
        label = {(k, aStart, aEnd): (Cost(startPaths[k, aStart]) - \
                  sum(eachEdgeSearched[getEdge(a)].pi for a in startPaths[k, aStart][0]) - \
                        onePathSearched[k].pi - sum(defU[k, getEdge(a)].pi
                        for a in startPaths[k, aStart][0]), startPaths[k, aStart]) for aEnd in Arcs
                        if startPaths[k, aStart][0][-1] == aEnd}
#        #generate more columns by using and extending label
        for a1 in Arcs:
            for a2 in arcCon[a1]:
                if (k, aStart, a1) not in label:
                    continue
                newPath = (label[k, aStart, a1][1][0] + (a2,), 
                           updateHist(label[k, aStart, a1], a2))
                legal, pCost = isLegal(newPath)
                if not legal:
                    continue
                rCost = pCost - sum(eachEdgeSearched[getEdge(a)].pi 
                        for a in newPath[0]) - \
                        onePathSearched[k].pi - sum(defU[k, getEdge(a)].pi
                        for a in newPath[0])
                if (k, aStart, a2) not in label or rCost < label[k, aStart, a2][0]:
                    label[k, aStart, a2] = (rCost, newPath)
        #add columns to problem if negative reduced cost
        for (k, aStart, aEnd) in label:
            if label[k, aStart, aEnd][0] < -EPS:
                addPath(label[k, aStart, aEnd][1], k)
                colsAdded += 1
    return colsAdded

#thresh = 0.9
#while thresh > 0.69:
#    while True:
#        RMP.optimize()
##        print(RMP.objVal)
##        msvcrt.getch()
#        print('Down to ', RMP.objVal)
##        msvcrt.getch()
#        Added = genColumns()
#        print('Added: ', Added)
##        msvcrt.getch()
#        if Added == 0:
#            break
#    numFixed = 0
#    for (p, k) in Z:
#        if Z[p, k].x > thresh and Z[p, k].x < 1 - EPS:
#            #set lower bound to be 1
#            Z[p, k].lb = 1
#            numFixed += 1
#    if numFixed == 0:
#        thresh -= 0.1
#    else:
#        print('*************************Fixed ', numFixed, thresh, '*************************')
#        
#RMP.setParam('OutputFlag', 1)
#
#for (p, k) in Z:
#    Z[p, k].vtype = GRB.BINARY
#RMP.setParam('MIPGap', EPS)
#RMP.optimize()
#
#for (p, k) in Z:
#    Z[p, k].lb = 0
#
#RMP.optimize()
                        
