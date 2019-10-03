from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time
import math

EPS = 0.001

a = 19
b = 15
probType = 0
graph, prob, Edges, seed = generateNetwork(a, b, probType)
Arcs = genArcs(graph)
Nodes = genNodes(graph)
numSearchers = 3
K = range(numSearchers)
maxTime = 21
#leafArcs = genLeaf(graph)

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

dist = {(a1, a2): abs(indexToXY(a1[1])[0] - indexToXY(a2[0])[0]) + 
                    abs(indexToXY(a1[1])[1] - indexToXY(a2[0])[1]) + 2
                    for a1 in Arcs for a2 in Arcs}

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

def getConnections(path):
    options = []
    for r in path:
        options.append(arcCon[r[-1]])
    con = list(itertools.product(*options))
#    print(temp)
#    pathsucc = []
#    for s in temp:
#        testPath = updateSearch(s, (path, currentHist))
#        if not canReach(testPath, t, maxTime, K):
#            pass
#        doubleBacks = 0
#        for a in path:
#            for x in s:
#                if x[1] == a[0]:
#                    doubleBacks += 1
##        print(doubleBacks)
#        if doubleBacks < 2:
#            pathsucc.append(s)
    return con

#def getConnections(r):
#    options = []
#    for a in r:
#        options.append(arcCon[a])
#    cons = list(itertools.product(*options))
#    return cons

def canConnect(r1, r2):
    if r2 in arcCon[r1]:
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
    
def getEdge(a):
    for e in Edges:
        if O[a, e]:
            return e

def routeCost(route, t, hist):
    newSearched = []
    for a in Arcs:
        if a != route[t - 1]:
            continue
        for e in Edges:
            if O[a, e] == 1 and e not in hist:
                hist.append(e)
                newSearched.append(e)
    cost = sum(prob[e] for e in newSearched)
#    print('Cost: ', cost)
    return cost, hist

def getHist(path):
    hist = []
    for a in path:
        for e in Edges:
            if O[a, e] and e not in hist:
                hist.append(e)
    return hist

def pathCost(pList):
    hist = []
    T = range(max([len(p) for p in pList]))
    alpha = [1 for t in T]
    for t in T[1:]:
        alpha[t] = alpha[t - 1]
        for k in K:
            sCost, hist = routeCost(pList[k], t, hist)
#            print('sCost = ', sCost)
            alpha[t] -= sCost
#        print(alpha[t])
    totCost = sum((alpha[t - 1] + alpha[t])/2 for t in T[1:])
    print(totCost)
    return totCost
    

def isLegalSub(p):
    if len(p) > maxTime:
        return False
    if len(p) > 1:
        for (a1, a2) in zip(p, p[1:]):
            if a2 not in arcCon[a1]:
                return False
    return True

def isLegalPath(sc, pList):
    for p in pList:
        if len(p) > maxTime:
            return (False, GRB.INFINITY)
        if len(p) > 1:
            for (a1, a2) in zip(p, p[1:]):
                if a2 not in arcCon[a1]:
                    return (False, GRB.INFINITY)
    return (True, pathCost(pList))



#print(initr)
#candPaths = [[[r] for r in list(itertools.combinations(Arcs, K))]]
candPaths = [[[a] for a in Arcs]]
#print(candPaths)
#print(candPaths)
legalSubs = [(a,) for a in Arcs if isLegalSub([a])]
#legalSubs = {(a, ): isLegalSub([a]) for a in Arcs}
t = 0
while t < math.ceil(maxTime/numSearchers) and len(candPaths[t]) > 0:
    print('t = ', t, len(legalSubs), len(candPaths[t]))
    candPaths.append([])
    for p in candPaths[t]:
        for r in arcCon[p[-1]]:
            newPath = p + [r]
            candPaths[t + 1].append(newPath)
            legal= isLegalSub(newPath)
            if legal:
                legalSubs.append(tuple(newPath))
#        for r in arcCon[tuple(a)]:
#            newPath = a + [r]
#            candPaths[t + 1].append(newPath)
#            legal, cost = isLegal(newPath)
#            if legal:
#                legalPaths[tuple(newPath)] = cost
    t += 1

#print(legalPaths)
print('Num Starting SubPaths: ', len(legalSubs))
#for p in legalPaths:
#    if legalPaths[p] > 1000:
#        print('O NO ', legalPaths[p])
#        print(p)
#msvcrt.getch()
def hasEdge(path, e):
    for a in path:
        if O[a, e]:
            return True
    return False

    
print('Start Actual Paths')
searcherPaths = {aStart: [p for p in legalSubs if p[0] == aStart and
                     len(p) == math.ceil(maxTime/numSearchers) - 1]
             for aStart in Arcs}
print('Done searcherPaths')
for aStart in searcherPaths:
    print(len(searcherPaths[aStart]))
startPoints = [[(k, aStart) for aStart in searcherPaths] for k in K]
print('Done startPoints')

startCombs = list(itertools.product(*startPoints))
#print(startCombs)
print('Done startCombs,  |sc| = ', len(startCombs))

#print(canReach)
#print(canReach)
print('Start validPaths')
validPaths = {}
for sc in startCombs:
    pathList = []
    for k in K:
        cand = []
        for p in searcherPaths[sc[k][1]]:
            cand.append(p)
        pathList.append(tuple(cand))
    validPaths[sc] = tuple(pathList)
    print(validPaths[sc])
    msvcrt.getch()
#    validPaths[sc] = tuple(itertools.product(*pathList))
print('Valid Paths: ', sum(len(validPaths[sc]) for sc in validPaths))

print('start redefine')
P = []
for sc in startCombs:
    for p in validPaths[sc]:
        P.append(p)
print('stop redefine, |P| = ', len(P))
for p in P[-3:]:
    print(p)

#print(len(validPaths[startCombs[0]]))
#print(validPaths[startCombs[0]])


#msvcrt.getch()
def delta(p, r):
    if r in p:
        return 1
    return 0
#delta = {}
#for sc in startCombs:
#    for p in validPaths[sc]:
#        for r in legalSubs:
#            if r in p:
#                delta[r, p] = 1
#            else:
#                delta[r, p] = 0
def gamma(r, e):
    for a in r:
        if O[a, e]:
            return 1
    return 0
#print('def gamma')
#gamma = {}
#for r in legalSubs:
#    for a in r:
#        for e in Edges:
#            if O[a, e]:
#                gamma[r, e] = 1
#            else:
#                gamma[r, e] = 0

#print('def canReach')
#canReach = {}
#for sc in startCombs:
#    for p in validPaths[sc]:
#        arcList = []
#        for r in p:
#            for a in Arcs:
#                if dist[r[-1], a] + len(p) > 2 * math.ceil(maxTime/numSearchers):
#                    arcList.append(a)
#        canReach[sc, p] = tuple(arcList)
#print('done canReach')
#Restricted Master Problem
RMP = Model('Search Path')

#variables
#Z[p] = 1 if we choose path p
Z = {p: RMP.addVar() for p in P}
print('|Z| = ', len(Z))
print(len(P))
S = {(r, k): RMP.addVar() for r in legalSubs for k in K}

#objective
RMP.setObjective(quicksum(pathCost(p) * Z[p] for p in Z), GRB.MINIMIZE)

#constraints
print('def onePathSearched')
onePathSearched = RMP.addConstr(quicksum(Z[p] for p in Z) == 1)
print('def ZRestrictsS')
#SRestrictsZ = {p: RMP.addConstr(Z[p] <= quicksum(S[r, k]
#                    for r in legalSubs if delta(p, r) == 1))
#                    for p in Z for k in K}

ZRestrictsS = {p: RMP.addConstr(quicksum(S[r, k] for k in K for r in p) == \
                    numSearchers * Z[p]) for p in Z}

print('def eachEdgeSearched')
eachEdgeSearched = {e: RMP.addConstr(quicksum(S[r, k] * gamma(r, e) for r in legalSubs
                    for k in K) >= 1) for e in Edges}
print('def oneSubPerSearcher')
oneSubPerSearcher = {k: RMP.addConstr(quicksum(S[r, k] for r in legalSubs) == 1)
                        for k in K}
#
##variable: Z[p, k] = 1 if we choose path p for searcher k
#Z = {(p, k): RMP.addVar() for p in legalPaths}
#
#
##objective
#RMP.setObjective(quicksum(legalPaths[p] * Z[p, k] for (p, k) in Z), GRB.MINIMIZE)
#
##constraints
#onePathPerSearcher = {k: RMP.addConstr(quicksum(Z[p, k] for p in 
#                        legalPaths) == 1) for k in K}
#noPathsOverlap = {p: RMP.addConstr(quicksum(Z[p, k] for k in K) <= 1)
#                    for p in legalPaths}
#print('Define eachEdgeSearched')
##############################THIS CONSTRAINT MIGHT BE WRONG##################
#eachEdgeSearched = {e: RMP.addConstr(quicksum(Z[p, k] for (p, k) in Z
#                        if hasEdge(p, e)) >= 1) for e in Edges}
#print('Done')

RMP.setParam('OutputFlag', 1)

print('RMP Defined')




#def genColumns():
#    colsAdded = 0
#    while colsAdded == 0:
#        pi = {e: eachEdgeSearched[e].pi for e in Edges}
#        for aStart in Arcs:
#            dist = {}
#            for k in K:
#                dist[aStart, k] = (pi[getEdge(aStart)], [aStart])
#            for a in Arcs:
#                for k in K:
#                    if (a, k) not in dist:
#                        continue
#                    aDist, pathSeg = dist[a, k]
#    #                    temp = [r for r in Arcs if r != a]
#                    for r in arcCon[a]:
#                        if len(pathSeg) < maxTime:
#                            newDist = aDist + pi[getEdge(r)]
#                            if (r, k) not in dist or newDist > dist[r, k][0]:
#                                dist[r, k] = (newDist, pathSeg + [r])
##            print(dist)
#            for (a, k) in dist:
#                legal, cost = isLegal(dist[a, k][1])
#                #if legal and neg reduced cost
#                if legal and cost - dist[a, k][0] < -EPS:
#                    #add column
#                    col = (tuple(dist[a, k][1]), k)
#                    if col in Z:
#                        print('***************DUPLICATE********************',
#                              col, cost - dist[a, k][0])
#                        print(Z[col].x, Z[col].ub)
#                    Z[col] = RMP.addVar(obj = cost)
#                    colsAdded += 1
#                    for r in col[0]:
#                        RMP.chgCoeff(eachEdgeSearched[getEdge(r)], Z[col], 1)
#            msvcrt.getch()

def addPath(sc, pList):
    for k in K:
        if (pList[k], k) in Z:
            print('***************DUPLICATE********************', path, k)
        else:
            Z[pList[k], k] = RMP.addVar(obj = Cost())
            for a in path:
                RMP.chgCoeff(eachEdgeSearched[getEdge(a)], Z[path, k], 1)
            
#def canReach(aStart):
#    pathList = []
#    for p in legalPaths:
#        if p[0] == aStart:
#            pathList.append(p)
#    return pathList
            


#canReach = {(aStart, p, k): [a for a in Arcs if dist[p[-1], a] + len(p) > 2 * \
#                math.ceil(maxTime/numSearchers)] for aStart in startPaths for p
#                in startPaths[aStart] for k in K}



def extendPath(pList, aList):
    combs = {k: [] for k in K}
    for k in K:
        for a in aList[k]:
            combs[k].append(pList[k] + (a,))
    newPaths = list(itertools.product(*[combs[k] for k in K]))
#        newPaths.append(pList[k] + (aList[k],))
    return tuple(newPaths)

def arcListCon(aList):
    cand = []
    for a in aList:
        cand.append(arcCon[a])
    return list(itertools.product(*cand))
def genColumns():
    colsAdded = 0
    for sc in validPaths:
        for p in validPaths[sc]:
            label = {(sc, p, aList): 
                    (pathCost(extendPath(p, aList)), extendPath(p, aList))
                    for aList in itertools.combinations(Arcs, k) if
                    aList in getConnections(p)}
            for aList in getConnections(p):
                for rList in arcListCon(aList):
                    if (sc, p, aList) not in label:
                        continue
                    newPath = extendPath(p, aList)
                    pCost = pathCost(newPath)
                    rCost = pCost - onePathSearched.pi - \
                            sum(oneSubPerSearcher[k].pi for k in K) -\
                            sum(SRestrictsZ[p, k].pi for sc in startCombs
                                for p in validPaths[sc] for k in K)
                    for r in newPath:
                        rCost -= sum(eachEdgeSearched[getEdge(a)].pi for a in r)
                    if (sc, newPath, rList) not in label or\
                    rCost < label[sc, newPath, rList][0]:
                        label[sc, newPath, rList] = (rCost, newPath)
#    for (sc, p) in canReach:
#        label = {(sc, p, aList): 
#                    (pathCost(sc, extendPath(p, aList)),
#                     extendPath(p, aList))
#                     for aList in canReach[sc, p]}
##        label = {(sc, pList, aList): (pathCost(sc, extendPath(pList, aList)),
##                  extendPath(pList, aList)) for sc in validPaths for pL}
#        for aList in canReach[sc]:
#            for rList in getConnections(aList):
#                if (sc, p, aList) not in label:
#                    continue
#                newPath = extendPath(label[sc, p, aList][1], aList)
#                pCost = pathCost(sc, newPath)
#                rCost = pCost
#                for p in newPath:
#                    rCost -= sum(eachEdgeSearched[getEdge(a)].pi
#                                for a in p)
#                if (sc, newPath, rList) not in label or \
#                    rCost < label[sc, newPath, rList][0]:
#                        label[sc, newPath, rList] = (rCost, newPath)
        for (sc, p, aList) in label:
            if label[sc, p, aList][0] < -EPS:
                addPath(label[sc, p, aList][0])
                colsAdded += 1
#    for (aStart, p, k) in canReach:
#        label = {(a, p, k): (Cost(p + (a,)) - sum(eachEdgeSearched[getEdge(r)].pi
#                             for r in p + (a,)), p + (a,)) for a in canReach[aStart, p, k]
#                                for k in K}
#        for a1 in canReach[aStart, p, k]:
#            for a2 in arcCon[a1]:
#                if (a1, p, k) not in label:
#                    continue
#                newPath = label[a1, p, k][1] + (a2,)
#                pCost = Cost(newPath)
#                if pCost == GRB.INFINITY:
#                    continue
#                rCost = pCost - sum(eachEdgeSearched[getEdge(a)].pi for a in newPath)
#                if (a2, label[a1, p, k][1], k) not in label or \
#                    rCost < label[a2, label[a1, p, k][1], k][0]:
#                    label[a2, label[a1, p, k][1], k] = (rCost, newPath)
#        for (a, p, k) in label:
#            if label[a, p, k][0] < -EPS:
#                addPath(label[a, p, k][1], k)
#                colsAdded += 1
    return colsAdded
#                print('added a col')
#            for a in Arcs:
#                if dist[p[-1], a] + len(p) > 2 * math.ceil(maxTime/numSearchers):
#                    continue
#                for k in K:
#                    label[a, p, k] = (Cost(p + (a,)) - 
#                         sum(eachEdgeSearched[getEdge(r)].pi) for r in p)
#                    for a2 in Arcs
            
            
#        print(canReach[aStart])
        #label with key being current arc and candidate arc
        #and searcher and value of cost and
        #current path
#        for p in canReach[aStart]:
##            print(p)
#            if len(p) == 1:
#                continue
#        for p in canReach[aStart]:
#            print('p: ', type(p))
#            for a in p:
#                print('a: ', type(p))
#        for p in legalPaths:
#            print(legalPaths[p])
#            msvcrt.getch()
        
#        label = {(t, k, aEnd): (legalPaths[])}
#        
#        label = {(pStart, k, aEnd): (legalPaths[pStart] - sum(eachEdgeSearched[getEdge(a)].pi
#                for a in pStart), pStart + (aEnd,)) for pStart in canReach[aStart] for k in K
#                for aEnd in arcCon[pStart[-1]]}
##        print(label)
##        label = {(aStart, k, r): 
##            (Cost((aStart, r)) - eachEdgeSearched[getEdge(aStart)].pi - 
##             eachEdgeSearched[getEdge(r)].pi, (aStart, r))
##            for k in K for r in arcCon[aStart]}
#        pathCands = {}
#        for (pStart, k, r1) in label:
#            pathCands[len(pStart) + 1, (pStart, k, r1)] = label[pStart, k, r1]
#            done = False
#            t = len(pStart) + 1
#            while not done:
#                print(aStart, t)
#                pathCands[t + 1] = []
#                for path in pathCands[t]:
#                    for r in arcCon[path[-1]]:
#                        newPath = path + (r,)
#                        legal, pCost = isLegal(newPath)
#                        if not legal:
#                            continue
#                        rCost = pCost - sum(eachEdgeSearched[getEdge(a)].pi
#                                for a in newPath)
#                        if (newPath, k, r) not in pathCands or rCost < label[newPath, k, r][0]:
#                            pathCands[t + 1, (newPath, )]
#                t += 1
#                if t == 2 * maxTime/numSearchers or len(pathCands) == 0:
#                    done = True
#            for t in pathCands:
##            for r2 in arcCon[r1]:
###                if (p, k, r1) not in label:
###                    continue
##                for k in K:
##                    path = label[p, k, r1][1] + (r2,)
##                    legal, pCost = isLegal(path)
##                    if not legal:
##                        continue
##                    rCost = pCost - sum(eachEdgeSearched[getEdge(a)].pi
##                                        for a in path)
##                    if (p, k, r2) not in label or rCost < label[p, k, r2][0]:
##                        label[p, k, r2] = (rCost, path)
#        for (p, k, a) in label:
#            if label[p, k, a][0] < -EPS:
#                addPath(label[p, k, a][1], k)
#                colsAdded += 1
#    return colsAdded
                    
                        
                
#        for p in canReach[aStart]:
#            for r1 in arcCon[p[-1]]:
#                for r2 in arcCon[r1]:
#                    if (aStart, 0, r1) not in label:
#                        continue
#                    
#        for r1 in arcCon[aStart]:
#            for r2 in arcCon[r1]:
#                for k in K:
#                    if (aStart, k, r1) not in label:
#                        continue
#                    for candPath in label[aStart, k, r1][1]:
#                        path = candPath + (r2, )
#                        pCost = Cost(path)
#                        if pCost == GRB.INFINITY:
#                            continue
#                        newCost = pCost - sum(eachEdgeSearched[getEdge(a)].pi
#                                    for a in path)
#                        if (aStart, k, r2) not in label or newCost < \
#                            label[aStart, k, r2][0]:
#                                label[aStart, k, r2] = (newCost, path)
#        for (aStart, k, r) in label:
#            if label[aStart, k, r][0] < -EPS:
#                addPath(label[aStart, k, r][1], k)
#                colsAdded += 1
#    return colsAdded

thresh = 0.9
while thresh > 0.69:
    while True:
        RMP.optimize()
#        print(RMP.objVal)
#        msvcrt.getch()
        print('Down to ', RMP.objVal)
        msvcrt.getch()
        Added = genColumns()
        print('Added: ', Added)
#        msvcrt.getch()
        if Added == 0:
            break
    numFixed = 0
    for (p, k) in Z:
        if Z[p, k].x > thresh and Z[p, k].x < 1 - EPS:
            #set lower bound to be 1
            Z[p, k].lb = 1
            numFixed += 1
    if numFixed == 0:
        thresh -= 0.1
    else:
        print('*************************Fixed ', numFixed, thresh, '*************************')
        
RMP.setParam('OutputFlag', 1)

for (p, k) in Z:
    Z[p, k].vtype = GRB.BINARY
RMP.setParam('MIPGap', EPS)
RMP.optimize()

for (p, k) in Z:
    Z[p, k].lb = 0

RMP.optimize()