from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time

#update which edges have been explored
def updateSearch(r, s):
    h = []
    for e in s[1]:
        if e not in r and (e[1], e[0]) not in r:
            h.append(e)
    return (r, tuple(h))
#    temp = []
#    for e in unexplored:
#        if e not in path and (e[1], e[0]) not in path:
#            temp.append(e)
#    return temp

def canReach(s, t, maxTime, K):
    if len(s[1])/K > (maxTime - t)/K:
        return False
    else:
        return True

def getConnections(path, succ, K, t, maxTime, currentHist):
    options = []
    for a in path:
        options.append(succ[a])
    temp = list(itertools.product(*options))
#    print(temp)
    pathsucc = []
    for s in temp:
        testPath = updateSearch(s, (path, currentHist))
        if not canReach(testPath, t, maxTime, K):
            pass
        doubleBacks = 0
        for a in path:
            for x in s:
                if x[1] == a[0]:
                    doubleBacks += 1
#        print(doubleBacks)
        if doubleBacks < 2:
            pathsucc.append(s)
    return pathsucc

def inList(A, p, K):
    instances = list(itertools.combinations(p, K))
    found = 0
    for i in instances:
        if i in A:
            found = 1
    return found

def findLeastDense(graph, Nodes, lb):
    lessDense = []
    minDense = min([len(graph[n]) for n in Nodes if len(graph[n]) > lb])
    for n in Nodes:
        if len(graph[n]) == minDense:
            lessDense.append(n)
    return lessDense

def nextStep(s, t, maxTime, O, succ, K, A):
    print('start search for time ', t)
    if t == maxTime and len(s[1]) > 0:
        return (t, 0, s)
    elif t == maxTime and len(s[1]) == 0:
        return (t, 1)
    else:
#        print('S0: ', s[0])
#        print(s[1])
        found = 0
#        print('start search for con')
        con = getConnections(s[0], succ, K, t, maxTime, s[1])
        print(len(con))
#        msvcrt.getch()
        for r in con:
#            print('route ', r)
#            print('route ', r)
#            print('update search')
            ur = updateSearch(r, s)
#            print(ur[1])
#            print('H decreased by ', len(s[1]) - len(ur[1]))
#            msvcrt.getch()
#            print('check next')
#            print(len(ur[1])/K)
#            print((maxTime - t)/K)
            if nextStep(ur, t + 1, maxTime, O, succ, K, A)[1] == 1:
                print('*****************FOUND ONE**************************')
                A[t + 1].append(ur)
        if found:
            return (t, 1, s)
        else:
            return (t, 0, s)
#def getStartPred(path, tend, start, startSucc):
#    if tend == 2:
#        pred[tend, path] = []
#        for s in start:
#            if path in startSucc[s]:
#                pred[tend, path].append(s)
#        return pred
#    else:
#        pred[tend, path]
    
def getPred(t, start, succ, pathCombs, K):
    if t == 2:
        pred = defaultdict(list)
        for s in start:
            for p in pathCombs:
                if p in getConnections(s, succ, K):
                    pred[t, p].append(s)
    else:
        pred = getPred(t - 1, start, succ, pathCombs, K)
        pathChoices = [k[1] for k in pred.keys() if k[0] == t - 1]
        print(len(pathChoices))
        for c in pathChoices:
            for p in pathCombs:
                if p in getConnections(c, succ, K):
                    pred[t, p].append(c)
    return pred


#generate a list of all arcs that are connected to each arc
def genSuccessors(graph, arcs):
    successors = {}
    for a in arcs:
        suclist = []
        for x in arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors


def genFeas(graph, prob, K, Arcs, Edges, Nodes, maxTime):
    TIMESTART = time.time()
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
    print('start leaf')
    leafArcs = genLeaf(graph)
    print('Leaf Arcs: ', leafArcs)
    initH = tuple(e for e in Edges)
    print('IH: ', initH)
    A = defaultdict(list)
    succ = genSuccessors(graph, Arcs)
    print('done prelim')
    tmin = maxTime
    if len(leafArcs) == K:
        print('LA = K')
        initr = tuple(leafArcs)
        ns = nextStep([initr, initH], 1, tmin, O, succ, K, A)
#        if ns[1] and ns[0] <= tmin:
#            tmin = ns[0]
        A[1].append([initr, initH])
    elif len(leafArcs) > K:
        print('LA > K')
        if K == 1:
            combs = [tuple(l) for l in leafArcs]
        else:
            combs = list(itertools.combinations(leafArcs, K))
        print(len(combs))
        print(combs)
        for r in combs:
            print('start route ', r)
            ns = nextStep([r, initH], 1, maxTime, O, succ, K, A)
            if ns[1] and ns[0] <= tmin:
                tmin = ns[0]
                A[1].append([initr, initH])
    else:
        print('LA < K')
        print('Cands:')
        if K- len(leafArcs) == 1:
            cands = [a for a in Arcs if a not in leafArcs]
            print(cands)
            print('start comb')
            combs = []
            for c in cands:
                temp = leafArcs + [c]
                combs.append(tuple(temp))
        else:
            cands = list(itertools.combinations([a for a in Arcs if a not in leafArcs], K - len(leafArcs)))
            print(cands)
            print('start comb')
            combs = []
            for c in cands:
                temp = leafArcs + [x for x in c]
                combs.append(tuple(temp))
        print(len(combs))
        print(combs)
#        msvcrt.getch()
        print('end comb')
        for r in combs:
            ns = nextStep([r, initH], 1, maxTime, O, succ, K, A)
#            if ns[1] and ns[0] <= tmin:
#                tmin = ns[0]
            A[1].append([r, initH])
    ENDSTART = time.time()
    print('Time taken: ', ENDSTART - TIMESTART)
    return A[1]

#g, prob, Edges, _ = generateNetwork(19, 15, 0)
##displayGraph(g)
#Arcs = genArcs(g)
#Nodes = genNodes(g)
#K = 3
#maxTime = 10
##maxTime = math.ceil(2 * len(Edges)/K + 3)
#A = genFeas(g, prob, K, Arcs, Edges, Nodes, maxTime)
#print(A)
#path = tuple([Arcs[0], Arcs[20], Arcs[11]])
#print(path)
#succ = genSuccessors(g, Arcs)
#currentHist = tuple([e for e in Edges])
#con = getConnections(path, succ, K, 1, maxTime, currentHist)
#print(con)
#displayGraph(g)
    
