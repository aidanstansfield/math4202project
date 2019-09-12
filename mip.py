from gurobipy import *
from problemGen import generateNetwork, genArcs, genNodes
from collections import defaultdict
from fractions import Fraction

def MIP(probType, K, numEdges, numNodes):
    #min time for searchers to search every arc
    maxTime = 50
    
    #sets
    M = range(0, numEdges)
    N = range(0, numNodes)
    L = range(0, 2 * numEdges)
    T = range(0, maxTime + 1)
    
    #data
    #gen network - p is pdf and edges is set of edges
    graph, p, edges = generateNetwork(numEdges, numNodes, probType)
    print(p)
    S = {}
    E = {}
    O = {}
    #gen set of arcs
    arcs = genArcs(graph)
    #gen set of nodes
    nodes = genNodes(graph)
    
    for l in L:
        for m in M:
            if ((arcs[l][0] == edges[m][0]) and (arcs[l][1] == edges[m][1])) or ((arcs[l][0] == edges[m][1]) and (arcs[l][1] == edges[m][0])):
                O[l, m] = 1
            else:
                O[l, m] = 0
        for n in N:
            if arcs[l][0] == nodes[n]:
                S[l, n] = 1
                E[l, n] = 0
            elif arcs[l][1] == nodes[n]:
                E[l, n] = 1
                S[l, n] = 0
            else:
                S[l, n] = 0
                E[l, n] = 0
    #model
    mip = Model("Model searchers searching for a randomly distributed immobile" \
    "target on a unit network")
    
    #variables
    X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
    Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
    alpha = {t: mip.addVar() for t in T}
    
    #objective
    mip.setObjective(quicksum((alpha[t - 1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)

    #constraints
    #num arcs searched can't exceed num searchers
    searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T[1:]}
    #define alpha as in paper
    defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]]* Y[t, m] for m in 
                M)) for t in T}
    #update search info after every time step
    updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
                    quicksum(O[l, m] * X[t, l] for l in L)) for t in 
                    T[1:] for m in M}
    #conserve arc flow on nodes
    consFlow = {(t, n): mip.addConstr(quicksum(E[l, n] * X[t, l] for l in L) == 
                          quicksum(S[l, n]* X[t + 1, l] for l in L)) for t in 
                            T[1: -1] for n in N}
    #initially, no edges have been searched
    initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
    #limit alpha so target is found by time maxTime
    mip.addConstr(alpha[maxTime] <= 0.001)
    
    mip.optimize()
    
    return mip.objVal, edges, p
    

    
#numEdges = 19
#numNodes = 15
#K = 1
#maxTime = 2 * numEdges
#
#M = range(0, numEdges)
#N = range(0, numNodes)
#L = range(0, 2 * numEdges)
#T = range(0, maxTime)
#
#mip = Model("Model searchers searching for a randomly distributed immobile" \
#    "target on a unit network")
#
##sparse instance, uniform prob
#graph, pUn, edges = generateNetwork(numEdges, numNodes, 0)
#print('Prob: ', pUn)
#print('Edges: ', edges)
#
#arcs = genArcs(graph)
#nodes = genNodes(graph)
#
#X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T for l in L}
#Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
#alpha = {t: mip.addVar() for t in T}
#
#mip.setObjective(quicksum((alpha[t]+alpha[t + 1])/2 for t in range(0, maxTime - 1)), GRB.MINIMIZE) #2
#
##num arcs searched can't exceed num searchers
#searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T}
##define alpha as in paper
#defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p * Y[t, m] for m in 
#            M for p in pUn.keys() if edges[m] in pUn[p])) for t in T}
##update search info after every time step
#updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
#                quicksum(O(arcs[l], edges[m]) * X[t, l] for l in L)) for t in 
#                range(1, maxTime) for m in M}
##conserve arc flow on nodes
#consFlow = {(t, n): mip.addConstr(quicksum(E(arcs[l], nodes[n]) * X[t, l] for l in L) == 
#                      quicksum(S(arcs[l], nodes[n])* X[t + 1, l] for l in L)) for t in 
#                        range(0, maxTime - 1) for n in N}
##initially, no edges have been searched
#initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
##limit alpha so target is found by time maxTime
#mip.addConstr(alpha[maxTime - 1] <= 0.001)
#
#mip.optimize()

    
