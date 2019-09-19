from gurobipy import *
from problemGen import generateNetwork, genArcs, genNodes, displayLattice
from collections import defaultdict
from fractions import Fraction
import math

def genNeighbours(edges):
    M = range(len(edges))
    neighbours = {}
    for m1 in M:
        neighbours[edges[m1]] = []
        for m2 in M:
            #if edges share a node
            if edges[m1][0] == edges[m2][0] or edges[m1][0] == edges[m2][1] or\
            edges[m1][1] == edges[m2][0] or edges[m1][1] == edges[m2][1]:
                neighbours[edges[m1]].append(edges[m2])
    return neighbours

def MIP(probType, K, numEdges, numNodes):
    #min time for searchers to search every arc
    maxTime = math.ceil(2 * numEdges/K)
    print('yes')
#    graph, p, edges = genEx(probType)
#    print('yes2')
#    #sets
#    numEdges = len(edges)
#    numNodes = len(graph.keys())
    M = range(0, numEdges)
    N = range(0, numNodes)
    L = range(0, 2 * numEdges)
    T = range(0, maxTime + 1)
    
    #data
    #gen network - p is pdf and edges is set of edges
    graph, p, edges = generateNetwork(numEdges, numNodes, probType)
    S = {}
    E = {}
    O = {}
    #gen set of arcs
    arcs = genArcs(graph)
    print(arcs)
    #gen set of nodes
    nodes = genNodes(graph)
    print('E: ', edges)
    #define values for functions S, E and O and store in dict.
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
    
    mip = Model("Model searchers searching for a randomly distributed immobile" \
                "target on a unit network")
    
    #variables
    X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
    Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
    alpha = {t: mip.addVar() for t in T}
    
    #objective
    mip.setObjective(quicksum((alpha[t - 1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)
    
    #constraints
    
    #num arcs searched can't exceed num searchers- implicitly defines capacity
    searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) == K) for t in T[1:]}
    #capacity of flow on each arc
#    lowerBound = {(t, l): mip.addConstr(X[t, l] >= 0) for t in T[1:] for l in L}
    #lower bound on flow
    #define alpha as in paper
    defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]] * Y[t, m] for m in 
                M)) for t in T}
    #update search info after every time step
    updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
                    quicksum(O[l, m] * X[t, l] for l in L)) for t in 
                    T[1:] for m in M}
    #conserve arc flow on nodes
    consFlow = {(t, n): mip.addConstr(quicksum(E[l, n] * X[t, l] for l in L) == 
                          quicksum(S[l, n]* X[t + 1, l] for l in L)) for t in 
                            T[1: -2] for n in N}
    #initially, no edges have been searched
    initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
    #limit y so that every arc is searched by T
    {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}
    
    mip.optimize()
    
    return mip.objVal, graph
    
