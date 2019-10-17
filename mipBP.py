from gurobipy import quicksum, GRB, Model
from problemGen import generateNetwork, genArcs, genNodes, genEdges, displayGraph, indexToXY, generateProbabilities, genLeaf
from time import clock
from matplotlib import pyplot, animation
import msvcrt
import random
import math

UNIFORM = 0
NON_UNIFORM = 1


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


def visualiseStrategy(state, arcs, graph):
    fig = pyplot.figure()
    
    def init():
        for key in graph.keys():
            keyCoords = indexToXY(key)
            for node in graph[key]:
                nodeCoord = indexToXY(node)
                pyplot.plot(
                    [keyCoords[0], nodeCoord[0]],
                    [keyCoords[1], nodeCoord[1]],
                    'b.-'
                )
            pyplot.annotate(str(key), (keyCoords[0], keyCoords[1]))

    def animate(t):
        pyplot.title('Sparse: t=' + str(t+1))
        if t == 0:
            return

        for l in state["L"]:
            if t > 1 and state["X"][t-1, l].x > 0.9:
                XCoords = [indexToXY(arcs[l][0])[0], indexToXY(arcs[l][1])[0]]
                YCoords = [indexToXY(arcs[l][0])[1], indexToXY(arcs[l][1])[1]]
                pyplot.plot(XCoords, YCoords, 'g-')

            if state["X"][t, l].x > 0.9:
                XCoords = [indexToXY(arcs[l][0])[0], indexToXY(arcs[l][1])[0]]
                YCoords = [indexToXY(arcs[l][0])[1], indexToXY(arcs[l][1])[1]]
                pyplot.plot(XCoords, YCoords, 'r-',)


    # Must be assigned to a variable or the animation doesn't play
    anim = animation.FuncAnimation(fig, animate, init_func=init,
            frames=state["maxTime"], interval = 750, repeat=False)

    pyplot.show()

#returns the edge containing the arc a
def getArc(e, Arcs, O):
    for a in Arcs:
        if O[a, e]:
            return a

#function takes set of arcs and the graph and returns dictionary with arcs
#as keys and values as list of arcs that they key arc can connect to
def genSuccessors(graph, Arcs):
    successors = {}
    for a in Arcs:
        suclist = []
        for x in Arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

def getDense(Edges, Arcs, arcCon, O):
    cands = []
    dense = 4
    for e in Edges:
        if len(arcCon[getArc(e, Arcs, O)]) <= dense:
            cands.append(getArc(e, Arcs, O))
    return cands

def getMaxEdges(Edges, prob):
    maxProb = max(prob.values())
    maxEdges = []
    for e in Edges:
        if prob[e] == maxProb:
            maxEdges.append(e)
    return maxEdges

#returns the edge containing the arc a
def getEdge(a, Edges, O):
    for e in Edges:
        if O[a, e]:
            return e

def MIPBP(probType, K, numEdges, numNodes, maxTime, graph = None, Edges = None, prob = None, seed=None):
    #gen network - p is pdf and edges is set of edges
    if seed is None:
        seed = random.seed
    if graph is None:
        graph, prob, Edges, _ = generateNetwork(numEdges, numNodes, probType, seed)
    if prob is None:
        prob, edges = generateProbabilities(graph, probType)
    if Edges is None:
        Edges = genEdges(graph)
    
    T = range(maxTime + 1)
    S = {}
    E = {}
    O = {}
    # gen set of arcs
    Arcs = genArcs(graph)
    # gen set of nodes
    Nodes = genNodes(graph)
    leafArcs = genLeaf(graph)
    #set of all arcConnections
    arcCon = genSuccessors(graph, Arcs)
    
    # define values for functions S, E and O and store in dict.
    for a in Arcs:
        for e in Edges:
            if a == e or (a[1], a[0]) == e:
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
    denseEdges = getDense(Edges, Arcs, arcCon, O)
#    print(denseEdges)
#    for d in denseEdges:
#        if d not in Edges:
#            print('ONO')
    mip = Model("Model searchers searching for a randomly distributed immobile" \
                "target on a unit network")
    
    # variables
    X = {(t, a): mip.addVar(vtype = GRB.BINARY) for t in T[1:] for a in Arcs}
    for (t, a) in X:
        X[t, a].BranchPriority = max(1, math.ceil(2 * maxTime) - 5 * t)
    Y = {(t, e): mip.addVar(vtype=GRB.BINARY) for t in T for e in Edges}
    YT = {t: mip.addVar(vtype = GRB.INTEGER) for t in range(1, 4)}
    c = {t: mip.addConstr(quicksum(Y[t, e] for e in Edges) == YT[t]) for t in YT}
    for t in YT:
        YT[t].BranchPriority = 10 - 2 * t
    alpha = {t: mip.addVar() for t in T}
    
    # objective
    mip.setObjective(quicksum((alpha[t-1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)
    
    # constraints
    
    # num arcs searched can't exceed num searchers- implicitly defines capacity
    searcherLim = {t: mip.addConstr(quicksum(X[t, a] for a in Arcs) <= K) for t in T[1:]}
    # capacity of flow on each arc
    #    lowerBound = {(t, l): mip.addConstr(X[t, l] >= 0) for t in T[1:] for l in L}
    # lower bound on flow
    # define alpha as in paper
    defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(prob[e] * Y[t, e] for e in 
                Edges)) for t in T}
    # update search info after every time step
    updateSearch = {(t, e): mip.addConstr(Y[t, e] <= Y[t - 1, e] + 
                    quicksum(O[a, e] * X[t, a] for a in Arcs)) for t in 
                    T[1:] for e in Edges}
    # conserve arc flow on nodes
    consFlow = {(t, n): mip.addConstr(quicksum(E[a, n] * X[t, a] for a in Arcs) == 
                          quicksum(S[a, n]* X[t + 1, a] for a in Arcs)) for t in 
                            T[1: -1] for n in Nodes}
    # initially, no edges have been searched
    initY = {e: mip.addConstr(Y[0, e] == 0) for e in Edges}
    # limit y so that every edge is searched by T
    everyEdgeSearched = {e: mip.addConstr(Y[maxTime, e] == 1) for e in Edges}
    
    # must use at least 1 leaf if uniform
    if probType == UNIFORM and len(leafArcs) != 0:
        mip.addConstr(quicksum(X[1, l] for l in leafArcs) >= 1)
    elif probType == NON_UNIFORM:
        maxEdges = getMaxEdges(Edges, prob)
        XE = mip.addVar(vtype = GRB.INTEGER)
        mip.addConstr(XE == quicksum(X[1, a] for a in Arcs if getEdge(a, Edges, O)))
        XE.BranchPriority = 15
    
    #Set the maximum time to 900 seconds
    mip.setParam('TimeLimit', 900.0)
    mip.setParam('OutputFlag', 0)
    mip.optimize()
    time = mip.Runtime

    return mip, graph, time

#seed = 568739401
#seed = 829083004
#seed = 2003701112
#mip, graph, time = MIP(NON_UNIFORM, 1, 24, 18, 48, seed = seed)

