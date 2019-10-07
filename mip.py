from gurobipy import quicksum, GRB, Model
from problemGen import generateNetwork, genArcs, genNodes, displayGraph, indexToXY
from time import clock
from matplotlib import pyplot, animation
import msvcrt

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


def MIP(probType, K, numEdges, numNodes, maxTime, seed=None):

    # min time for searchers to search every arc
#    graph, p, edges = genEx(probType)
#    #sets

    M = range(0, numEdges)
    N = range(0, numNodes)
    L = range(0, 2 * numEdges)
    T = range(0, maxTime + 1)

    #data
    #gen network - p is pdf and edges is set of edges
    graph, p, edges, _ = generateNetwork(numEdges, numNodes, probType, seed)
    #    displayGraph(graph)
    S = {}
    E = {}
    O = {}
    # gen set of arcs
    arcs = genArcs(graph)
#    print(arcs)
    # gen set of nodes
    nodes = genNodes(graph)
#    print('E: ', edges)
    # define values for functions S, E and O and store in dict.
    for l in L:
        for m in M:
            if arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
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
    
    # variables
    X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
    Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
    alpha = {t: mip.addVar() for t in T}
    
    # objective
    mip.setObjective(quicksum((alpha[t-1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)
    
    # constraints
    
    # num arcs searched can't exceed num searchers- implicitly defines capacity
    searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T[1:]}
    # capacity of flow on each arc
    #    lowerBound = {(t, l): mip.addConstr(X[t, l] >= 0) for t in T[1:] for l in L}
    # lower bound on flow
    # define alpha as in paper
    defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]] * Y[t, m] for m in 
                M)) for t in T}
    # mip.addConstr(alpha[0] == 1)
    # update search info after every time step
    updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
                    quicksum(O[l, m] * X[t, l] for l in L)) for t in 
                    T[1:] for m in M}
    # conserve arc flow on nodes
    consFlow = {(t, n): mip.addConstr(quicksum(E[l, n] * X[t, l] for l in L) == 
                          quicksum(S[l, n]* X[t + 1, l] for l in L)) for t in 
                            T[1: -1] for n in N}
    # initially, no edges have been searched
    initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
    # limit y so that every edge is searched by T
    {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}
    
    
    
#     Changinge Branch priority based on aggregation
#    XT = mip.addVar(vtype=GRB.INTEGER)
#    mip.addConstr(XT==quicksum(X.values()))
#    XT.BranchPriority = 10
#    mip.setParam('GURO_PAR_MINBPFORBID', 1)
    
    mip.optimize()
    time = mip.Runtime

    state = {
        "X": X,
        "Y": Y,
        "T": T,
        "L": L,
        "maxTime": maxTime
    }
    
#    visualiseStrategy(state, arcs, graph)

    return mip, graph, time#, X, p, edges, O, arcs, L, M, T, alpha



mip, graph, _ = MIP(UNIFORM, 1, 50, 14, 50)


# 3358408176512599648

# SLow:
# 4772197386045408510
# 6726931912431499781
# 2600597230088613908

#
#ob = [0 for i in range(10)]
#time = [0 for i in range(10)]
#gs = {}
#for i in range(10):
#    ob[i], gs[i], time[i] = MIP(1, 1, 19, 15, 30)
#avET = sum(ob)/len(ob)
