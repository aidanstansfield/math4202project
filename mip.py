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


<<<<<<< HEAD
def MIP(probType, K, numEdges, numNodes, maxTime, seed = None, graph = None,
        p = None, edges = None):
#    startgen = clock()
=======
def MIP(probType, K, numEdges, numNodes, maxTime, seed=None):

>>>>>>> 1128224e9cd6eb2d2a571a508981035cdd2a6d79
    # min time for searchers to search every arc
#    graph, p, edges = genEx(probType)
#    #sets

    M = range(0, numEdges)
    N = range(0, numNodes)
    L = range(0, 2 * numEdges)
    T = range(0, maxTime + 1)

    #data
    #gen network - p is pdf and edges is set of edges
    if graph is None:
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
    
    
    
    # Changinge Branch priority based on aggregation
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

#def Cost(searchPath, L, M, arcs, Edges, O, T, exp):
##    print(searchPath)
##    alpha = [0 for t in T]
##    alpha[0] = 1
##    exp = []
##    for t in T[1:]:
###        alpha[t] = alpha[t - 1]
##        print('P: ', path[t - 1])
##        for l in L:
##            if arcs[l] not in path[t - 1]:
##                continue
##            for m in M:
##                if O[l, m] == 1:
##                    exp.append(Edges[m])
##        alpha[t] = 1 - sum(prob[Edges[m]] for m in M if Edges[m] in exp)
##    print('My alpha: ')
##    for t in T:
##        print(alpha[t])
##    cost = sum((alpha[t - 1] + alpha[t])/2 for t in T[1:])
##    cost = 0
#    searchCont = []
#    for l in L:
#        if arcs[l] != searchPath[t - 1]:
#            continue
#        for m in M:
#            if O[l, m] == 1 and Edges[m] not in exp:
#                exp.append(Edges[m])
#                searchCont.append(Edges[m])
##    for t in T[1:]:
###        print('P: ', searchPath[t - 1])
###        msvcrt.getch()
##        for l in L:
##            if arcs[l] != searchPath[t - 1]:
##                continue
##            for m in M:
##                if O[l, m] == 1 and Edges[m] not in exp:
##                    exp.append(Edges[m])
##                    searchCont.append(Edges[m])
##        print(exp)
#    cost =  sum(prob[Edges[m]] for m in M if Edges[m] in searchCont)
#    print('|SC|: ', len(searchCont))
#    print('Cost: ', cost)
##        alpha[t] = 1 - sum(prob[Edges[m]] for m in M if Edges[m] in exp)
#    cost = sum((alpha[t - 1] + alpha[t])/2 for t in T[1:])
#    for k in range(K):
#        exp = []
#        for t in T[1:]:
#            print('P: ', searchPath[t - 1])
#            for l in L:
#                if arcs[l] not in searchPath[t]:
#                    continue
#                for m in M:
#                    if O[l, m] == 1:
#                        exp.append(Edges[m])
#            alpha[k, t] = 1 - sum(prob[Edges[m]] for m in M if Edges[m] in exp)
#        cost[k] = sum((alpha[k, t - 1] + alpha[k, t])/2 for t in T[1:])
#    return cost
                
#    for l in L:
#        if arcs[l] not in path:
#            continue
#        for m in M:
#            if O[l, m] == 1:
#                cost += (1 - prob[Edges[m]])/2
#                t += 1
#    return cost, exp

<<<<<<< HEAD
#def getEdge(a):
#    for e in Edges:
#        if O[a, e]:
#            return e
#K = 3
#mip, g, t, X, prob, Edges, O, Arcs, L, M, T, alpha = MIP(0, K, 19, 15, 42)
#print('E: ', Edges)
#
#print('Time taken: ', t)
#print('Obj val: ', mip.objVal)
#searchPath = {}
#path = []
#for t in T[1:]:
#    route = []
#    for l in L:
#        if X[t, l].x == 1:
#            route.append(Arcs[l])
#    path.append(route)
#print(path)
#for k in range(K):
#    searchPath[k] = []
#    for t in T[1:]:
#        searchPath[k].append(path[t - 1][k])
#    
#print('SP: ', searchPath)
##for (t, l) in X:
##    if X[t, l].x == 1:
##        path.append(Arcs[l])
##print('RP: ', path)
#exp = []
#c = [1 for t in T]
#print('MAXTIME: ', T)
#for t in T[1:]:
#    c[t] = c[t - 1]
#    print('alpha: ', alpha[t].x)
#    for k in range(K):
#        sCost, exp = Cost(searchPath[k], L, M, Arcs, Edges, O, T, exp)
#        print('sCost: ', sCost)
#        c[t] -= sCost
##    c[t] = 1 - sum(Cost(searchPath[k], L, M, Arcs, Edges, O, T, exp) for k in range(K))
#    print(c[t])
##sCost = {}
##for k in range(K):
##    sCost[k] = Cost(searchPath[k], L, M, Arcs, Edges, O, T)
##    print('Seach Cost ', k, ': ', sCost[k])
##totCost = sum(sCost[k] for k in range(K))
#totCost = sum((c[t - 1] + c[t])/2 for t in T[1:])
#print('TOTAL COST: ', totCost)
##pCost = Cost(path, L, M, Arcs, Edges, O, T)
##print('Cost: ', pCost)
#displayGraph(g)
mip, graph, _ = MIP(NON_UNIFORM, 30, 200, 150, 70, graph = None, p = None,
                    edges = None)
=======

mip, graph, _ = MIP(UNIFORM, 1, 19, 15, 25, 6726931912431499781)


# 3358408176512599648

# SLow:
# 4772197386045408510
# 6726931912431499781
# 2600597230088613908

#
>>>>>>> 1128224e9cd6eb2d2a571a508981035cdd2a6d79
#ob = [0 for i in range(10)]
#time = [0 for i in range(10)]
#gs = {}
#for i in range(10):
#    ob[i], gs[i], time[i] = MIP(1, 1, 19, 15, 30)
#avET = sum(ob)/len(ob)
