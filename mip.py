from gurobipy import quicksum, GRB, Model, max_, min_
from problemGen import generateNetwork, genArcs, genNodes, genEdges, displayGraph, indexToXY, generateProbabilities, genLeaf
from time import clock
from matplotlib import pyplot, animation
import msvcrt
from math import inf

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


def visualiseStrategy(state, graph):
    arcs = state["A"]
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
#        if sum(i.x for i in state["X"].values()) == state[""]


    # Must be assigned to a variable or the animation doesn't play
    anim = animation.FuncAnimation(fig, animate, init_func=init,
            frames=state["maxTime"], interval = 750, repeat=False)

    pyplot.show()


def MIP(probType, K, numEdges, numNodes, maxTime, graph = None, edges = None, p = None, seed=None):

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
    if p is None:
        p, edges = generateProbabilities(graph, probType)
    if edges is None:
        edges = genEdges(graph)
    S = {}
    E = {}
    O = {}
    # gen set of arcs
    arcs = genArcs(graph)
    # gen set of nodes
    nodes = genNodes(graph)
    leafs = genLeaf(graph)
    leafIndices = [arcs.index(leaf) for leaf in leafs]
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

    # define alpha as in paper
    defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]] * Y[t, m] for m in 
                M)) for t in T}
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
    everyEdgeSearched = {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}
    mustFindTarget = {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}

    # must use at least 1 leaf if uniform
    if probType == UNIFORM and len(leafs) != 0:
        mip.addConstr(quicksum(X[1, arcs.index(leaf)] for leaf in leafs) >= 1)


#    mip.setParam('OutputFlag', 0)

    startAtLeaf = mip.addConstr(quicksum(X[1, l] for l in leafIndices) >= 1)

    #mip.setParam('OutputFlag', 0)
    #Set the maximum time to 900 seconds
    mip.setParam('TimeLimit', 900.0)
    mip.setParam('MipGap', 0)
    mip.setParam('Method', 2)
    mip.optimize()
    state = {
        "X": X,
        "Y": Y,
        "T": T,
        "L": L,
        "A": arcs,
        "maxTime": maxTime
    }
    visualiseStrategy(state, graph)

    return mip, graph, state

# Floyd-Warshall algorithm
def shortest_paths(graph):
    V = graph.keys()
    dist = {}
    for v in V:
        dist[v] = {}
        for v2 in V:
            dist[v][v2] = inf
    for v in V:
        dist[v][v] = 0
        for v2 in graph[v]:
            dist[v][v2] = 1
    for k in V:
        for i in V:
            for j in V:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

"""
from_node - starting node (note it is the actual node, not an index in L)
to_edge - the edge that we want to know the distance to
distances - a list of distances from every node to every node
returns the shortest distance to that edge (will be one end of that edge)
"""
def distance(from_node, to_edge, distances):
    return min(distances[from_node][to_edge[i]] for i in range(2))


if __name__ == "__main__":
    if False:
        # run mip
        MIP(probType=UNIFORM,K=1,numEdges=19, numNodes=15, maxTime=38)
    else:
        numEdges = 19
        numNodes = 15
        seed = 748345644471475368
        #seed = 5628902086360812845
        K = 1
        maxTime = 2 * numEdges
        probType = 0
        M = range(0, numEdges)
        N = range(0, numNodes)
        L = range(0, 2 * numEdges)
        T = range(0, maxTime + 1)
        #data
        #gen network - p is pdf and edges is set of edges
        graph, p, edges, _ = generateNetwork(numEdges, numNodes, probType, seed)
        S = {}
        E = {}
        O = {}
        # gen set of arcs
        arcs = genArcs(graph)
        # gen set of nodes
        nodes = genNodes(graph)
        leafs = genLeaf(graph)
        leafIndices = [arcs.index(leaf) for leaf in leafs]
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

        # define alpha as in paper
        defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]] * Y[t, m] for m in 
                    M)) for t in T}
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
        mustFindTarget = {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}
        
        # must use at least 1 leaf if uniform
        if probType == UNIFORM and len(leafs) != 0:
            startAtLeaf = mip.addConstr(quicksum(X[1, l] for l in leafIndices) >= 1)
        
        """ NEEDS BENCHMARK
        # add a constraint that only permits
        # from what i've tested, when adding to the original MIP, it can speed things up
        # however, when used with the other leaf node constraint (which speeds things up a lot),
        # it seems to perform worse than just using the other leaf node constraint
                
        leafConstraints = {(t, l): mip.addConstr(X[t, l] <= 1 - 
                           quicksum(Y[t - 1 , m] for m in M if O[l, m]))
                            for t in T[1:] for l in L if arcs[l][::-1] in leafs}
        """
        """ NEEDS BENCHMARK
        # add a constraint that only permits an arc to be searched if it either
        # a) puts us closer to an unsearched edge
        # b) all edges have already been searched
        
        distances = shortest_paths(graph)
        closer = {}
        further = {}
        for l in L:
            closer[l] = []
            further[l] = []
            for m in M:
                if distance(arcs[l][1], edges[m], distances) < distance(arcs[l][0], edges[m], distances):
                    closer[l].append(m)
                elif arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
                    closer[l].append(m)
                else:
                    further[l].append(m)
        
        numEdgesUnsearched = {t: mip.addVar(vtype=GRB.INTEGER) for t in T[1:]}
        someEdgesUnsearched = {t: mip.addVar(vtype=GRB.BINARY) for t in T[1:]}
        
        for t in T[1:]:
            mip.addConstr(numEdgesUnsearched[t] == quicksum(1-Y[t, m] for m in M))
            mip.addConstr(someEdgesUnsearched[t] == min_(1, numEdgesUnsearched[t]))
        
        
        moveTowardsUnsearched = {(t, l): mip.addConstr(X[t, l] <= 
                                quicksum(1 - Y[t - 1, m] for m in closer[l]) +
                                1 - someEdgesUnsearched[t - 1])
                                for t in T[2:] for l in L}
        """
        mip.setParam('TimeLimit', 1000.0)
        mip.setParam("Method",2)
        mip.optimize()
        state = {
            "X": X,
            "Y": Y,
            "T": T,
            "L": L,
            "maxTime": maxTime
        }
        visualiseStrategy(state, arcs, graph)

# SLow:
# 4772197386045408510
# 6726931912431499781
# 2600597230088613908