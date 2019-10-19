from gurobipy import quicksum, GRB, Model, max_, min_
from problemGen import generateNetwork, genArcs, genNodes, genEdges, displayGraph, indexToXY, generateProbabilities, genLeaf
from matplotlib import pyplot, animation
from math import inf
import math
import warnings

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
    numEdges = len(state["E"])
    numNodes = len(state["N"])
    if numEdges<= (2*numNodes - math.ceil(2*math.sqrt(numNodes))):
        fig = pyplot.figure()
        
        def init():
            #Ignore matplotlib internal deprecation warning
            warnings.filterwarnings("ignore")
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
            pyplot.title('Sparse: t=' + str(t))
            if t == 0:
                return
    
            for a in state["A"]:
                if t > 1 and state["X"][t-1, a].x > 0.9:
                    XCoords = [indexToXY(a[0])[0], indexToXY(a[1])[0]]
                    YCoords = [indexToXY(a[0])[1], indexToXY(a[1])[1]]
                    pyplot.plot(XCoords, YCoords, 'g-')
    
                if state["X"][t, a].x > 0.9:
                    XCoords = [indexToXY(a[0])[0], indexToXY(a[1])[0]]
                    YCoords = [indexToXY(a[0])[1], indexToXY(a[1])[1]]
                    pyplot.plot(XCoords, YCoords, 'r-',)
    
        # Must be assigned to a variable or the animation doesn't play
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                frames=state["maxTime"]+1, interval = 750)
    
        pyplot.show()
    else:
        print("Cannot visualise strategy for dense graphs. Graph will be displayed without strategy.")
        displayGraph(graph)


def genSuccessors(graph, Arcs):
    successors = {}
    for a in Arcs:
        suclist = []
        for x in Arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

def MIP(probType, K, numEdges, numNodes, maxTime, graph = None, edges = None, p = None, seed=None):
#    #sets
    if graph is None:
        graph, prob, Edges, _ = generateNetwork(numEdges, numNodes, probType, seed)
    if prob is None:
        prob, Edges = generateProbabilities(graph, probType)
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
    
    #set of leaf arcs
    leafArcs = genLeaf(graph)
    
    #dict containing all the arcs thet start at the ending node of each arc
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

    #model
    mip = Model("Model searchers searching for a randomly distributed immobile" \
                "target on a unit network")
    
    #variables
    
    #X[t, a] = 1 if we traverse arc a at time t, 0 otherwise
    X = {(t, a): mip.addVar(vtype = GRB.BINARY) for t in T[1:] for a in Arcs}
    
    """*******************BRANCH PRIORITIES********************************"""
#    for (t, a) in X:
#        if a in leafArcs and t == 1:
#            for x in arcCon[a]:
#                #if a shares a node with another leaf arc
#                if x in leafArcs:
#                    #higher branch priority than many other variables yet lower
#                    #than the variables for leaf arcs that aren't next to 
#                    #another leaf
#                    X[t, a].BranchPriority = 15
#                    break
#            else:
#                #higher branch priority because these leaf arcs will have
#                #bigger impact on mip gap
#                X[t, a].BranchPriority = 25
#        else:
#            #for earlier times, these have highest branch priority, but as time
#            #goes on their contribution gets less significant
#            X[t, a].BranchPriority = max(1, 2 * maxTime - 5 * t)
            
    """#################### MOD: MAKE Y CONTINUOUS ###########################"""
    
    #Y[t, e] = 1 if we have searched edge e by time t, 0 otherwise
    Y = {(t, e): mip.addVar(vtype = GRB.BINARY) for t in T for e in Edges}
    
#    Y is a continuous variable indicating whether an edge has been searched at
    #time t or not- it will take maximal value of 1 when edge has been searched
#    Y = {(t, e): mip.addVar(ub=1) for t in T for e in Edges}
    
    """*******************BRANCH PRIORITIES********************************"""
    #YT[t] is the number of new edges we explore at time t- only defined for
    #early time points because these are the decisions with the most impact 
    #when branched upon
#    YT = {t: mip.addVar() for t in range(1, 4)}
#    defYT = {t: mip.addConstr(quicksum(Y[t, e] for e in Edges) == YT[t]) 
#            for t in YT}
#    for t in YT: 
#        #branch priority starts at 10 (higher than many other variables but
#        #lower than others we are setting branch priorities for) and decreases 
#        #with t
#        YT[t].BranchPriority = 10 - 2 * t
    
    #alpha[t] is the probability that the target won't be found by maxTime
    """##################### MOD: COMMENT ALPHA WHEN Y IS CONTINUOUS ###########"""
    alpha = {t: mip.addVar() for t in T}
    
    # objective
    """##################### MOD: COMMENT ALPHA OBJ WHEN Y CONTINUOUS ##########"""
    mip.setObjective(quicksum((alpha[t - 1] + alpha[t])/2 for t in T[1:]),
                     GRB.MINIMIZE)
#    mip.setObjective(maxTime + 1/2 - quicksum(Y[t, e] * prob[e] for e in Edges
#                    for t in T[1:]), GRB.MINIMIZE)
    
    # constraints
    
    # num arcs searched can't exceed num searchers- implicitly defines capacity
    searcherLim = {t: mip.addConstr(quicksum(X[t, a] for a in Arcs) <= K) 
                    for t in T[1:]}
    
    """################### MOD: COMMENT ALPHA CONSTR WHEN Y CONT ##########"""
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
    
    """##################### MOD: UNCOMMENT WHEN Y CONTINUOUS ##################"""
    #upper bound on Y when Y is continuous
#    yUB = {(t, e): mip.addConstr(Y[t, e] <= 1) for (t, e) in Y}
    
    """---------------------MOD: MUST SEARCH LEAF ARC AT TIME 1 ------------"""
    # must use at least 1 leaf if uniform
    if probType == UNIFORM and len(leafArcs) != 0:
        mip.addConstr(quicksum(X[1, l] for l in leafArcs) >= 1)
    
    """*******************BRANCH PRIORITIES********************************"""
    #if non-uniform, we preferably want to start by searching edges with high
    #probs first- branch on this
#    if probType == NON_UNIFORM:
#        maxEdges = getMaxEdges(Edges, prob)
#        XE = mip.addVar(vtype = GRB.INTEGER)
#        defXe = mip.addConstr(XE == quicksum(X[1, a] for a in Arcs if 
#                            getEdge(a, Edges, O)))
#        XE.BranchPriority = 15
    #Parameter Adjustments
    
    #Set the maximum time to 900 seconds
    mip.setParam('TimeLimit', 900.0)
    
    """........................SET METHOD = 2............................."""
    #Run barrier algorithm for mip root node
#    mip.setParam("Method",2)
    
    #set optimality gap to 0
    mip.setParam('MipGap', 0)
    
    mip.optimize()
    
    #Display Search Path
    
    state = {
        "X": X,
        "Y": Y,
        "T": T,
        "E": Edges,
        "L": leafArcs,
        "A": Arcs,
        "N": Nodes,
        "maxTime": maxTime,
        "seed": seed
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
    MIP(probType=UNIFORM,K=2,numEdges=19, numNodes=15, maxTime=11, seed=8951724032167401572)
#    MIP(probType=UNIFORM,K=19,numEdges=30, numNodes=15, maxTime=50, seed=8951724032167401572)
#    if True:
#        # run mip
#        seed = 2003701112
#        K = 1
#        MIP(probType=UNIFORM,K=1,numEdges=19, numNodes=15, maxTime=38, seed = seed)
#    else:
#        numEdges = 24
#        numNodes = 18
#        seed = 748345644471475368
#        #seed = 5628902086360812845
#        K = 1
#        maxTime = 2 * numEdges
#        probType = 0
#        M = range(0, numEdges)
#        N = range(0, numNodes)
#        L = range(0, 2 * numEdges)
#        T = range(0, maxTime + 1)
#        #data
#        #gen network - p is pdf and edges is set of edges
#        graph, p, edges, _ = generateNetwork(numEdges, numNodes, probType, seed)
#        S = {}
#        E = {}
#        O = {}
#        # gen set of arcs
#        arcs = genArcs(graph)
#        # gen set of nodes
#        nodes = genNodes(graph)
#        leafs = genLeaf(graph)
#        leafIndices = [arcs.index(leaf) for leaf in leafs]
#        for l in L:
#            for m in M:
#                if arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
#                    O[l, m] = 1
#                else:
#                    O[l, m] = 0
#            for n in N:
#                if arcs[l][0] == nodes[n]:
#                    S[l, n] = 1
#                    E[l, n] = 0
#                elif arcs[l][1] == nodes[n]:
#                    E[l, n] = 1
#                    S[l, n] = 0
#                else:
#                    S[l, n] = 0
#                    E[l, n] = 0
#        
#        mip = Model("Model searchers searching for a randomly distributed immobile" \
#                    "target on a unit network")
#        # variables
#        X = {(t, l): mip.addVar(vtype=GRB.BINARY) for t in T[1:] for l in L}
#        Y = {(t, m): mip.addVar(vtype=GRB.BINARY) for t in T for m in M}
#        alpha = {t: mip.addVar() for t in T}
#        
#        # objective
#        mip.setObjective(quicksum((alpha[t-1]+alpha[t])/2 for t in T[1:]), GRB.MINIMIZE)
#        
#        # constraints
#        # num arcs searched can't exceed num searchers- implicitly defines capacity
#        searcherLim = {t: mip.addConstr(quicksum(X[t, l] for l in L) <= K) for t in T[1:]}
#
#        # define alpha as in paper
#        defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(p[edges[m]] * Y[t, m] for m in 
#                    M)) for t in T}
#        # update search info after every time step
#        updateSearch = {(t, m): mip.addConstr(Y[t, m] <= Y[t - 1, m] + 
#                        quicksum(O[l, m] * X[t, l] for l in L)) for t in 
#                        T[1:] for m in M}
#        # conserve arc flow on nodes
#        consFlow = {(t, n): mip.addConstr(quicksum(E[l, n] * X[t, l] for l in L) == 
#                              quicksum(S[l, n]* X[t + 1, l] for l in L)) for t in 
#                                T[1: -1] for n in N}
#        # initially, no edges have been searched
#        initY = {m: mip.addConstr(Y[0, m] == 0) for m in M}
#        # limit y so that every edge is searched by T
#        mustFindTarget = {m: mip.addConstr(Y[maxTime, m] == 1) for m in M}
#        
#        # must use at least 1 leaf if uniform
#        if probType == UNIFORM and len(leafs) != 0:
#            startAtLeaf = mip.addConstr(quicksum(X[1, l] for l in leafIndices) >= 1)
#        """ NEEDS BENCHMARK
#        # add a constraint that only permits
#        # from what i've tested, when adding to the original MIP, it can speed things up
#        # however, when used with the other leaf node constraint (which speeds things up a lot),
#        # it seems to perform worse than just using the other leaf node constraint
#                
#        leafConstraints = {(t, l): mip.addConstr(X[t, l] <= 1 - 
#                           quicksum(Y[t - 1 , m] for m in M if O[l, m]))
#                            for t in T[1:] for l in L if arcs[l][::-1] in leafs}
#        """
#        """ NEEDS BENCHMARK
#        # add a constraint that only permits an arc to be searched if it either
#        # a) puts us closer to an unsearched edge
#        # b) all edges have already been searched
#        
#        distances = shortest_paths(graph)
#        closer = {}
#        further = {}
#        for l in L:
#            closer[l] = []
#            further[l] = []
#            for m in M:
#                if distance(arcs[l][1], edges[m], distances) < distance(arcs[l][0], edges[m], distances):
#                    closer[l].append(m)
#                elif arcs[l] == edges[m] or arcs[l] == edges[m][::-1]:
#                    closer[l].append(m)
#                else:
#                    further[l].append(m)
#        
#        numEdgesUnsearched = {t: mip.addVar(vtype=GRB.INTEGER) for t in T[1:]}
#        someEdgesUnsearched = {t: mip.addVar(vtype=GRB.BINARY) for t in T[1:]}
#        
#        for t in T[1:]:
#            mip.addConstr(numEdgesUnsearched[t] == quicksum(1-Y[t, m] for m in M))
#            mip.addConstr(someEdgesUnsearched[t] == min_(1, numEdgesUnsearched[t]))
#        
#        
#        moveTowardsUnsearched = {(t, l): mip.addConstr(X[t, l] <= 
#                                quicksum(1 - Y[t - 1, m] for m in closer[l]) +
#                                1 - someEdgesUnsearched[t - 1])
#                                for t in T[2:] for l in L}
#        """
#        mip.setParam('TimeLimit', 1000.0)
#        mip.setParam("Method",2)
#        mip.optimize()
#        state = {
#            "X": X,
#            "Y": Y,
#            "T": T,
#            "L": L,
#            "maxTime": maxTime
#        }
#        visualiseStrategy(state, arcs, graph)
# SLow:
# 4772197386045408510
# 6726931912431499781
# 2600597230088613908