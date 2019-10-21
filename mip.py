from gurobipy import quicksum, GRB, Model, min_
from problemGen import *
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
            # if edges share a node
            if edges[m1][0] == edges[m2][0] or edges[m1][0] == edges[m2][1] or\
            edges[m1][1] == edges[m2][0] or edges[m1][1] == edges[m2][1]:
                neighbours[edges[m1]].append(edges[m2])
    return neighbours


def visualiseStrategy(state, graph):
    """ Helper function to visualise the final search strategy """
    numEdges = len(state["E"])
    numNodes = len(state["N"])
    if numEdges<= (2*numNodes - math.ceil(2*math.sqrt(numNodes))):
            fig = pyplot.figure()
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
                frames=state["maxTime"], interval = 750)

        pyplot.show()
    else:
        print("Visualisation of search strategy is currently not spported for dense instances.\nThe instance will be displayed alone.")
        displayGraph(graph)


def genSuccessors(graph, Arcs):
    """ Generate a map of arcs to successor arcs """
    successors = {}
    for a in Arcs:
        suclist = []
        for x in Arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

def getMaxEdges(Edges, prob):
    """ Get edges that have the highest probability """
    maxProb = max(prob.values())
    maxEdges = []
    for e in Edges:
        if prob[e] == maxProb:
            maxEdges.append(e)
    return maxEdges

def getEdge(a, Edges, O):
    """ returns the edge containing the arc a """
    for e in Edges:
        if O[a, e]:
            return e

def MIP(probType, K, numEdges, numNodes, maxTime, graph=None, Edges=None, 
        prob=None, seed=None, improvements=None):
    if improvements == None:
        improvements = {
            "start_at_leaf_constraint": False,
            "start_at_leaf_BP": False,
            "start_at_leaf_hint": False,
            "dont_visit_searched_leaves": False,
            "travel_towards_unsearched": False,
            "barrier_log": False,
            "Y_cts": False,
            "early_X_BP": False,
            "Y_BP": False,
            "high_prob_edges_BP": False
        }
    # sets
    if graph is None:
        graph, prob, Edges, _ = generateNetwork(numEdges, numNodes, probType, 
                                                seed)
    if prob is None:
        prob, Edges = generateProbabilities(graph, probType)
    if Edges is None:
        Edges = genEdges(graph)
    
    # gen set of arcs
    Arcs = genArcs(graph)
    # gen set of nodes
    Nodes = genNodes(graph)
    
    # set of leaf arcs
    leafArcs = genLeaf(graph)
    
    # dict containing all the arcs that start at the ending node of each arc
    arcCon = genSuccessors(graph, Arcs)
    
    T = range(maxTime + 1)
    S = {} # whether arc a begins at node n
    E = {} # whether arc a ends at node n
    O = {} # whether arc a is on edge e
    
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

    # model
    mip = Model("Model searchers searching for a randomly distributed immobile" \
                "target on a unit network")
    
    # variables
    
    # X[t, a] = 1 if we traverse arc a at time t, 0 otherwise
    X = {(t, a): mip.addVar(vtype = GRB.BINARY) for t in T[1:] for a in Arcs}
    
    if improvements['start_at_leaf_BP']:
        for (t, a) in X:
            if a in leafArcs and t == 1:
                for x in arcCon[a]:
                    # if a shares a node with another leaf arc
                    if x in leafArcs:
                        # higher branch priority than many other variables yet lower
                        # than the variables for leaf arcs that aren't next to 
                        # another leaf
                        X[t, a].BranchPriority = 15
                    else:
                        # higher branch priority because these leaf arcs will have
                        # bigger impact on mip gap
                        X[t, a].BranchPriority = 25
    
    if improvements['early_X_BP']:
        for (t, a) in X:
            # for earlier times, these have highest branch priority, but as time
            # goes on their contribution gets less significant
            X[t, a].BranchPriority = max(1, 2 * maxTime - 5 * t)
    
    # constraints
    
    if improvements['Y_cts']:
        """ This improvement simplifies the MIP by removing the alpha variables
        and allowing the Y variables to be continuous """
        # Y is a continuous variable indicating whether an edge has been searched at
        # time t or not- it will take maximal value of 1 when edge has been searched
        Y = {(t, e): mip.addVar(ub=1) for t in T for e in Edges}
        
        # objective
        mip.setObjective(maxTime + 1/2 - quicksum(Y[t, e] * prob[e] for e in Edges
                                                  for t in T[1:]), GRB.MINIMIZE)
    else:
        # Y[t, e] = 1 if we have searched edge e by time t, 0 otherwise
        Y = {(t, e): mip.addVar(vtype = GRB.BINARY) for t in T for e in Edges}
        
        #alpha[t] is the probability that the target is not found by time t
        alpha = {t: mip.addVar() for t in T}
        
        # objective
        mip.setObjective(quicksum((alpha[t - 1] + alpha[t])/2 for t in T[1:]),
                         GRB.MINIMIZE)
        
        # define alpha as in paper
        defAlpha = {t: mip.addConstr(alpha[t] == 1 - quicksum(prob[e] * Y[t, e] 
                        for e in Edges)) for t in T}
    
    if improvements['Y_BP']:
        # YT[t] is the number of new edges we explore at time t- only defined for
        # early time points because these are the decisions with the most impact 
        # when branched upon
        YT = {t: mip.addVar() for t in range(1, 4)}
        defYT = {t: mip.addConstr(quicksum(Y[t, e] for e in Edges) == YT[t]) 
                for t in YT}
        for t in YT: 
            # branch priority starts at 10 (higher than many other variables but
            # lower than others we are setting branch priorities for) and decreases 
            # with t
            YT[t].BranchPriority = 10 - 2 * t
    
    
    
    # num arcs searched can't exceed num searchers- implicitly defines capacity
    searcherLim = {t: mip.addConstr(quicksum(X[t, a] for a in Arcs) <= K) 
                    for t in T[1:]}
    
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
    
    if improvements['start_at_leaf_constraint']:
        # must have at least 1 searcher beginning at a leaf
        if probType == UNIFORM and len(leafArcs) != 0:
            mip.addConstr(quicksum(X[1, l] for l in leafArcs) >= 1)
    
    if improvements['start_at_leaf_hint'] and probType == UNIFORM:
        # hint to gurobi to start searchers at leaf nodes
        for a in leafArcs:
            X[1, a].VarHintVal = 1
    
    if improvements['high_prob_edges_BP']:
        #if non-uniform, we preferably want to start by searching edges with high
        #probs first- branch on this
        if probType == NON_UNIFORM:
            maxEdges = getMaxEdges(Edges, prob)
            XE = mip.addVar(vtype = GRB.INTEGER)
            defXe = mip.addConstr(XE == quicksum(X[1, a] for a in Arcs if 
                                getEdge(a, Edges, O)))
            XE.BranchPriority = 15
    
    if improvements['dont_visit_searched_leaves']:
        # don't search a leaf edge if it has already been searched
        leafConstraints = {(t, a): mip.addConstr(X[t, a] <= 1 - 
                           quicksum(O[a, e]*Y[t - 1 , e] for e in Edges))
                            for t in T[1:] for a in Arcs if a[::-1] in leafArcs}
    
    if improvements['travel_towards_unsearched']:
        # add a constraint that only permits an arc to be searched if it either
        # a) puts us closer to an unsearched edge
        # b) all edges have already been searched
        
        distances = shortest_paths(graph)
        closer = {}
        further = {}
        for a in Arcs:
            closer[a] = []
            further[a] = []
            for e in Edges:
                if distance(a[1], e, distances) < distance(a[0], e, distances):
                    closer[a].append(e)
                elif a == e or a == e[::-1]:
                    closer[a].append(e)
                else:
                    further[a].append(e)
        
        numEdgesUnsearched = {t: mip.addVar(vtype=GRB.INTEGER) for t in T[1:]}
        someEdgesUnsearched = {t: mip.addVar(vtype=GRB.BINARY) for t in T[1:]}
        
        for t in T[1:]:
            mip.addConstr(numEdgesUnsearched[t] == quicksum(1 - Y[t, e] for e in Edges))
            mip.addConstr(someEdgesUnsearched[t] == min_(1, numEdgesUnsearched[t]))
        
        moveTowardsUnsearched = {(t, a): mip.addConstr(X[t, a] <= 
                                quicksum(1 - Y[t - 1, e] for e in closer[a]) +
                                1 - someEdgesUnsearched[t - 1])
                                for t in T[2:] for a in Arcs}
    
    #Parameter Adjustments
    #Set the maximum time to 900 seconds
    mip.setParam('TimeLimit', 900.0)
    
    if improvements['barrier_log']:
        # Run barrier algorithm for mip root node
        mip.setParam("Method", 2)
    
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
    # if you want to visualise the strategy, uncomment the below line
    #visualiseStrategy(state, graph)
    
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

def distance(from_node, to_edge, distances):
    """
    from_node - starting node
    to_edge - the edge that we want to know the distance to
    distances - a list of distances from every node to every node
    returns the shortest distance to that edge (will be one end of that edge)
    """
    return min(distances[from_node][to_edge[i]] for i in range(2))

if __name__ == "__main__":
    """ You can run our problem in various ways. If you would like to generate
    your own graph, you can set the below parameters accordingly"""
    # The bare minimum you need to run the base MIP is as follows
    numEdges = 19
    numNodes = 15
    K = 1 # how many searchers
    probType = UNIFORM # alternatively, NON_UNIFORM
    maxTime = 2 * numEdges//K # upper bound on time
    
    # now run the MIP
    MIP(probType, K, numEdges, numNodes, maxTime)
    
    # if you know a particular seed you want to try, you can pass it in as follows
    seed = 123456789 # a None seed will generate a random seed for you
    MIP(probType, K, numEdges, numNodes, maxTime, seed=seed)
    
    # if you would like to toggle one or multiple of our improvements, simply
    # edit the below dictionary and run the MIP with it
    # NOTE: Some of these improvements did not make it into the paper, as they
    # did not offer much improvement (or made it worse) and we were page limited
    improvements = {
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": False,
        "start_at_leaf_hint": False,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": False,
        "Y_BP": False,
        "high_prob_edges_BP": False
    }
    MIP(probType,K,numEdges, numNodes, maxTime, improvements=improvements)
    
    # If you would like to read in one of the graphs we used, you can do such
    # (edit the path as necessary)
    graph, _ = readGraph('./problemInstances/M19N15/Non-Uniform/M19N15_296644637.txt')
    MIP(probType, K, numEdges, numNodes, maxTime, graph=graph)
    
    # if you would like to use the probabilities we used for that graph
    graph, prob = readGraph('./problemInstances/M19N15/Non-Uniform/M19N15_296644637.txt')
    MIP(probType, K, numEdges, numNodes, maxTime, graph=graph, prob=prob)