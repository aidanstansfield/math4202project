from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt


#update which edges have been explored
def updateSearch(path, unexplored, O):
    temp = []
    for e in unexplored:
        if e not in path and (e[1], e[0]) not in path:
            temp.append(e)
    return temp

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

#generate a feasible region
def genFeas(graph, prob, K, Arcs, Edges, Nodes, maxTime):
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
    start = None
    #dictionary of states
    A = {1: []}
    #dictionary of search history with t values as keys
    H = {1: []}
    #find arcs connected to leaf nodes in network
    leafArcs = genLeaf(graph)
    #initially, all edges unexplored
    initUnexp= [e for e in Edges]
    
    #split into cases to define starting situations
    #if num searchers = num leaf arcs, then the only starting situation is
    #to start in the state containing every leaf arc
    if len(leafArcs) == K:
        print('LA = K')
        start = leafArcs
        unexplored = updateSearch(start, initUnexp, O)
        A[1] = [start]
        H[1] = unexplored
    #if there are more leaf arcs than searchers, valid starting states will be
    #every combination of length K of leaf arcs
    elif len(leafArcs) > K:
        print('LA > K')
        start = list(itertools.combinations(leafArcs, K))
        for s in range(len(start)):
            unexplored = updateSearch(start[s], initUnexp, O)
            A[1].append([start[s], unexplored])
    #if less leaf arcs than searchers, add the leaf arcs to each valid state 
    #and generate all combinations of adding arcs in to fill the gap
    else:
        print('LA < K')
        #candArcs is list of non-leaf arcs (arcs we will add on the end of the
        #state)
        candArcs = []
        for a in Arcs:
            if a in leafArcs:
                pass
            else:
                candArcs.append(a)
        #add list of leaf Arcs to end of candArcs
        candArcs.append(leafArcs)
        #this will generate a list of all combinations of members of candArcs
        #we only want the ones containing the list of leaf arcs
        start = list(itertools.combinations(candArcs, K - len(leafArcs) + 1))
        
        #extract elements of start that contain leafArcs
        temp = []
        for s in range(len(start)):
            slist = []
            hasLeaf = 0
            for x in start[s]:
                if type(x) is list:
                    for a in x:
                        slist.append(a)
                    hasLeaf = 1
                else:
                    slist.append(x)
            if hasLeaf:
                temp.append(slist)
        start = temp
        
        for s in range(len(start)):
            temp = initUnexp
            unexplored = updateSearch(start[s], temp, O)
            A[1].append(start[s])
            H[1].append(unexplored)
    #initialise predecessors
    pred = {}
    for r in range(len(A[1])):
        #no predecessors for t = 1
        pred[1, r] = []
    if start is None:
        print('*********************ERROR IN FINDING STARTING NODES ***\
              **************************')

    print('Number of Root Nodes: ', len(A[1]))
    #this bit just stops the code so you can check the size of the problem
    msvcrt.getch()
    
    #generate valid successors for each arc
    succ = genSuccessors(graph, Arcs)
   
    print('Init Done')
    t = 2
    numFeas = 0
    while t < maxTime:
        print(len(A[t - 1]))
        A[t] = []
        H[t] = []
        ind = -1
        
        for r in range(len(A[t - 1])):
            #if there's more to explore
            if len(H[t - 1]) > 0:
                #only pick connected arcs
                nextStep = list(itertools.product(*[succ[a] for a in A[t - 1][r]]))
                for s in range(len(nextStep)):
                    nextStep[s] = list(nextStep[s])
                    ind += 1
                    unexplored = updateSearch(nextStep[s], H[t - 1], O)
                    pred[t, ind] = A[t - 1][r]
                    A[t].append(nextStep[s])
                    H[t].append(unexplored)
            else:
                numFeas += 1
                print('Found Feasible Sol')
            print('Done node ', r)
        print('------------------Complete t = ', t, '----------------')
        t += 1
    print('Number of Feasible Solutions: ', numFeas)
    displayGraph(graph)
    return A

g, prob, Edges, _ = generateNetwork(19, 15, 0)
Arcs = genArcs(g)
Nodes = genNodes(g)
K = 5
maxTime = math.ceil(2 * len(Edges)/K + 5)
A = genFeas(g, prob, K, Arcs, Edges, Nodes, maxTime)
