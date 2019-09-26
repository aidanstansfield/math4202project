from gurobipy import *
from mip import *
from problemGen import *

#def genFeas(graph, prob, maxTime, S, E, O, L, M, N, arcs):
#    canReach = {}
#    for l in L:
#        t = 1
#        canReach[l] = {1: [i for i in L if arcs[i][0] == arcs[l][1]]}
#        while t < maxTime:
#            print(t)
#            canReach[l][t + 1] = []
#            for i in L:
#                for j in canReach[l][t]:
#                    if arcs[i][0] == arcs[j][1]:
#                        canReach[l][t + 1].append(i)
#            t += 1
#        print('Done for arc ', l)
#                
#                
##    Successors = {}
#    return canReach

def belowCap(ls, K):
    if len(ls) <= K:
        return True
    else:
        return False
    
def isSearched(ls, m, O):
    searched = 0
    for l in ls:
        if O[l, m]:
            searched = 1
    return searched

def allSearched(ls, M):
    numSearched = 0
    for m in M:
        if O[]

def genFeas(graph, prob, K, maxTime, S, E, O, L, M, N, T, arcs):
#    reach = {(n, l): {t: [ls, ys, alphas]}
#    reach = {(n, l): {t: [] for t in T[1:]} for n in N for l in L if S[l, n]}
    successors = genSuccessors(graph, arcs, S, E, O, L, M, N)
    pathList = {1: l for l in L}
    
#    reach = {}
#    for l in L:
#        temp = [arcs[l]]
#        for t in T[1:]:
#            for x in L:
#                for xx in temp:
#                    if arcs[x] in successors[xx]:
                        
#    for l in L:
#        reach[arcs[l]] = {}
#        for t in T[1:]:
#            reach[arcs[l]][t] = []
#            for x in L:
#                if t == 1 and arcs[x] in successors[arcs[l]]:
#                    reach[arcs[l]][t].append(arcs[x])
#                    print('added for ', l, t, x)
#                elif t > 1:
#                    for xx in reach[arcs[l]][t - 1]:
#                        if arcs[x] in successors[xx]:
#                            reach[arcs[l]][t].append(arcs[x])
#    for k in [i for i in reach.keys()]:
#        if len(reach[k][1]) == 0:
#            reach.pop(k)
#                        if (len(reach[l, n][t]) == 0 and x in successors[l]) or\
#                            x in [successors[a] for a in reach[l, n][t - 1]]:
#                            reach[l, n][t].append(x)
#        {t: [x for x in L if all(
#                [])] 
#        for t in T[1:]} for n in N for l in L if S[l, n]}
    return successors
        
    
def genSuccessors(graph, arcs, S, E, O, L, M, N):
    successors = {}
#    for l in L:
#        successors[arcs[l]] = []
#        for i in L:
#            if arcs[l][1] == arcs[i][0]:
#                successors[arcs[l]].append(arcs[i])
    for l in L:
        suclist = []
        for x in L:
            if arcs[l][1] == arcs[x][0]:
                suclist.append(arcs[x])
        successors[arcs[l]] = suclist
    return successors

def DCG(probType, K, numEdges, numNodes, maxTime, seed = None):
    M = range(0, numEdges)
    N = range(0, numNodes)
    L = range(0, 2 * numEdges)
    T = range(0, maxTime + 1)
    
    #data
    #gen network - p is pdf and edges is set of edges
    graph, prob, edges = generateNetwork(numEdges, numNodes, probType, seed)
    print('Done Gen')
#    displayGraph(graph)
    S = {}
    E = {}
    O = {}
    #gen set of arcs
    arcs = genArcs(graph)
    #gen set of nodes
    nodes = genNodes(graph)
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
    print('Done Prelim')
    return graph, prob, maxTime, S, E, O, L, M, N, arcs

maxTime = 50
T = range(maxTime)
K = 1
g, prob, maxTime, S, E, O, L, M, N, arcs = DCG(0, 1, 19, 15, maxTime)
successors = genFeas(g, prob, K, maxTime, S, E, O, L, M, N, T, arcs)
#successors = genSuccessors(g, arcs, S, E, O, L, M, N)
displayGraph(g)
print(successors)
    
    