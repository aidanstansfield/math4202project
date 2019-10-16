from gurobipy import *
from mip import *
from problemGen import *
import itertools
from collections import defaultdict
import msvcrt
import time
import math
import random

#some lower bound
EPS = 0.001

a = 30
b = 24
probType = 0
graph, prob, Edges, seed = generateNetwork(a, b, probType)

Arcs = genArcs(graph)
Nodes = genNodes(graph)
numSearchers = 4
K = range(numSearchers)
maxTime = 50

#solve MIP for graph
#mip, _, time = MIP(probType, numSearchers, a, b, maxTime, graph, Edges, prob)
#print('**************MIP OBJECTIVE VAL: ', mip.objVal, '*********************')

#distance (represented as units in the horizontal and vertical in a tuple)
#between two arcs
dist = {(a1, a2): (abs(indexToXY(a1[1])[0] - indexToXY(a2[0])[0]), 
                    abs(indexToXY(a1[1])[1] - indexToXY(a2[0])[1]))
                    for a1 in Arcs for a2 in Arcs}

#initialise the search history for the start and end path segments
initH = ()
finalH = [e for e in Edges]

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
            
#function takes set of arcs and the graph and returns dictionary with arcs
#as keys and values as list of arcs that they key arc can connect to
def genSuccessors(graph, arcs):
    successors = {}
    for a in arcs:
        suclist = []
        for x in arcs:
            if a[1] == x[0]:
                suclist.append(x)
        successors[a] = suclist
    return successors

#set of all arcConnections
arcCon = genSuccessors(graph, Arcs)

def canConnect(p1, p2):
    for i in K:
        if p2[i] not in arcCon[p1[i]]:
            return False
    return True

#returns the edge containing the arc a
def getEdge(a):
    for e in Edges:
        if O[a, e]:
            return e
        
#return set of arcs with specific density (number of arcs it connects to)
def getDense(Arcs):
    cands = []
    dense = 3
    for a in Arcs:
        if len(arcCon[a]) <= dense:
            cands.append(a)
    return cands

#generate a set of candidate starting points- arcs that are searched at time 1
leafArcs = genLeaf(graph)
startChoices = [l for l in leafArcs]
for i in range(numSearchers - len(leafArcs) + 1):
    newArc = random.choice(Arcs)
    while newArc in startChoices:
        newArc = random.choice(Arcs)
    startChoices.append(newArc)
startStates = list(itertools.combinations(startChoices, numSearchers))

#generate a set of candidate ending points- only a few of these to reduce
#dimensionality
dense = getDense(Arcs)
endChoices = list(itertools.combinations(dense, numSearchers))
endStates = []
for i in range(3):
    newState = random.choice(endChoices)
    while newState in endStates:
        newState = random.choice(endChoices)
    endStates.append(newState)
    
#function updates the search history (moving forward) for a path
def updateHist(hist, searched):
#    print('Searched: ', searched)
    temp = list(hist)
    for s in searched: 
        if getEdge(s) not in temp:
            temp.append(getEdge(s))
    return tuple(temp)

#updates search history of a path moving backwards in time
def revHist(h, prevPath):
    temp = list(h)
    for a in prevPath:
        #if double ups for edges in that path
        if getEdge(a) not in temp:
            temp.remove(getEdge(a))
    return tuple(temp)

#successors
succ = defaultdict(list)
#predecessors
pred = defaultdict(list)
#starting segments
startSeg = defaultdict(list)
#ending segments
endSeg = defaultdict(list)

#the time value we generate each direction up to 
tVal = math.floor(len(Edges)/numSearchers * 1/2)

#define successors for start states
print('------------------Generating Starting states------------------')
startSeg[0] = [(s, updateHist(initH, s)) for s in startStates]
for s in startSeg[0]:
    succ[1, s[0]] = list(itertools.product(*[[x for x in arcCon[a]]for a in s[0]]))
    
print('Num States: ', len(startSeg[0]))
print('t = 0')
for t in range(1, tVal + 1):
    for p in startSeg[t - 1]:
        if t == 1:
            print('s: ', len(succ[t, p[0]]))
        for s in succ[t, p[0]]:
            startSeg[t].append((s, updateHist(p[1], s)))
            succ[t + 1, s] = list(itertools.product(*[[x for x in arcCon[a] if len(arcCon[x]) <= 3 ]
                            for a in s]))
    print('t = ', t)
    print('Num States: ', len(startSeg[t]))

#generate an ending path segments
endSeg = {t: [] for t in range(tVal + 1)}
endSeg[0] = [(x, revHist(finalH, x)) for x in endStates]
#this dict stores successors for end states- convenient for later stages
print('------------------Generating Ending states------------------')
print('t = 0')
print('Num States: ', len(endSeg[0]))

for t in range(1, tVal + 1):
    print('t: ', t)
    for p in endSeg[t - 1]:
        arcOptions = []
        for a in p[0]:
            conList = [x for x in Arcs if a in arcCon[x] if len(arcCon[x])]
            arcOptions.append(conList)
        pred[t - 1, p[0]] = list(itertools.product(*arcOptions))
        for x in pred[t - 1, p[0]]:
#            backSucc[t, x].append(p[0])
            endSeg[t].append((x, revHist(p[1], x)))
    print('Num States: ', len(endSeg[t]))

#does a path segment contain an edge
def edgeInPathSeg(seg, e):
    for p in seg:
        for a in p[1]:
            if getEdge(a) == e:
                return True
    return False

#extend startSegs until we find paths that connect to paths in endSegs and 
#all remaining unexplored edges are in the end path
print('----------')
print('tVal = ', tVal)
print('----------')

t = tVal + 1
done = False
#msvcrt.getch()
print('------------------Joining Start and Ending Segments------------------')
while not done:
    for p in startSeg[t - 1]:
#        print('succ: ', succ[t, p[0]])
        for s in succ[t, p[0]]:
            newState = (s, updateHist(p[1], s))
            succ[t + 1, newState[0]] = list(itertools.product(*[[x for x in arcCon[a]]
                        for a in newState[0]]))
#            print('lensuc = ', len(succ[t + 1, newState[0]]))
            startSeg[t].append(newState)
            remEdges = [e for e in Edges if e not in newState[1]
                        and not edgeInPathSeg(endSeg[tVal], e)]
#            print('nedges = ', len(remEdges))
            if len(remEdges) == 0:
                done = True
                #value we finish at
                tFin = t
                
    print('t = ', t)
    print('Num States: ', len(startSeg[t]))
    t += 1
    
print('')
print('tFin = ', tFin)
initS = defaultdict(list)

#join all generated states in one set ordered by time- filter out all states
#that don't lead to a feasible ending time
#start by adding end states to the end of the set
print('------------------Filtering States------------------')
initS[tFin + tVal + 1] = endSeg[0]
for t in range(1, tVal + 1):
    initS[tFin + tVal + 1 - t] = endSeg[t]
    for p in endSeg[t - 1]:
        for q in pred[t - 1, p[0]]:
            succ[tFin + tVal + 1 - t, p[0]].append(q)
    print('t = : ', tFin + tVal + 1 - t)
    print('Num States: ', len(initS[tFin + tVal + 1 - t]))
    

for t in range(tFin + 1):
    print('t = ', tFin - t)
    for p in startSeg[tFin - t]:
        for q in initS[tFin - t + 1]:
            if q[0] in succ[tFin - t + 1, p[0]]:
                initS[tFin - t].append(p)
                break
    print('Num states: ', len(initS[tFin - t]))

#max ending time for search
maxT = tFin + tVal + 1

#final set of states
S = {t: [] for t in range(maxT)}
#filter out all states that don't connect to a state in a previous time 
print('------------------Final State Adjustments------------------')
S[0] = initS[0]
for t in range(1, maxT):
    for p in initS[t]:
        for q in S[t - 1]:
            if p[0] in succ[t, q[0]]:
                S[t].append(p)
                break
        else:
            continue
    print('t = ', t)
    print('Num States: ', len(S[t]))
    
#set of time points
T = range(maxT)

#cost of taking path segment p at time t
def Cost(p, t):
    alpha = 1
    for e in p[1]:
        alpha -= prob[e]
    return alpha

#Restricted Master Problem
RMP = Model('Search Paths')

#Z = 1 if we choose segment p at time t
Z = {(p, t): RMP.addVar(vtype = GRB.BINARY) for t in T for p in S[t]}

#objective
RMP.setObjective(quicksum(Cost(p, t) * Z[p, t] for (p, t) in Z), GRB.MINIMIZE)

#constraints
print('constr1')
onePathPerTime = {t: RMP.addConstr(quicksum(Z[p, t] for p in S[t]) >= 1)
                    for t in T}

#not sure about this one: gives infeasible sol

#print('constr2')
#consFlow = {(p, t): RMP.addConstr(quicksum(Z[q, t + 1] for q in S[t + 1]
#                        if q[0] in succ[t + 1, p[0]]) <= Z[p, t])
#                for t in T[:-1] for p in S[t]}

RMP.optimize()

#print out objective values for both models as well as run times
#print('MIP: ', mip.objVal, time)
#add 1.5 to account for 0.5 factor of alpha[0] in MIP and 1 unit of time to 
#reach end
print('RMP: ', RMP.objVal + 1.5, RMP.Runtime)

#display graph used
displayGraph(graph)