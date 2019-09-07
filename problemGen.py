# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:08:57 2019

@author: Daniel
"""
from collections import defaultdict
from random import randint, choice, seed


# For displaying the generated networks
# If using the anaconda python environment (as recomended by gurobi) run:
#    conda install -c anaconda networkx
import networkx as nx
import matplotlib.pyplot as plot

GRID_SIZE = 10
# Display a square lattice of dimensions size (default 10)
# Can also pass a graph to display on the lattice
# This is to be used for displaying sparse graphs in the nice grid format like
# in the paper


def displayLattice(size=GRID_SIZE, graph=None):
    for i in range(size):  # Rows
        for j in range(size):  # Columns
            # Index to a node
            index = j + i*size
            left = (j-1) + i*size
            right = (j+1) + i*size
            # print("index:", index, 'left', left, 'right', right)
            exists = index in graph.keys()  # Does the node exist

            if graph is not None and exists:
                dispChar = '*'
            else:
                dispChar = ' '
            print(dispChar, end=' ')

            if not (j == size - 1) and exists and right in graph[index]:
                dispChar = '-'
            else:
                dispChar = ' '
            print(dispChar, end=' ')

        print()  # New line
        for j in range(size):
            index = j + i*size
            up = j + (i-1)*size
            down = j + (i+1)*size
            # Don't print connections after last row
            if not (i == size-1) and down in graph[index]:
                dispChar = '| '
            else:
                dispChar = '  '
            # Not on the edge of a graph
            print(dispChar, end='  ')

        print()


# Output of this can be put in here:
# https://csacademy.com/app/graph_editor/ to visualise
def displayGraph(graph):
    for k, v in graph.items():
        for i in v:
            print(k, i)


# Return the indexes of the neighbour of a point in order clockwise
# Up, right, down, left
def gridNeighbours(index, size=GRID_SIZE):
    up = index - size
    right = index + 1
    down = index + size
    left = index - 1

    if index < size:
        up = None

    if index >= size*(size-1):
        down = None

    if index % size == 0:
        left = None

    if index % size == size-1:
        right = None

    return (up, right, down, left)


def indexToXY(index, size=GRID_SIZE):
    x = index % GRID_SIZE
    y = GRID_SIZE - (index - x) // GRID_SIZE - 1
    return (x, y)


#def generateProbabilities(graph, prob):
#    numEdges = sum(len(graph[n]) for n in graph.keys())/2
#    print(numEdges)
#    if prob is None:
#        # Uniform
#        pass
    
def genProbs(graph, probType):
    pUn = None #uniform prob
    pNon = None #non-uniform prob
    #number of nodes
    numNodes = sum(n for n in graph.keys())
    #number of edges
    numEdges = sum(len(graph[n]) for n in graph.keys())/2
    if not probType:
        #gen uniformly dist. probs
        pUn = round(1/numEdges, 2)
        edgeNums = numEdges
    elif probType:
        #define number of edges st. Cl: Cm: Ch app = 1: 2: 1
        edgeFloor = numEdges//4
        edgeMod = numEdges % 4 #num edges mod 4
        #we need to split the edges into groups approximately of the ratio
        #1: 2: 1 (4 "pieces")- decide what to do with leftovers depending on 
        #how many leftovers
        edgeNums = []
        if edgeMod == 0:
            edgeNums = [edgefloor, 2 * edgefloor, edgefloor]
        elif edgeMod <= 2:
            edgeNums = [edgefloor, 2 * edgefloor + edgemod, edgefloor]
        elif edgeMod  == 3:
            edgeNums = [edgefloor + 1, 2 *edgefloor + 1, edgefloor + 1]
        else:
            print('Something Wrong')
            return -1
        #define list of probs of the form [pl, pm, ph]
        p = 1/(edgeNums[0] + 2 * edgeNums[1] + 3 * edgeNums[2])
        pNon = [p, 2 * p, 3 * p]
        pNon = [round(p, 2) for p in pNon]
    else:
        print('Invalid Graph Type')
        return -1
    return pUn, pNon, edgeNums
      
        

# Generate a network with the given number of edges and nodes
def generateNetwork(edges, nodes, probType, initSeed=None):
    # Seed the RNG
    seed(initSeed)
    a = edges
    b = nodes

    # The network / graph is rperesented with nodes as keys,
    # and the nodes they connect to as values (edges are pairs of nodes)
    graph = defaultdict(set)
    if a/b < 2:
        # Sparse network - like a manhattan network
        start = randint(0, GRID_SIZE**2 - 1)
        graph[start] = set()
        visited = [start]  # Nodes visited already
        queue = [start]  # Nodes to visit
        # Make a-1 edges
        while len(visited) < b and queue:  # Breadth first search
            currentNode = queue.pop()
            # Generate a tuple of valid neighbour indexes
            # (if a value is on the end of the grid,
            # neighbours outside the grid are None)
            validNeighbours = tuple(
                x for x in gridNeighbours(currentNode)
                if x is not None
                )
            for v in validNeighbours:
                if len(visited) >= b:
                    break
                if v not in visited:
                    graph[currentNode].add(v)
                    graph[v].add(currentNode)
                    visited.append(v)
                    queue.append(v)

        numEdges = len(visited) - 1
        for i in graph.keys():
            for j in gridNeighbours(i):
                if numEdges < a:
                    if j in graph.keys() and j not in graph[i] and j is not None:
                        graph[i].add(j)
                        graph[j].add(i)
                        numEdges += 1
    else:
#        # Dense network
        for i in range(b):
            graph[b] = set()
        edgesMade = 0
        
        while edgesMade < a:
            node1 = randint(1, b)
            node2 = randint(1, b)
            while node2 == node1:
                node2= randint(1, b)
            if node2 not in graph[node1] and node1 not in graph[node2]:
                graph[node1].add(node2)
                graph[node2].add(node1)
                edgesMade += 1
#            edgeType = randint(0, 2)
#            while generatedTypes[edgeType] == edgeNums[edgeType]:
#                edgeType = randint(0, 2)
#            generatedTypes[edgeType] += 1
            else:
                pass
    #assign probabilities to existing edges in graph
    pUn, pNon, edgeNums = genProbs(graph, probType)
    edges = genEdges(graph)
    prob = {}
    if pUn is not None:
        #uniform case
        prob[pUn] = [e for e in edges.values()]
    else:
        generatedTypes = [0, 0, 0]
        for e in edges:
            edgeType = randint(0, 2)
            while generatedTypes[edgeType] == edgeNums[edgeType]:
                edgeType = randint(0, 2)
            prob[pNon[edgeType]].append(edges(e))
            generatedTypes[edgeType] += 1
            
            
        
#
#        floorHalf = a//2  # The floor of half the edges
#        # sum of the edges of each type must equal a
#        # If floorHalf is an odd number, we need to add 1 to it to satisfy this
#        if floorHalf % 2 == 1:
#            floorHalf += 1
#        # Maximum number of each edge type (low, medium, high)
#        maxEdgeTypes = (floorHalf//2, floorHalf, floorHalf//2)
#        # Number of each type already generated
#        generatedTypes = [0, 0, 0]
#        
#        while edgesMade < a:
#            node1 = randint(1, b)
#            node2 = randint(1, b)
#            while node2 == node1:
#                node2 = randint(1, b)
#            
#            nodeType = randint(0, 2)
#
#            generatedTypes[nodeType] += 1
#
#            prob = round((nodeType+1)*0.2, 2)
#            
#            if node2 not in graph[node1] and node1 not in graph[node2]:
#                graph[node1].add(node2)
#                graph[node2].add(node1)
#                edgesMade += 1
#        print(edgesMade)
    return graph
#def genArcs(graph):
#    arcs = []
#    for i in graph.keys():
#        for j in graph[i]:
#            if j is not None:
#                arcs.append((i, j))
#    return arcs
def genEdges(graph):
    edges = {}
    e = 0
    for i in graph.keys():
        for j in graph[i]:
            if (i, j) not in edges.values() and (j, i) not in edges.values():
                edges[e] = (i, j)
                e += 1
    return edges
#
#def genNodes(graph):
#    nodes = []
#    for n in graph:
#        nodes.append(n)
#    return nodes
#
#def arcStart(a, n):
#    if a[0] == n:
#        return 1
#    else:
#        return 0
#    
#def arcEnd(a, n):
#    if a[1] == n:
#        return 1
#    else:
#        return 0
#    
#def arcOnEdge(a, e):
#    if a[0] == e[0] and a[1] == e[1]:
#        return True
#    else:
#        return False

    fig, (ax1, ax2) = plot.subplots(1, 2)

    ax1.set_title('Dense')
    ax1.set_axis_off()
    nx.draw_networkx(dense, ax=ax1)

    ax2.set_title('Sparse')
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-1, 10)

    for key in sparseInstance.keys():
        keyCoords = indexToXY(key)
        for node in sparseInstance[key]:
            nodeCoord = indexToXY(node)
            ax2.plot(
                [keyCoords[0], nodeCoord[0]],
                [keyCoords[1], nodeCoord[1]],
                'b.-'
                )

    plot.show()
