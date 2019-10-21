# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:08:57 2019
@author: Daniel
"""
from collections import defaultdict
from random import *
import sys
import os
import ast
import math
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


# Display the given graph
def displayGraph(graph):
    numEdges = getNumEdges(graph)
    numNodes = getNumNodes(graph)
    plot.figure(1, figsize=(10, 10), dpi=72)
    plot.rcParams.update({'font.size': 22})
    if numEdges > (2*numNodes - math.ceil(2*math.sqrt(numNodes))):
        plot.title('Dense')
        plot.axis()
        dense = nx.Graph(graph)
        pos = nx.circular_layout(dense)
        nx.draw_networkx(dense, pos=pos)
    else:
        plot.title('Sparse')

        for key in graph.keys():
            keyCoords = indexToXY(key, GRID_SIZE)
            for node in graph[key]:
                nodeCoord = indexToXY(node, GRID_SIZE)
                plot.plot(
                    [keyCoords[0], nodeCoord[0]],
                    [keyCoords[1], nodeCoord[1]],
                    'b.-'
                )
            plot.annotate(str(key), (keyCoords[0], keyCoords[1]))

    plot.show()


def genLeaf(graph):
    leaf = []
    for n in graph.keys():
        if len(graph[n]) == 1:
            if (n, graph[n]) not in leaf and (graph[n], n) not in leaf:
                leaf.append((n, list(graph[n])[0]))
    return leaf


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


# Convert the index of a node in a sparse graph to X and Y coordinates
def indexToXY(index, size=GRID_SIZE):

    x = index % size
    y = size - (index - x) // size - 1
    return (x, y)


# Generate probabilities of distribution probType for the given graph
def generateProbabilities(graph, probType):
    pUn = None  # uniform prob
    pNon = None  # non-uniform prob
    # number of nodes
    numNodes = getNumNodes(graph)
    # number of edges
    numEdges = getNumEdges(graph)
    if not probType:
        # gen uniformly dist. probs
        pUn = 1/numEdges
        edgeNums = numEdges
    else:
        # define number of edges st. Cl: Cm: Ch app = 1: 2: 1
        edgeFloor = numEdges//4
        edgeMod = numEdges % 4  # num edges mod 4
        # we need to split the edges into groups approximately of the ratio
        # 1: 2: 1 (4 "pieces")- decide what to do with leftovers depending on
        # how many leftovers
        edgeNums = [edgeFloor, 2 * edgeFloor, edgeFloor]
        if edgeMod == 1 or edgeMod == 2:
            edgeNums[1] += edgeMod
        elif edgeMod == 3:
            edgeNums = [i+1 for i in edgeNums]

        # define list of probs of the form [pl, pm, ph]
        p = 1/(edgeNums[0] + 2 * edgeNums[1] + 3 * edgeNums[2])
        pNon = [p, 2 * p, 3 * p]
        pNon = [p for p in pNon]

    edges = genEdges(graph)
    prob = {}
    if pUn is not None:
        # uniform case
        for e in edges:
            prob[e] = pUn
    else:
        generatedTypes = [0, 0, 0]
        for e in edges:
            edgeType = randint(0, 2)
            while generatedTypes[edgeType] == edgeNums[edgeType]:
                edgeType = randint(0, 2)
            prob[e] = pNon[edgeType]
            generatedTypes[edgeType] += 1

    return prob, edges


# Generate a dense graph with the given number of edges and nodes
def generateDense(edges, nodes, graph):
    maxValid = (nodes)*(nodes-1)//2
    if edges > maxValid:
        print("WARNING: Too Many Edges. Using maximum valid edges (",
              maxValid, ") for", nodes, "nodes.")
        edges = maxValid
    edgesMade = 0
    for i in range(1, nodes+1):
        graph[i] = set()

    while edgesMade < edges:
        node1 = randint(1, nodes)
        node2 = randint(1, nodes)
        while node2 == node1:
            node2 = randint(1, nodes)
        if (node2 not in graph[node1]) or (node1 not in graph[node2]):
            graph[node1].add(node2)
            graph[node2].add(node1)
            edgesMade += 1
    return graph


# Move leaf nodes somewhere that can make more edges
def adjustLeafNodes(graph, edgesNeeded):
    leaf = []
    for n in graph.keys():
        if len(graph[n]) == 1:
            neighboursInGraph = sum((i in graph.keys())
                                    for i in gridNeighbours(n, GRID_SIZE))

            if neighboursInGraph == 1:
                leaf.append(n)

    if (len(leaf) < edgesNeeded):
        return graph
    else:
        for i in range(edgesNeeded):
            # Remove leaf node with no in-network neighbours from graph
            neighbour = graph[leaf[i]].pop()
            del graph[leaf[i]]
            graph[neighbour].remove(leaf[i])

        return graph


# Generate a sparse network with the given number of edges and nodes
def generateSparse(edges, nodes, graph):
    # Sparse network - like a manhattan network
    # randomly choose a starting point on the grid
    start = randint(0, GRID_SIZE ** 2 - 1)
    # list of nodes we have visited (and thus will be added to the network)
    visited = [start]
    graph[start] = set()
    numEdges = 0
    while len(visited) < nodes:
        current = choice(visited)
        validNeighbours = tuple(x for x in gridNeighbours(current, GRID_SIZE)
                                if x is not None)
        unvisited = [v for v in validNeighbours if v not in visited]

        numToAdd = randint(0, min(nodes-len(visited), len(unvisited)))
        for n in range(numToAdd):
            newNode = choice(unvisited)
            graph[current].add(newNode)
            graph[newNode].add(current)
            visited.append(newNode)
            unvisited.remove(newNode)
            numEdges += 1
    if numEdges < edges:
        possibleNodes = {}
        uniqueEdges = 0
        for n in graph.keys():
            validNeighbours = list(x for x in gridNeighbours(n, GRID_SIZE)
                                   if x in graph.keys() and n not in graph[x])
            if validNeighbours:
                possibleNodes[n] = validNeighbours
                uniqueEdges = sum(len(possibleNodes[n])
                                  for n in possibleNodes)//2

        if uniqueEdges < edges-numEdges:

            edgesNeeded = edges-numEdges-uniqueEdges
            # Not enough spots to add a needed edge
            graph = adjustLeafNodes(graph, edgesNeeded)
            # Return early. Generate network will run a check
            return graph, uniqueEdges
        while numEdges < edges:
            edgeNode1 = choice(list(possibleNodes.keys()))

            # Pick a neighbour from the possible ones calculated above
            edgeNode2 = choice(possibleNodes[edgeNode1])
            possibleNodes[edgeNode1].remove(edgeNode2)
            possibleNodes[edgeNode2].remove(edgeNode1)
            if len(possibleNodes[edgeNode1]) == 0:
                del possibleNodes[edgeNode1]
            if len(possibleNodes[edgeNode2]) == 0:
                del possibleNodes[edgeNode2]
            graph[edgeNode1].add(edgeNode2)
            graph[edgeNode2].add(edgeNode1)

            numEdges += 1

    return graph, 0


# Generate a network with the given number of edges and nodes
# Params:
#     edges - number of edges to use
#     nodes- number of nodes to use
#     probType - type of probability distribution to use uniform or non-uniform
#     initSeed - optional seed to use when generating graphs
def generateNetwork(edges, nodes, probType=None, initSeed=None):
    if initSeed is None:
        initSeed = randrange(sys.maxsize)
    # Seed the RNG
    seed(initSeed)
    print("Network Class: M", edges, "N", nodes, " Seed:", initSeed, sep="")
    global GRID_SIZE
    a = edges
    b = nodes
    if b > (GRID_SIZE ** 2):
        GRID_SIZE = math.ceil(math.sqrt(b))
    # The network / graph is rperesented with nodes as keys,
    # and the nodes they connect to as values (edges are pairs of nodes)
    graph = defaultdict(set)
    if a <= (2*b - math.ceil(2*math.sqrt(b))):
        graph, edgesToAdd = generateSparse(a, b, graph)

        retries = 0
        bestNumEdges = getNumEdges(graph)
        bestGraph = graph

        # and retries < 50:
        while (getNumEdges(graph) < a or getNumNodes(graph) < b):
            graph.clear()
            graph, edgesToAdd = generateSparse(a, b, graph)
            if getNumEdges(graph) + edgesToAdd > bestNumEdges:
                bestGraph = graph
                bestNumEdges = getNumEdges(graph) + edgesToAdd
            retries += 1
        if bestNumEdges != edges:
            print("Could not generate a graph with the required number of edges.\n Try using",
                  bestNumEdges, "edges.")
            graph = bestGraph

    else:
        # Dense network
        graph = generateDense(a, b, graph)
        # use network x to check if it's connected
        connectedTest = nx.Graph(graph)
        # If the generated dense graph is disconnected, generate a new one
        while not nx.is_connected(connectedTest):
            graph.clear()
            graph = generateDense(a, b, graph)
            connectedTest = nx.Graph(graph)

        connectedTest = None

    # assign probabilities to existing edges in graph
    prob, edges = generateProbabilities(graph, probType)

    return graph, prob, edges, initSeed


def genEdges(graph):
    """ generate a list of edges of the graph """
    edges = []
    e = 0
    for i in graph.keys():
        for j in graph[i]:
            if (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))
                e += 1
    return edges


def genArcs(graph):
    """ generate a list of arcs of the graph """
    arcs = []
    for i in graph:
        for j in graph[i]:
            if j is not None and (i, j) not in arcs:
                arcs.append((i, j))
                arcs.append((j, i))
    return arcs

def genNodes(graph):
    """ generate a list of nodes of the graph """
    return [n for n in graph.keys() if len(graph[n]) > 0]


def getNumEdges(graph):
    return sum(len(graph[n]) for n in graph.keys())//2


def getNumNodes(graph):
    return len(graph.keys())

# Write a graph to a file
def writeGraph(graph, seed, prob, path='./'):
    nodes = getNumNodes(graph)
    edges = getNumEdges(graph)
    graphClass = "M" + str(edges) + "N" + str(nodes)
    fileName = graphClass+"_"+str(seed)+".txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path+fileName, 'w') as f:
        f.write(str(dict(graph)) + '\n' + str(prob))
    return path+fileName

# read in graph and probability distribution from file
def readGraph(file):
    graph = defaultdict(set)
    prob = {}
    try:
        with open(file, 'r') as f:
            graph = ast.literal_eval(f.readline())
            prob = ast.literal_eval(f.readline())
    except FileNotFoundError as fne:
        print("File does not exist.")
    except ValueError as ve:
        print("An error occured reading the file", ve)
    return graph, prob
