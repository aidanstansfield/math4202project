# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:08:57 2019

@author: Daniel
"""
from collections import defaultdict
from random import randint, choice, seed, randrange
from datetime import datetime
import sys
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


def generateProbabilities(graph, probType):
    pUn = None  # uniform prob
    pNon = None  # non-uniform prob
    # number of nodes
    numNodes = sum(n for n in graph.keys())
    # number of edges
    numEdges = sum(len(graph[n]) for n in graph.keys())//2
    if not probType:
        # gen uniformly dist. probs
        pUn = 1/numEdges
        edgeNums = numEdges
    elif probType:
        # define number of edges st. Cl: Cm: Ch app = 1: 2: 1
        edgeFloor = numEdges//4
        edgeMod = numEdges % 4  # num edges mod 4
        # we need to split the edges into groups approximately of the ratio
        # 1: 2: 1 (4 "pieces")- decide what to do with leftovers depending on
        # how many leftovers
        edgeNums = [edgeFloor, 2 * edgeFloor, edgeFloor]

        if edgeMod < 0 or edgeMod > 3:
            print('Something went wrong')
            return -1
        if edgeMod == 1 or edgeMod == 2:
            edgeNums[1] += edgeMod
        elif edgeMod == 3:
            edgeNums = [i+1 for i in edgeNums]

        # define list of probs of the form [pl, pm, ph]
        p = 1/(edgeNums[0] + 2 * edgeNums[1] + 3 * edgeNums[2])
        pNon = [p, 2 * p, 3 * p]
        pNon = [p for p in pNon]
    else:
        print('Invalid Graph Type')
        return -1
    return pUn, pNon, edgeNums


def generateDense(edges, nodes, graph):
    for i in range(nodes):
        graph[nodes] = set()
    edgesMade = 0

    while edgesMade < edges:
        node1 = randint(1, nodes)
        node2 = randint(1, nodes)
        while node2 == node1:
            node2 = randint(1, nodes)
        if node2 not in graph[node1] and node1 not in graph[node2]:
            graph[node1].add(node2)
            graph[node2].add(node1)
            edgesMade += 1
    return graph


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
        # Dense network
        graph = generateDense(a,b, graph)

        # use network x to check if it's connected
        connectedTest = nx.Graph(graph)

        # If the generated dense graph is disconnected, generate a new one
        while not nx.is_connected(connectedTest):
            graph.clear()
            graph = generateDense(a, b, graph)
            connectedTest = nx.Graph(denseInstance)

        connectedTest = None

    # assign probabilities to existing edges in graph
    pUn, pNon, edgeNums = generateProbabilities(graph, probType)
    edges = genEdges(graph)
    prob = {}
    if pUn is not None:
        # uniform case
        prob[pUn] = [e for e in edges.values()]
    else:
        generatedTypes = [0, 0, 0]
        for p in pNon:
            prob[p] = []
        for e in edges.keys():
            edgeType = randint(0, 2)
            while generatedTypes[edgeType] == edgeNums[edgeType]:
                edgeType = randint(0, 2)
            prob[pNon[edgeType]].append(edges[e])
            generatedTypes[edgeType] += 1

    return graph, prob, edges


def genEdges(graph):
    edges = {}
    e = 0
    for i in graph.keys():
        for j in graph[i]:
            if (i, j) not in edges.values() and (j, i) not in edges.values():
                edges[e] = (i, j)
                e += 1
    return edges


def genArcs(graph):
    arcs = []
    for i in graph.keys():
        for j in graph[i]:
            if j is not None and (i, j) not in arcs:
                arcs.append((i, j))
    return arcs


def genNodes(graph):
    nodes = []
    for n in graph:
        nodes.append(n)
    return nodes


def S(a, n):
    # does arc a start on node n
    if a[0] == n:
        return 1
    else:
        return 0


def E(a, n):
    # does arc a end on node n
    if a[1] == n:
        return 1
    else:
        return 0


def O(a, e):
    # does edge e contain arc a
    if a[0] == e[0] and a[1] == e[1]:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sparseInstance, p1, _ = generateNetwork(24, 18, 0)
    denseInstance, p2, _ = generateNetwork(28, 12, 1)

    displayLattice(graph=sparseInstance)

    dense = nx.Graph(denseInstance)

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
