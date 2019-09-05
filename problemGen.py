# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:08:57 2019

@author: Daniel
"""
from collections import defaultdict
from random import randrange, choice

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
    print('Sparse Network')
    for i in range(size):
        for j in range(size):
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
    print('End Sparse Network')


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


# Generate a network with the given number of edges and nodes
def generateNetwork(edges, nodes):
    a = edges
    b = nodes

    # The network / graph is rperesented with nodes as keys,
    # and the nodes they connect to as values (edges are pairs of nodes)
    graph = defaultdict(list)
    if a/b < 2:

        # TODO - Finish this
        # Sparse network - like a manhattan network
        start = randrange(0, 99)
        graph[start] = []
        prevNode = start
        visited = [start]
        # Make a-1 edges
        while len(visited) < b:
            # Generate a tuple of valid neighbour indexes
            # (if a value is on the end of the grid,
            # neighbours outside the grid are None)
            validNeighbours = tuple(
                x for x in gridNeighbours(prevNode)
                if x is not None
                )
            nextNode = choice(validNeighbours)
            if nextNode not in visited:
                graph[prevNode].append(nextNode)
                graph[nextNode].append(prevNode)
                visited.append(nextNode)

            prevNode = nextNode
        print(len(visited))
        print("edges to add:", a - len(visited)-1)
        # Add edges until number off edges is a
        # If a new node is needed to do this, remove a leaf node, then add the node required
        print([k for i in graph for k in gridNeighbours(i) if k in graph])
        leafNodes = [n for n in graph if len(graph[n]) == 1]
        print(leafNodes)

    else:
        # Dense network
        for i in range(b):
            graph[b] = []
        edgesMade = 0
        while edgesMade < a:
            node1 = randrange(0, b+1)
            node2 = randrange(0, b+1)
            if node2 not in graph[node1]:
                graph[node1].append(node2)
                if node1 not in graph[node2]:
                    graph[node2].append(node1)
                edgesMade += 1
    return graph

if __name__ == "__main__":
    sparseInstance = generateNetwork(24, 18)
    
    denseInstance = generateNetwork(30, 12)
    
    displayLattice(graph=sparseInstance)
    dense = nx.Graph(denseInstance)
    # If the generated dense graph is disconnected, generate a new one
    while not nx.is_connected(dense):
        denseInstance = generateNetwork(30, 12)
        dense = nx.Graph(denseInstance)
    
    # print(dense.edges)
    print(denseInstance)
    print(nx.to_dict_of_lists(dense))
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
