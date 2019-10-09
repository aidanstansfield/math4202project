from problemGen import generateNetwork, writeGraph, readGraph
from mip import MIP
import re
import os


def parseClass(classType):
    return [int(x) for x in re.split(r'[MN]', classType)[1:]]


def generateProblems(classes, num):
    for c in classes:
        edges, nodes = parseClass(c)
        for graph in range(num):
            graphSeed = None
            for pType in ['Uniform', 'Non-Uniform']:
                probType = 0 if (pType == 'Uniform') else 1
                graph, prob, _, graphSeed = generateNetwork(
                    edges, nodes, probType, graphSeed)
                test = writeGraph(graph, graphSeed, prob,
                                  './problemInstances/'+c+'/'+pType+'/')


def displayResults(results, classes, maxSearchers, instances):
    for c in classes:
        print("*****", c, "*****")
        for k in range(1, maxSearchers+1):
            for pType in ['Uniform', 'Non-Uniform']:
                print(c, " K", k, "-", pType, sep='')
                objTotal = 0
                timeTotal = 0
                gapTotal = 0
                for i in range(1, instances+1):

                    if (c, pType, k, i) in results:
                        obj, time, mipgap = results[c, pType, k, i]
                        print(i, " |Obj:", obj, " |Runtime:",
                              time, " |MipGap:", mipgap, sep='\t')
                        objTotal += obj
                        timeTotal += time
                        gapTotal += mipgap

                print('Avg Obj:', objTotal/instances)
                print('Avg Time', timeTotal/instances)
                print('Avg Gap', gapTotal/instances)
                print()


if __name__ == '__main__':
    #    classes = ['M19N15', 'M24N18', 'M20N15', 'M20N8', 'M30N22', 'M30N12',
    #               'M30N9', 'M40N38', 'M40N16', 'M40N11', 'M50N35', 'M50N20',
    #               'M50N14']

    classes = ['M19N15', 'M24N18']

    instances = 10
    # generateProblems(classes, instances)
    UNIFORM = 0
    NON_UNIFORM = 1

    with open('NonSameResults_M19N15_M25N18.txt', 'r') as f:
        results = eval(f.read())

#    displayResults(results, classes, 3, 10)
    results = {}
    for c in classes:
        numEdges, numNodes = parseClass(c)
        for k in range(2, 0, -1):
             for pType in ['Uniform', 'Non-Uniform']:
                 path = './problemInstances/'+c+'/'+pType+'/'
                 count = 0
                 for file in os.listdir(path):
                     try:
                         graph, prob = readGraph(path+file)
                     # Found a results.txt, so skip it
                     except SyntaxError:
                         print("skipped", file)
                         continue
                     count += 1

                     # ProbType param doesn't matter since p is passed in
                     mip, graph, _ = MIP(
                         UNIFORM, k, numEdges, numNodes, 2*numEdges//k, graph=graph, p=prob)
                     
                     results[c, pType, k, count] = (mip.objVal, mip.RunTime, mip.MipGap)
                     
                     print((c, pType, k, count), ":", (mip.objVal, mip.RunTime, mip.mipGap))
                     print('Complete', c, pType, "searchers:", k)

                 with open(path+'results.txt', 'w') as f:
                     f.write(str(results))
    print(results)
    displayResults(results, classes, 2, instances)