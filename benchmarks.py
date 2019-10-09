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
                graph, prob, _, graphSeed = generateNetwork(edges, nodes, probType, graphSeed)
                writeGraph(graph, graphSeed, prob, './problemInstances/'+c+'/'+pType+'/')

def displayResults(results):
    
    
    
if __name__ == '__main__':
#    classes = ['M19N15', 'M24N18', 'M20N15', 'M20N8', 'M30N22', 'M30N12',
#               'M30N9', 'M40N38', 'M40N16', 'M40N11', 'M50N35', 'M50N20',
#               'M50N14']
    
    classes = ['M19N15', 'M24N18']
    
    instances = 10
    # generateProblems(classes, instances)
    UNIFORM = 0
    NON_UNIFORM = 1

    results = {}
    for c in classes:
        numEdges, numNodes = parseClass(c)
        for k in range(1, 3):
            for pType in ['Uniform', 'Non-Uniform']:
                path = './problemInstances/'+c+'/'+pType+'/'
                count = 0
                for file in os.listdir(path):
                    try :
                        graph, prob = readGraph(path+file)
                    #Found a results.txt, so skip it
                    except SyntaxError:
                        print("skipped", file)
                        continue
                    count+=1
                    
                    # ProbType param doesn't matter since p is passed in
                    mip, graph, _ = MIP(UNIFORM, k, numEdges, numNodes, 2*numEdges//k, graph=graph, p=prob)
                    results[c, pType, k, count] = (mip.objVal, mip.RunTime, mip.MipGap)
                    print('Complete', c, pType, "searchers:", k)
                
                with open(path+'results.txt', 'w') as f:
                    f.write(str(results))
    print(results)