from problemGen import generateNetwork, writeGraph, readGraph
from mip import MIP
#from mipBP import *
import re
import os


# Get the number of edges and nodes from a grraph class
def parseClass(classType):
    return [int(x) for x in re.split(r'[MN]', classType)[1:]]


# Generate the given number of poblems for each graph class
# Generated graphs are written to a file in ./problemInstances/class/probType/
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


# Display the given results in readable format in the console
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


# Display the generated results as a LaTeX table
def displayLatexFormat(results, classes, maxSearchers, instances):
    # Make a table for each return value (objective, runtime and mipgap)
    for val in range(3):
        for c in classes:
            print('\n\\begin{table}[H]')
            print('\\scriptsize')
            print(c)
            print('\\begin{tabular} {', '|c'*(instances+2), '|}', sep='')
            print('\hline')
            print('Scenario', end=' ')
            for i in range(1, instances+1):
                print('& ', i, sep='', end=' ')
            print('& Average\\\\')
            print('\hline')
            for k in range(1, maxSearchers+1):
                for pType in ['Uniform', 'Non-Uniform']:
                    print("K", k, "-", pType, sep='', end=' ')
                    totals = [0, 0, 0]
                    for i in range(1, instances+1):

                        print('& ', round(
                            results[c, pType, k, i][val], 3), sep='', end=' ')
                        totals[val] += results[c, pType, k, i][val]
                    print(' & ', round(totals[val]/instances, 3),
                          '\\\\ \hline', sep='')
            print('\end{tabular}')
            print('\end{table}')
            
def runBenchmarks(classes,improvements, maxSearchers=2, probType=['Uniform', 'Non-Uniform']):
    results = {}
    for c in classes:
        numEdges, numNodes = parseClass(c)
        for k in range(maxSearchers, 0, -1):
            for pType in probType:
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
                    #Hacky way to get seed
                    seed = file.split('.txt')[0].split(c+'_')[1]
                    # ProbType param doesn't matter since p is passed in
                    mip, graph, _ = MIP(
                        0, k, numEdges, numNodes, 2*numEdges//k, 
                        graph=graph, prob=prob, improvements=improvements)

                    results[c, pType, k, count] = (mip.objVal, mip.RunTime, mip.MipGap, seed)

                    print((c, pType, k, count), ":", (mip.objVal, mip.RunTime, mip.mipGap))
                    print('Complete', c, pType, "searchers:", k)
                problem = ''
                for label in improvements:
                    if improvements[label]:
                        problem += label + "_"
                with open(path+problem+'results.txt', 'w') as f:
                     f.write(str(results))
    return results


if __name__ == '__main__':
    #    classes = ['M19N15', 'M24N18', 'M20N15', 'M20N8', 'M30N22', 'M30N12',
    #               'M30N9', 'M40N38', 'M40N16', 'M40N11', 'M50N35', 'M50N20',
    #               'M50N14']

    classes = ['M24N18']
    improvements = {
        "tighter_T_bound": False,
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": True,
        "start_at_leaf_hint": False,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "branch_direction": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": False,
        "Y_BP": False,
        "high_prob_edges_BP": False
    }
    runBenchmarks(classes, improvements, maxSearchers=1, probType=['Uniform'])
    improvements = {
        "tighter_T_bound": False,
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": False,
        "start_at_leaf_hint": True,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "branch_direction": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": False,
        "Y_BP": False,
        "high_prob_edges_BP": False
    }
    runBenchmarks(classes, improvements, maxSearchers=1, probType=['Uniform'])
    improvements = {
        "tighter_T_bound": False,
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": False,
        "start_at_leaf_hint": False,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "branch_direction": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": True,
        "Y_BP": False,
        "high_prob_edges_BP": False
    }
    runBenchmarks(classes, improvements, maxSearchers=1, probType=['Uniform'])
    improvements = {
        "tighter_T_bound": False,
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": False,
        "start_at_leaf_hint": False,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "branch_direction": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": False,
        "Y_BP": True,
        "high_prob_edges_BP": False
    }
    runBenchmarks(classes, improvements, maxSearchers=1, probType=['Uniform'])
    improvements = {
        "tighter_T_bound": False,
        "start_at_leaf_constraint": False,
        "start_at_leaf_BP": False,
        "start_at_leaf_hint": False,
        "dont_visit_searched_leaves": False,
        "travel_towards_unsearched": False,
        "branch_direction": False,
        "barrier_log": False,
        "Y_cts": False,
        "early_X_BP": False,
        "Y_BP": False,
        "high_prob_edges_BP": True
    }
    runBenchmarks(classes, improvements, maxSearchers=1, probType=['Uniform'])
    
    #classes = ['M19N15']
    # generateProblems(classes, instances)
#    with open('BranchPriorityResults.txt', 'r') as f:
#        results = eval(f.read())
#    print(results)
#    displayResults(results, classes, 2, instances)
#    displayLatexFormat(results, classes, 2, instances)
    
#    for c in classes:
#        with open(c + 'results.txt', 'r') as f:
#            temp = eval(f.read())
#            for key in temp:
#                results[key] = temp[key]
#
#    displayResults(results, classes, 2, instances)
#    displayLatexFormat(results, classes, 2, instances)
    
    
