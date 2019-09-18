import unittest
from problemGen import generateNetwork

class TestGenerateNetwork(unittest.TestCase):

    def setUp(self):
        self.denseNodes = 12
        self.denseEdges = 28
        self.sparseNodes = 18
        self.sparseEdges = 24
    
    def testCorrectNodesSparseNetwork(self):
        network,_,_ = generateNetwork(self.sparseEdges, self.sparseNodes)

        self.assertEqual(self.sparseNodes, len(network.keys()))

    def testCorrectEdgesSparseNetwork(self):
        network,_,_ = generateNetwork(self.sparseEdges, self.sparseNodes)

        self.assertEqual(self.sparseEdges, sum(len(network[n]) for n in network.keys())//2)

    def testCorrectNodesDenseNetwork(self):
        network,_,_ = generateNetwork(self.denseEdges, self.denseNodes)

        self.assertEqual(self.denseNodes, len(network.keys()))

    def testCorrectEdgesDenseNetwork(self):
        network,_,_ = generateNetwork(self.denseEdges, self.denseNodes)

        self.assertEqual(self.denseEdges, sum(len(network[n]) for n in network.keys())//2)



if __name__ == "__main__":
    unittest.main()