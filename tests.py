import unittest
from problemGen import generateNetwork


class TestGenerateNetwork(unittest.TestCase):

    def setUp(self):
        self.smallDenseNodes = 12
        self.smallDenseEdges = 28
        self.largeDenseNodes = 150
        self.largeDenseEdges = 300
        self.sparseNodes = 15
        self.sparseEdges = 19
    
    def testCorrectNodesSparseNetwork(self):
        network, _, _, _ = generateNetwork(self.sparseEdges, self.sparseNodes)

        self.assertEqual(self.sparseNodes, len(network.keys()))

    def testCorrectEdgesSparseNetwork(self):
        network, _, _, _ = generateNetwork(self.sparseEdges, self.sparseNodes)

        self.assertEqual(self.sparseEdges, sum(len(network[n]) for n in network.keys())//2)

    def testCorrectNodesDenseNetwork(self):
        network, _, _, _  = generateNetwork(self.smallDenseEdges, self.smallDenseNodes)

        self.assertEqual(self.smallDenseNodes, len(network.keys()))

    def testCorrectEdgesSmallDenseNetwork(self):
        network, _, _, _ = generateNetwork(self.smallDenseEdges, self.smallDenseNodes)

        self.assertEqual(self.smallDenseEdges, sum(len(network[n]) for n in network.keys())//2)
    
    def testCorrectEdgesLargeDenseNetwork(self):
        network, _, _, _ = generateNetwork(self.largeDenseEdges, self.largeDenseNodes)

        self.assertEqual(self.largeDenseEdges, sum(len(network[n]) for n in network.keys())//2)


if __name__ == "__main__":
    unittest.main()
