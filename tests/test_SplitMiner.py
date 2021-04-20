from unittest import TestCase

from src.SplitMiner import SplitMiner


class TestSplitMiner(TestCase):
	def setUp(self) -> None:
		self.split_miner = SplitMiner("../logs/preprocessed/B1.csv")
	
	def test_get_most_frequent_edge_from_set(self):
		arc_frequencies = {('a', 'b'): 2, ('c', 'd'): 5, ('e', 'f'): 6}
		edges = {('a', 'b'), ('c', 'd')}
		result = self.split_miner.get_most_frequent_edge_from_set(edges, arc_frequencies)
		self.assertEqual(('c', 'd'), result)
	
	def test_add_edges_with_greater_threshold(self):
		arc_frequencies = {('a', 'b'): 2, ('c', 'd'): 5, ('e', 'f'): 6}
		threshold = 3
		actual_edges = set()
		graph = {'a': {'b'}, 'c': {'d'}, 'e': {'f'}}
		result = self.split_miner.add_edges_with_greater_threshold(threshold, actual_edges, graph, arc_frequencies)
		self.assertEqual({('c', 'd'), ('e', 'f')}, result)
	
	def test_get_percentile_frequency(self):
		arc_frequencies = {('a', 'b'): 2, ('c', 'd'): 5, ('e', 'f'): 6, ('r', 'f'): 5, ('g', 'f'): 3}

		frequent_edges = {('a', 'b'), ('c', 'd'), ('e', 'f'), ('r', 'f')}
		eta = 50
		result = self.split_miner.get_percentile_frequency(frequent_edges, eta, arc_frequencies)
		self.assertEqual(5, result)
		