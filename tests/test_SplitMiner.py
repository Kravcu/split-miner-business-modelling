from unittest import TestCase

from src.BpmnModel import BpmnModel
from src.SplitMiner import SplitMiner


class TestSplitMiner(TestCase):
	def setUp(self) -> None:
		self.split_miner = SplitMiner("../logs/preprocessed/B1.csv")
	
	def test_get_most_frequent_edge_from_set(self):
		arc_frequencies = {('a', 'b'): 2, ('c', 'd'): 5, ('e', 'f'): 6}
		edges = {('a', 'b'), ('c', 'd')}
		result = self.split_miner.get_most_frequent_edge_from_set(edges, arc_frequencies)
		self.assertEqual(('c', 'd'), result)
		
	def test_get_most_frequent_edge_from_set_when_real_input(self):
		arc_frequency = {('a', 'b'): 60, ('a', 'c'): 20, ('a', 'd'): 20,
		                 ('b', 'e'): 40, ('b', 'f'): 20, ('d', 'g'): 20,
		                 ('e', 'c'): 10, ('e', 'h'): 20, ('c', 'f'): 10,
		                 ('c', 'g'): 20, ('f', 'g'): 30, ('g', 'h'): 80}
		
		edges = {('a', 'd'), ('a', 'b'), ('d', 'g'), ('e', 'h'), ('c', 'g'), ('a', 'c'), ('b', 'f'), ('g', 'h'),
		         ('f', 'g'), ('b', 'e')}
		
		result = self.split_miner.get_most_frequent_edge_from_set(edges, arc_frequency)
		self.assertEqual(('g', 'h'), result)
	
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
	
	def test_get_predecessors(self):
		graph = {'a': {'b'}, 'c': {'d', 'b'}, 'e': {'f'}}
		node = 'b'
		result = self.split_miner.get_predecessors(graph, node)
		self.assertEqual({'a', 'c'}, result)
	
	def test_get_most_frequent_edge_for_each_node(self):
		graph = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g', 'f'}, 'd': {'g'},
		         'e': {'c', 'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
		arc_frequency = {('a', 'b'): 60, ('a', 'c'): 20, ('a', 'd'): 20,
		                 ('b', 'e'): 40, ('b', 'f'): 20, ('d', 'g'): 20,
		                 ('e', 'c'): 10, ('e', 'h'): 20, ('c', 'f'): 10,
		                 ('c', 'g'): 20, ('f', 'g'): 30, ('g', 'h'): 80}
		
		expected = {('a', 'c'), ('a', 'd'), ('a', 'b'), ('b', 'f'),
		            ('b', 'e'), ('e', 'h'), ('f', 'g'), ('d', 'g'),
		            ('g', 'h'), ('c', 'g')}
		result = self.split_miner.get_most_frequent_edge_for_each_node(graph, arc_frequency)
		self.assertEqual(expected, result)
		
	def test_filter_graph(self):
		pdfg = graph = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g', 'f'}, 'd': {'g'},
		         'e': {'c', 'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
		arc_frequency = {('a', 'b'): 60, ('a', 'c'): 20, ('a', 'd'): 20,
		                 ('b', 'e'): 40, ('b', 'f'): 20, ('d', 'g'): 20,
		                 ('e', 'c'): 10, ('e', 'h'): 20, ('c', 'f'): 10,
		                 ('c', 'g'): 20, ('f', 'g'): 30, ('g', 'h'): 80}
		self.split_miner.filter_graph(pdfg, 50, arc_frequency)
		result = self.split_miner.filtered_graph
		expected = {'a': {'c', 'b', 'd'}, 'b': {'f', 'e'}, 'c': {'g'},
		            'd': {'g'}, 'e': {'h'}, 'f': {'g'}, 'g': {'h'},
		            'h': set()}
		
		self.assertEqual(expected, result)

	def test_get_init_splits_for_node(self):
		concurrent_nodes = {("b", "c"), ("b", "d")}
		successors = {"b", "c", "d"}
		splits = dict()
		result = self.split_miner.get_init_splits_for_node(concurrent_nodes, successors, splits)
		expected = {"b": ({"b"}, {"c", "d"}),
					"c": ({"c"}, {"b"}),
					"d": ({"d"}, {"b"})}

		self.assertEqual(expected, result)

	def test_discover_xor_splits(self):
		bpmn = BpmnModel("", "", set(), set(), set(), set(), set())
		splits = {"b": ({"b"}, {"c", "d"}),
					"c": ({"c"}, {"b"}),
					"d": ({"d"}, {"b"})}
		succesors = {"b", "c", "d"}
		actual_node = "b"
		self.split_miner.discover_xor_splits(bpmn, succesors, splits, actual_node)
		expected = {"b": ({"b"}, {"c", "d"}),
					"xorb": ({"c", "d"}, {"b"})}

		self.assertEqual(expected, splits)


