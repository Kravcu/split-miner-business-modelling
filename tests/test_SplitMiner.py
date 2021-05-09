from unittest import TestCase

from BPMNModel import BPMNModel
from LogFile import LogFile
from SplitMiner import SplitMiner


class TestSplitMiner(TestCase):
    def setUp(self) -> None:
        self.split_miner = SplitMiner("../logs/B1.csv")
        self.default_test_path = "../logs/B1.csv"

    def test_get_most_frequent_edge_from_set(self):
        arc_frequencies = {('a', 'b'): 2, ('c', 'd'): 5, ('e', 'f'): 6}
        edges = {('a', 'b'), ('c', 'd')}
        result = self.split_miner.get_most_frequent_edge_from_set(edges, arc_frequencies)
        self.assertEqual(('c', 'd'), result)
    
    def test_get_dfg(self):
        temp_split_miner = SplitMiner(self.default_test_path)
        expected_start_set = {'a'}
        expected_end_set = {'e', 'd'}
        expected_dfg = {'a': {'b', 'c'}, 'b': {'a'}, 'c': {'c', 'e', 'd'}, 'd': {'e'}, 'e': {'d'}}
        actual_dfg, actual_start_set, actual_end_set = temp_split_miner.get_dfg()
        self.assertEqual(expected_start_set, actual_start_set)
        self.assertEqual(expected_end_set, actual_end_set)
        self.assertEqual(expected_dfg, actual_dfg)
    
    def test_find_self_loops(self):
        temp_split_miner = SplitMiner(self.default_test_path)
        expected_self_loops = {'c'}
        self.assertEqual(expected_self_loops, temp_split_miner.find_self_loops())
    
    def test_find_short_loops(self):
        temp_split_miner = SplitMiner(self.default_test_path)
        expected_short_loops = {('a', 'b'): 3}
        self.assertEqual(expected_short_loops, temp_split_miner.short_loops)

    def test_count_arc_frequency(self):
        temp_split_miner = SplitMiner(self.default_test_path)
        expected_frequencies = {('a', 'b'): 3, ('b', 'a'): 3, ('a', 'c'): 5, ('c', 'd'): 3, ('d', 'e'): 3,
                                ('c', 'c'): 4, ('c', 'e'): 2, ('e', 'd'): 2}
        self.assertEqual(expected_frequencies, temp_split_miner.arc_frequency)
    
    def test_remove_short_loops(self):
        temp_split_miner = SplitMiner(self.default_test_path)
        expected_frequencies = {('a', 'c'): 5, ('c', 'd'): 3, ('d', 'e'): 3,
                                ('c', 'c'): 4, ('c', 'e'): 2, ('e', 'd'): 2}
        expected_dfg = {'a': { 'c'}, 'b': set(), 'c': {'c', 'e', 'd'}, 'd': {'e'}, 'e': {'d'}}
        temp_split_miner.remove_short_loops()
        self.assertEqual(expected_frequencies, temp_split_miner.arc_frequency)
        self.assertEqual(expected_dfg, temp_split_miner.direct_follows_graph)
        
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
        pdfg = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g', 'f'}, 'd': {'g'},
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
        bpmn = BPMNModel("", "", set(), set(), set(), set(), set())
        splits = {"b": ({"b"}, {"c", "d"}),
                  "c": ({"c"}, {"b"}),
                  "d": ({"d"}, {"b"})}
        succesors = {"b", "c", "d"}
        actual_node = "a"
        self.split_miner.discover_xor_splits(bpmn, succesors, splits, actual_node)
        expected = {"b": ({"b"}, {"c", "d"}),
                    "xora": ({"c", "d"}, {"b"})}

        self.assertEqual(expected, splits)

        expected_edges = {("xora", "c"), ("xora", "d")}
        result_edges = bpmn.edges
        self.assertEqual(expected_edges, result_edges)

        self.assertEqual({"b", "xora"}, succesors)

    def test_discover_and_splits(self):
        bpmn = BPMNModel("", "", set(), set(), set(), {("xora", "c"), ("xora", "d")}, set())
        input_splits = {"b": ({"b"}, {"c", "d"}),
                    "xora": ({"c", "d"}, {"b"})}
        succesors = {"b", "xora"}
        actual_node = "a"

        expected = {"anda": ({"b", "c", "d"}, set())}
        self.split_miner.discover_and_splits(bpmn, succesors, input_splits, actual_node)
        self.assertEqual(expected, input_splits)
        expected_edges = {("anda", "b"), ("anda", "xora"), ("xora", "c"), ("xora", "d")}
        result_edges = bpmn.edges
        self.assertEqual(expected_edges, result_edges)

    def test_get_init_bpmn_edges_without_actual_node(self):
        pdfg = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g', 'f'}, 'd': {'g'},
                 'e': {'c', 'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
        actual_node = 'a'
        expected = {('b', 'e'), ('b', 'f'), ('d', 'g'),
                     ('e', 'c'), ('e', 'h'), ('c', 'f'),
                     ('c', 'g'), ('f', 'g'), ('g', 'h')}
        result = self.split_miner.get_init_bpmn_edges_without_actual_node(pdfg, actual_node)
        self.assertEqual(expected, result)

    def test_discover_splits_of_node(self):
        pdfg = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g', 'f'}, 'd': {'g'},
                'e': {'c', 'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
        actual_node = 'a'
        expected = {('a', 'anda'), ('anda', 'b'), ('anda', 'xora'), ('xora', 'c'), ('xora', 'd'),
                    ('b', 'e'), ('b', 'f'), ('d', 'g'),
                    ('e', 'c'), ('e', 'h'), ('c', 'f'),
                    ('c', 'g'), ('f', 'g'), ('g', 'h')}
        self.split_miner.bpmn_model.edges = {('b', 'e'), ('b', 'f'), ('d', 'g'),
                     ('e', 'c'), ('e', 'h'), ('c', 'f'),
                     ('c', 'g'), ('f', 'g'), ('g', 'h')}
        concurrent_node = {("b", "c"), ("b", "d")}
        self.split_miner.discover_splits_of_node(pdfg, actual_node, concurrent_node)
        self.assertEqual(expected, self.split_miner.bpmn_model.edges)

    def test_discover_all_splits(self):
        pdfg = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g'}, 'd': {'g'},
                'e': {'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
        concurrent_node = {("b", "c"), ("b", "d")}
        self.split_miner.filtered_graph = pdfg
        self.split_miner.concurrent_nodes = concurrent_node

        self.split_miner.discover_all_splits()
        expected_edges = {('a', 'anda'), ('anda', 'b'), ('anda', 'xora'), ('xora', 'c'), ('xora', 'd'),
                    ('b', 'xorb'), ('xorb', 'f'), ('xorb', 'e'), ('d', 'g'),
                    ('e', 'h'),
                    ('c', 'g'), ('f', 'g'), ('g', 'h')}
        result = self.split_miner.bpmn_model.edges
        self.assertEqual(expected_edges, result)

    def test_get_nodes_with_multiple_successors(self):
        pdfg = {'a': {'d', 'b', 'c'}, 'b': {'e', 'f'}, 'c': {'g'}, 'd': {'g'},
                'e': {'h'}, 'f': {'g'}, 'g': {'h'}, 'h': {}}
        self.split_miner.filtered_graph = pdfg

        expected = {'a', 'b'}
        result = self.split_miner.get_nodes_with_multiple_successors()

        self.assertEqual(expected, result)

