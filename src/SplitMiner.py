import copy
from collections import defaultdict
from itertools import combinations
from typing import Set, Dict, Tuple, List
import numpy as np
from more_itertools import pairwise
from LogFile import LogFile
from BPMNModel import BPMNModel


class SplitMiner:

    def __init__(self, path):

        self.log = LogFile(path)
        self.direct_follows_graph, self.start_event_set, self.end_event_set = self.get_dfg()
        self.self_loops = self.find_self_loops()
        self.short_loops = self.find_short_loops()
        self.arc_frequency = self.count_arc_frequency()
        self.concurrent_nodes = set()
        self.concurrent_nodes.add("Not performed")
        self.pdfg = dict()
        self.filtered_graph = dict()
        self.bpmn_model = BPMNModel("Not implemented", "Not implemented", set(), set(), set(), set(), set())

    def get_dfg(self) -> Tuple[Dict[str, set], set, set]:
        """
        Function to get DFG graph, and start/end event sets from DataFrame containing traces
        :return: direct_follows_graph, start_event_set, end_event_set
        :rtype: Tuple[Dict[str, set], set, set]
        """
        start_event_set = set()
        end_event_set = set()
        direct_follows_graph = dict()
        for trace in self.log.traces['trace'].values:
            if trace[0] not in start_event_set:
                start_event_set.add(trace[0])
            if trace[-1] not in end_event_set:
                end_event_set.add(trace[-1])
            for ev_i, ev_j in pairwise(trace):
                if ev_i not in direct_follows_graph.keys():
                    direct_follows_graph[ev_i] = set()
                direct_follows_graph[ev_i].add(ev_j)
            for event in end_event_set:
                if event not in direct_follows_graph.keys():
                    direct_follows_graph[event] = set()
        return direct_follows_graph, start_event_set, end_event_set

    def find_self_loops(self) -> Set[str]:
        """
        Function to discover self loops in DFG. Self loop is when a node has an outgoing edge to itself.
        :return: Set of nodes that are in self-loops
        :rtype: Set[str]
        """
        self_loops: Set[str] = set()
        for event in self.direct_follows_graph:
            if event in self.direct_follows_graph[event]:
                self_loops.add(event)
        return self_loops

    def find_short_loops(self) -> Set[Tuple[str, str]]:
        """
        Function to find short-loops in traces.
        A short loop is a pattern {a,b,a} in a trace, where a,b - events.
        {a,a,a} is not considered a short-loop but a self-loop
        :return: Set containing pairs (a,b)
        :rtype: Set[Tuple[str, str]]
        """
        short_loops: Set[Tuple[str, str]] = set()
        # getting trace column, transforming it to tuples, and getting unique traces
        traces: List[List[str]] = self.log.traces['trace'].transform(tuple).unique()
        for trace in traces:
            for source, node, target in zip(trace[0:], trace[1:], trace[2:]):
                if source == target and (source != node):
                    short_loops.add((source, node))
        return short_loops

    def count_arc_frequency(self) -> Dict[Tuple[str, str], int]:
        """
        Function to count arc frequency. Returns a
        :return: dictionary in format (a,b) : frequency, where a,b - events,
            frequency - times this transition happens
        :rtype: Dict[Tuple[str, str], int]
        """
        arc_frequency: Dict[Tuple[str, str], int] = defaultdict(int)
        for trace in self.log.traces['trace'].values:
            for source, target in zip(trace, trace[1:]):
                arc_frequency[(source, target)] += 1
        return arc_frequency

    def remove_self_short_loops_from_dfg(self) -> None:
        """
        Function to remove self and short loops from dfg.
        :return: None
        :rtype: None
        """
        for self_loop_node in self.self_loops:
            self.direct_follows_graph[self_loop_node].remove(self_loop_node)

        for node_a, node_b in self.short_loops:
            self.direct_follows_graph[node_a].remove(node_b)

    def perform_mining(self) -> None:
        """
        Function to perform sequential steps to run split miner stages
        Returns a
        :return: None
        :rtype: None
        """
        self.remove_self_short_loops_from_dfg()
        self.concurrent_nodes = self.find_concurrency()
        self.pdfg = self.generate_pdfg()

    def find_concurrency(self, epsilon=0.8) -> Set[Tuple[str, str]]:
        """
        Function to find concurrent events. It is based on the dfg and on three formal conditions.
        Returns a set with concurrent events.
        :return: Set containing pairs (a,b)
        :rtype: Set[Tuple[str, str]]
        """
        concurrent_nodes: Set[Tuple[str, str]] = set()
        arc_frequency: Dict[Tuple[str, str], int] = self.count_arc_frequency()
        for node_a, node_b in combinations(self.direct_follows_graph.keys(), 2):  # check time complexity
            print(node_a, node_b)
            if node_b in self.direct_follows_graph[node_a] and node_a in self.direct_follows_graph[node_b]:
                if ((node_a, node_b) not in self.short_loops) and ((node_b, node_a) not in self.short_loops):
                    if (abs(arc_frequency[(node_a, node_b)] -
                            arc_frequency[(node_b, node_a)])) / (arc_frequency[(node_a, node_b)] +
                                                                 arc_frequency[(node_b, node_a)]) < epsilon:
                        concurrent_nodes.add((node_a, node_b))
        return concurrent_nodes

    def generate_pdfg(self) -> Dict[str, set]:
        """
        Function to prune concurrent events and generate pruned dfg.
        Returns a
        :return: pruned direct_follows_graph
        :rtype: Dict[str, set]
        """
        pdfg = self.direct_follows_graph.copy()
        for node_a, node_b in self.concurrent_nodes:
            pdfg[node_a].remove(node_b)
            pdfg[node_b].remove(node_a)
        return pdfg

    def filter_graph(self, pdfg: Dict[str, set], eta, arc_frequency):

        most_frequent_edges = self.get_most_frequent_edge_for_each_node(pdfg, arc_frequency)
        frequency_threshold = self.get_percentile_frequency(most_frequent_edges, eta, arc_frequency)
        most_frequent_edges = self.add_edges_with_greater_threshold(frequency_threshold,
                                                                    most_frequent_edges,
                                                                    pdfg,
                                                                    arc_frequency)
        filtered_edges = set()
        filtered_graph: Dict[str, set] = dict()
        for node in pdfg.keys():
            filtered_graph[node] = set()
        while len(most_frequent_edges) > 0:
            edge = self.get_most_frequent_edge_from_set(most_frequent_edges, arc_frequency)
            node_a, node_b = edge
            if (self.arc_frequency[edge] > frequency_threshold
                    or len(filtered_graph[node_a]) == 0
                    or len(self.get_predecessors(filtered_graph, node_b)) == 0):
                filtered_edges.add(edge)
                filtered_graph[node_a].add(node_b)
            most_frequent_edges.remove(edge)
        self.filtered_graph = filtered_graph

    def get_most_frequent_edge_for_each_node(self, graph: Dict[str, set], arc_frequency) -> Set[Tuple[str, str]]:
        """
        Function to get most frequent incoming and outgoing edge of each node
        Returns a
        :return: set of the most frequent edges
        :rtype: Set[str, str]
        """
        most_frequent_edges = set()
        for graph_node in graph.keys():
            outgoing_edges_freq: Dict[Tuple[str, str], int] = dict()
            for outgoing_node in graph[graph_node]:
                outgoing_edges_freq[(graph_node, outgoing_node)] = arc_frequency[(graph_node, outgoing_node)]

            if len(graph[graph_node]) > 0:
                max_outgoing_edge = max(outgoing_edges_freq, key=outgoing_edges_freq.get)
                most_frequent_edges.add(max_outgoing_edge)

            # to do predecessors
            predecessors = self.get_predecessors(graph, graph_node)
            incoming_edges_freq: Dict[Tuple[str, str], int] = dict()
            for predecessor in predecessors:
                incoming_edges_freq[predecessor, graph_node] = arc_frequency[(predecessor, graph_node)]
            if len(predecessors) > 0:
                max_incoming_edge = max(incoming_edges_freq, key=incoming_edges_freq.get)
                most_frequent_edges.add(max_incoming_edge)

        return most_frequent_edges

    @staticmethod
    def get_predecessors(graph: Dict[str, set], node: str) -> Set[str]:
        predecessors = set()
        for graph_node in graph.keys():
            if node in graph[graph_node]:
                predecessors.add(graph_node)
        return predecessors

    @staticmethod
    def get_percentile_frequency(most_frequent_edges: Set[Tuple[str, str]], eta, arc_frequency) -> float:
        """
        Function to compute frequency threshold based on the most frequent edges and percentile eta
        Returns a
        :return: frequency threshold
        :rtype: float
        """
        frequencies = []
        for node_a, node_b in most_frequent_edges:
            frequencies.append(arc_frequency[(node_a, node_b)])
        return np.percentile(np.array(frequencies), eta)

    @staticmethod
    def add_edges_with_greater_threshold(threshold, actual_edges: Set[Tuple[str, str]],
                                         graph: Dict[str, set], arc_frequency) -> Set[Tuple[str, str]]:
        """
        Function to add all edges with the frequency greater than threshold to the actual set of max edges
        Returns a
        :return: updated max edges
        :rtype: Set[Tuple[str, str]]
        """
        for node, succ_set in graph.items():
            for succ in succ_set:
                if arc_frequency[(node, succ)] > threshold:
                    actual_edges.add((node, succ))
        return actual_edges

    @staticmethod
    def get_most_frequent_edge_from_set(edges: Set[Tuple[str, str]], arc_frequency) -> Tuple[str, str]:
        """
        Function to find the most frequent edge from a given set of edges
        Returns a
        :return: edge with the highest frequency
        :rtype: Tuple[str, str]
        """
        frequencies = dict()
        for edge in edges:
            frequencies[edge] = arc_frequency[edge]
        return max(frequencies, key=frequencies.get)

    def discover_splits(self, filtered_pdfg: Dict[str, set], node_a: str,
                        concurrent_nodes: Set[Tuple[str, str]]):
        """
        Function to orchestrate split discovery, it uses functions which discover different types of splits
        """
        splits: Dict[str, Tuple[set, set]] = dict()
        node_a_successors = filtered_pdfg[node_a]
        splits = self.get_init_splits_for_node(concurrent_nodes, node_a_successors, splits)

    def get_init_splits_for_node(self, concurrent_nodes: Set[Tuple[str, str]], node_a_successors: Set[str],
                                 splits: Dict[str, Tuple[set, set]]) -> Dict[str, Tuple[set, set]]:
        """
        Function to generate initial covers and futures of each successor of the given task
        Returns a
        :return:initialized splits with the cover and future
        :rtype: Dict[str, Tuple[set, set]]
        """

        for successor in node_a_successors:
            successor_cover = {successor}
            successor_future = set()
            for successor_temp in node_a_successors:
                if successor_temp != successor and ((successor, successor_temp) in concurrent_nodes or
                                                    (successor_temp, successor) in concurrent_nodes):
                    successor_future.add(successor_temp)
            splits[successor] = (successor_cover, successor_future)
        return splits

    def discover_xor_splits(self, bpmn: BPMNModel, successors: Set[str], splits: Dict[str, Tuple[set, set]],
                            actual_node: str) -> None:
        """
        Function to modify the given split structure in order to introduce xor splits. It is base on the algorithm 3.
        Returns a
        :return:None
        :rtype: None
        """
        flag = True
        while flag:
            x = set()
            for successor1 in successors:
                cover_u, future_s = splits[successor1]
                future_k1 = copy.copy(future_s)
                for successor2 in successors:
                    future_k2 = splits[successor2][1]
                    if future_k1 == future_k2 and successor1 != successor2:
                        x.add(successor2)
                        cover_u = cover_u | splits[successor2][0]
                if x:
                    x = x | {successor1}
                    break
            if x:
                xor = f"xor{actual_node}"
                bpmn.xor_events.add(xor)
                for node in x:
                    bpmn.edges.add((xor, node))
                    splits.pop(node)
                for node in cover_u:
                    cover, future = splits.get(xor, (set(), set()))
                    cover.add(node)
                    splits[xor] = (cover, future)
                for node in future_s:
                    cover, future = splits.get(xor, (set(), set()))
                    future.add(node)
                    splits[xor] = (cover, future)
                successors.add(xor)
                successors = successors - x
            if not x:
                flag = False

    def discover_and_splits(self, bpmn: BPMNModel, successors: Set[str], splits: Dict[str, Tuple[set, set]],
                            actual_node: str) -> None:
        """
        Function to modify the given split structure in order to introduce and splits. It is base on the  algorithm 4.
        Returns a
        :return:None
        :rtype: None
        """
        a = set()
        for successor1 in successors:
            a = set()
            cover_u, future_i = splits[successor1]
            cover_future_1 = cover_u | future_i

            for successor2 in successors:
                cover2, future2 = splits[successor2]
                cover_future_2 = cover2 | future2
                if cover_future_1 == cover_future_2 and successor2 is not successor1:
                    a.add(successor2)
                    cover_u = cover_u | cover2
                    future_i = future_i.intersection(future2)

            if a:
                a = a | {successor1}
                break
        if a:
            and_gateway = f"and{actual_node}"
            for node in a:
                bpmn.edges.add((and_gateway, node))
                splits.pop(node)
            for node in cover_u:
                cover, future = splits.get(and_gateway, (set(), set()))
                cover.add(node)
                splits[and_gateway] = (cover, future)
            for node in future_i:
                cover, future = splits.get(and_gateway, (set(), set()))
                future.add(node)
                splits[and_gateway] = (cover, future)
            successors.add(and_gateway)
            successors = successors - a


# log = SplitMiner("../logs/preprocessed/B1.csv")
# log.perform_mining()
# print(log.direct_follows_graph)
"""
Initial DFG from Fig 2a (8 page)
dfg_report = \
    {
        'a': {'b', 'c', 'd'},
        'b': {'c', 'd', 'e', 'f'},
        'c':{'b','f','g',},
        'd':{'b','e','g'},
        'e':{'d','c','g','h'},
        'f':{'g'},
        'g': {'e','h'},
        'h':{}
    }
"""
