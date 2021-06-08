import copy
from collections import defaultdict
from itertools import combinations
from typing import Set, Dict, Tuple, List

import numpy as np
from more_itertools import pairwise

from BPMNModel import BPMNModel
from JavaCaller import JavaCaller
from LogFile import LogFile
from OrderedSet import OrderedSet


class SplitMiner:
    concurrent_nodes: Set[Tuple[str, str]]

    def __init__(self, path):

        self.log = LogFile(path)
        self.direct_follows_graph, self.start_event_set, self.end_event_set = self.get_dfg()
        #print(self.direct_follows_graph)
        self.self_loops = self.find_self_loops()
        self.short_loops = self.find_short_loops()
        self.arc_frequency = self.count_arc_frequency()
        self.concurrent_nodes = OrderedSet()
        self.concurrent_nodes.add("Not performed")
        self.pdfg = dict()
        self.filtered_graph = dict()
        self.bpmn_model = BPMNModel("Not implemented", "Not implemented", OrderedSet(), OrderedSet(), OrderedSet(),
                                    OrderedSet(), OrderedSet(),
                                    self.self_loops)

    def get_dfg(self) -> Tuple[Dict[str, set], set, set]:
        """
        Function to get DFG graph, and start/end event sets from DataFrame containing traces
        :return: direct_follows_graph, start_event_set, end_event_set
        :rtype: Tuple[Dict[str, set], set, set]
        """
        start_event_set = OrderedSet()
        end_event_set = OrderedSet()
        direct_follows_graph = dict()
        for trace in self.log.traces['trace'].values:
            if trace[0] not in start_event_set:
                start_event_set.add(trace[0])
            if trace[-1] not in end_event_set:
                end_event_set.add(trace[-1])
            for ev_i, ev_j in pairwise(trace):
                if ev_i not in direct_follows_graph.keys():
                    direct_follows_graph[ev_i] = OrderedSet()
                direct_follows_graph[ev_i].add(ev_j)
            for event in end_event_set:
                if event not in direct_follows_graph.keys():
                    direct_follows_graph[event] = OrderedSet()
        return direct_follows_graph, start_event_set, end_event_set

    def find_self_loops(self) -> Set[str]:
        """
        Function to discover self loops in DFG. Self loop is when a node has an outgoing edge to itself.
        :return: Set of nodes that are in self-loops
        :rtype: Set[str]
        """
        self_loops: Set[str] = OrderedSet()
        for event in self.direct_follows_graph:
            if event in self.direct_follows_graph[event]:
                self_loops.add(event)
        return self_loops

    def find_short_loops(self) -> Dict[Tuple[str, str], int]:
        """
        Function to find short-loops in traces.
        A short loop is a pattern {a,b,a} in a trace, where a,b - events.
        {a,a,a} is not considered a short-loop but a self-loop
        :return: dict containing pairs (a,b) and corresponding arc frequency
        :rtype: Dict[Tuple[str, str], int]
        """
        short_loops: Dict[Tuple[str, str], int] = defaultdict(int)
        # getting trace column, transforming it to tuples, and getting unique traces
        traces: List[List[str]] = self.log.traces['trace'].transform(tuple).unique()
        for trace in traces:
            was_detected_in_previous_node = False
            for source, node, target in zip(trace[0:], trace[1:], trace[2:]):
                if not was_detected_in_previous_node:
                    if source == target and (source != node):
                        short_loops[(source, node)] += 1
                        was_detected_in_previous_node = True
                else:
                    was_detected_in_previous_node = False
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

        self.remove_short_loops()

    def remove_short_loops(self):
        for node_a, node_b in self.short_loops.keys():
            self.arc_frequency[(node_a, node_b)] -= self.short_loops[(node_a, node_b)]
            self.arc_frequency[(node_b, node_a)] -= self.short_loops[(node_a, node_b)]
            self.delete_nodes_if_zero_frequency(node_a, node_b)
            self.delete_nodes_if_zero_frequency(node_b, node_a)

    def delete_nodes_if_zero_frequency(self, node_a, node_b):
        if self.arc_frequency[(node_a, node_b)] <= 0:
            self.arc_frequency.pop((node_a, node_b))
            if node_b in self.direct_follows_graph[node_a]:
                self.direct_follows_graph[node_a].remove(node_b)

    def restore_short_loops(self):
        short_loops = copy.copy(self.short_loops)
        for node_a, node_b in short_loops.keys():
            self.direct_follows_graph[node_a].add(node_b)
            self.direct_follows_graph[node_b].add(node_a)
            self.add_arc_frequency(node_a, node_b)
            self.add_arc_frequency(node_b, node_a)

    def add_arc_frequency(self, node_a, node_b):
        if (node_a, node_b) in self.arc_frequency:
            self.arc_frequency[(node_a, node_b)] += self.short_loops[(node_a, node_b)]
        else:
            self.arc_frequency[(node_a, node_b)] = self.short_loops[(node_a, node_b)]

    def perform_mining(self, eta=50) -> None:
        """
        Function to perform sequential steps to run split miner stages
        :return: None
        :rtype: None
        """
        self.remove_self_short_loops_from_dfg()
        self.concurrent_nodes = self.find_concurrency()
        self.restore_short_loops()
        self.pdfg = self.generate_pdfg()
        self.filter_graph(self.pdfg, eta, self.arc_frequency)
        self.init_bpmn()
        self.discover_all_splits()
        self.discover_start_splits()
        self.discover_joins(self.pdfg)
        self.discover_joins(self.pdfg)

    def find_concurrency(self, epsilon=0.8) -> Set[Tuple[str, str]]:
        """
        Function to find concurrent events. It is based on the dfg and on three formal conditions.
        Returns a set with concurrent events.
        :return: Set containing pairs (a,b)
        :rtype: Set[Tuple[str, str]]
        """
        concurrent_nodes: Set[Tuple[str, str]] = OrderedSet()
        arc_frequency: Dict[Tuple[str, str], int] = self.count_arc_frequency()
        for node_a, node_b in combinations(self.direct_follows_graph.keys(), 2):  # check time complexity
           # print(node_a, node_b)
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
        filtered_edges = OrderedSet()
        filtered_graph: Dict[str, set] = dict()
        for node in pdfg.keys():
            filtered_graph[node] = OrderedSet()
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
        most_frequent_edges = OrderedSet()
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
        predecessors = OrderedSet()
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

    def discover_splits_of_node(self, filtered_pdfg: Dict[str, set], node_a: str,
                                concurrent_nodes: Set[Tuple[str, str]]):
        """
        Function to orchestrate split discovery, it uses functions which discover different types of splits
        """
        successors = filtered_pdfg[node_a]
        splits: Dict[str, Tuple[set, set]] = dict()
        node_a_successors = filtered_pdfg[node_a]
        splits = self.get_init_splits_for_node(concurrent_nodes, node_a_successors, splits)
        # edges = self.get_init_bpmn_edges_without_actual_node(filtered_pdfg, node_a)
        # self.init_bpmn(edges)
        while len(successors) > 1:
            self.discover_xor_splits(self.bpmn_model, successors, splits, node_a)
            self.discover_and_splits(self.bpmn_model, successors, splits, node_a)
        for successor in successors:
            self.bpmn_model.edges.add((node_a, successor))

    def get_init_splits_for_node(self, concurrent_nodes: Set[Tuple[str, str]], node_a_successors: Set[str],
                                 splits: Dict[str, Tuple[set, set]]) -> Dict[str, Tuple[set, set]]:
        """
        Function to generate initial covers and futures of each successor of the given task
        Returns a
        :return:initialized splits with the cover and future
        :rtype: Dict[str, Tuple[set, set]]
        """

        for successor in node_a_successors:
            successor_cover = OrderedSet(successor)

            successor_future = OrderedSet()
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
            x = OrderedSet()
            for successor1 in successors:
                cover_u, future_s = splits[successor1]
                future_k1 = copy.copy(future_s)
                for successor2 in successors:
                    future_k2 = splits[successor2][1]
                    if future_k1 == future_k2 and successor1 != successor2:
                        x.add(successor2)
                        cover_u = cover_u | splits[successor2][0]
                if x:
                    x = x | OrderedSet(successor1)
                    break
            if x:
                xor = f"xor{actual_node}"
                bpmn.xor_events.add(xor)
                for node in x:
                    bpmn.edges.add((xor, node))
                    splits.pop(node)
                for node in cover_u:
                    cover, future = splits.get(xor, (OrderedSet(), OrderedSet()))
                    cover.add(node)
                    splits[xor] = (cover, future)
                for node in future_s:
                    cover, future = splits.get(xor, (OrderedSet(), OrderedSet()))
                    future.add(node)
                    splits[xor] = (cover, future)
                successors.add(xor)
                successors.difference_update(x)
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
        a = OrderedSet()
        for successor1 in successors:
            a = OrderedSet()
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
                a = a | OrderedSet(successor1)
                break
        if a:
            and_gateway = f"and{actual_node}"
            for node in a:
                bpmn.edges.add((and_gateway, node))
                splits.pop(node)
            for node in cover_u:
                cover, future = splits.get(and_gateway, (OrderedSet(), OrderedSet()))
                cover.add(node)
                splits[and_gateway] = (cover, future)
            for node in future_i:
                cover, future = splits.get(and_gateway, (OrderedSet(), OrderedSet()))
                future.add(node)
                splits[and_gateway] = (cover, future)
            successors.add(and_gateway)
            successors.difference_update(a)

    def init_bpmn(self):
        self.bpmn_model.start_events = self.start_event_set
        self.bpmn_model.end_events = self.end_event_set
        self.bpmn_model.tasks = self.direct_follows_graph.keys()

    def get_init_bpmn_edges_without_actual_node(self, pdfg: Dict[str, set], actual_node: str) -> Set[Tuple[str, str]]:
        edges = OrderedSet()
        for node, successors in pdfg.items():
            if node != actual_node:
                for successor in successors:
                    edges.add((node, successor))
        return edges

    def discover_all_splits(self):
        """
        Function to run split discovery on each filtered pdfg node, which has more than one outgoing node
        Returns a
        :return:None
        :rtype: None
        """
        multiple_outgoing_nodes = self.get_nodes_with_multiple_successors()

        for node in self.filtered_graph.keys():
            if node in multiple_outgoing_nodes:
                self.discover_splits_of_node(self.filtered_graph, node, self.concurrent_nodes)
            else:
                self.add_edges_to_bpmn_from_node(node)

    def discover_start_splits(self):
        lst = [self.in_nested_list(self.concurrent_nodes, elem) for elem in self.start_event_set]
        if any(lst):
            # if nodes are concurrent => relation is and
            relation = "and"
        else:
            relation = "xor"
        # print(relation)
        # print("LEN BEFORE", len(self.bpmn_model.edges))
        if len(self.start_event_set) > 1:
            self.bpmn_model.edges.add(('start', relation + 'start'))
            for elem in self.start_event_set:
                self.bpmn_model.edges.add((relation + 'start', elem))
        else:
            for elem in self.start_event_set:
                self.bpmn_model.edges.add(('start', elem))
            # print("EDGE:", relation + 'start', elem)

        # print("LEN AFTER", len(self.bpmn_model.edges))

    def in_nested_list(self, my_list, item):
        """
        Determines if an item is in my_list, even if nested in a lower-level list.
        """
        if item in my_list:
            return True
        else:
            return any(self.in_nested_list(sublist, item) for sublist in my_list if
                       (isinstance(sublist, list) or isinstance(sublist, Tuple)))

    def get_nodes_with_multiple_successors(self) -> Set[str]:
        """
        Function to modify the given split structure in order to introduce and splits. It is base on the  algorithm 4.
        Returns a
        :return:set of nodes which have more than one outgoing edge in filtered pdfg
        :rtype: Set[str]
        """
        multiple_nodes = OrderedSet()
        for node in self.filtered_graph.keys():
            if len(self.filtered_graph[node]) > 1:
                multiple_nodes.add(node)
        return multiple_nodes

    def add_edges_to_bpmn_from_node(self, node):
        for successor in self.filtered_graph[node]:
            self.bpmn_model.edges.add((node, successor))

    def discover_joins(self, dfg):
        java_caller = JavaCaller()
        java_caller.make_call_and_get_formatted_result(self.bpmn_model.get_representation_for_java(),1)
        # TODO: parse output from java to graph


log = SplitMiner("../logs/B4.csv")
log.perform_mining()
print(str(log.bpmn_model.get_representation_for_java()))
#log.bpmn_model.draw()
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
