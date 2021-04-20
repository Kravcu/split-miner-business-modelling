import ast
import copy
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
import pandas as pd
from more_itertools import pairwise
from typing import Set, Dict, Tuple, List
from itertools import combinations
import numpy as np

class LogType(Enum):
    CSV = 0
    X = 1  # kluza wspominał coś o jakimś drugim formacie logów, dodałem od razu bo pasowałoby go też mieć


class SimpleLog:
    USED_COLUMNS = ['trace']

    def __init__(self, path):

        self.path: Path = Path(path)
        if not os.path.isfile(path):
            raise Exception(f"Specified path is not a file or does not exist: {path}")
        self.log_type: LogType = LogType.CSV if '.csv' in path else LogType.X if '.x' in path else None
        self.df: pd.DataFrame = self.parse_into_df()
        self.direct_follows_graph, self.start_event_set, self.end_event_set = self.get_dfg()
        self.self_loops = self.find_self_loops()
        self.short_loops = self.find_short_loops()
        self.arc_frequency = self.count_arc_frequency()
        self.concurrent_nodes = set()
        self.concurrent_nodes.add("Not performed")
        self.pdfg = dict()
        self.filtered_graph = dict()

    def parse_into_df(self) -> pd.DataFrame:
        """
            Function to parse the specified file into the dataframe.
            I'm assuming that this class is fed with preprocessed data so the .csv contains traces only
        """
        if self.log_type == LogType.CSV:
            df: pd.DataFrame = pd.read_csv(self.path, header=0, converters={'trace': ast.literal_eval})
            try:
                if not all(elem in df.columns for elem in self.USED_COLUMNS):
                    df = self.get_traces(self.path)
                df = df[self.USED_COLUMNS]
            except KeyError as e:
                raise Exception("Not all needed columns are present in the specified log file.\n"
                                f"Needed: {self.USED_COLUMNS}\n"
                                f"Traceback: {str(e)}")
            return df
        elif self.log_type == LogType.X:
            raise NotImplementedError

    def get_dfg(self) -> Tuple[Dict[str, set], set, set]:
        """
            Function to get DFG graph, and start/end event sets from DataFrame containing traces
        :return: direct_follows_graph, start_event_set, end_event_set
        :rtype: Tuple[Dict[str, set], set, set]
        """
        start_event_set = set()
        end_event_set = set()
        direct_follows_graph = dict()
        for index, row in self.df.iterrows():
            if row['trace'][0] not in start_event_set:
                start_event_set.add(row['trace'][0])
            if row['trace'][-1] not in end_event_set:
                end_event_set.add(row['trace'][-1])
            for ev_i, ev_j in pairwise(row['trace']):
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
            A short loop is a pattern {a,b,a} in a trace, where a,b - events
        :return: Set containing pairs (a,b)
        :rtype: Set[Tuple[str, str]]
        """
        short_loops: Set[Tuple[str, str]] = set()
        # getting trace column, transforming it to tuples, and getting unique traces
        traces: List[List[str]] = self.df['trace'].transform(tuple).unique()
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
        traces: List[List[str]] = self.df['trace'].tolist()
        for trace in traces:
            for source, target in zip(trace, trace[1:]):
                arc_frequency[(source, target)] += 1
        return arc_frequency

    def remove_self_short_loops_from_dfg(self) -> None:
        """
        Function to remove self and short loops from dfg. Returns a
        :return: None
        :rtype: None
        """
        for self_loop_node in self.self_loops:
            self.direct_follows_graph[self_loop_node].remove(self_loop_node)

        for node_a, node_b in self.short_loops:
            self.direct_follows_graph[node_a].remove(node_b)

    def get_traces(self, input_path, case_column='Case ID',
                            activity_column='Activity', start_column='Start Timestamp',
                            output_column='trace') -> pd.DataFrame:
        """
        Function to generate list of traces from csv log file.
         Input csv has to have Case ID,Activity,Start Timestamp columns. Returns a
        :return: string which represents file name for generated trace file
        :rtype: pd.DataFrame
        """
        df = pd.read_csv(input_path)
        dfs = df[[case_column, activity_column, start_column]]
        dfs = dfs.sort_values(by=[case_column, start_column]) \
            .groupby([case_column]) \
            .agg({activity_column: ';'.join})
        # dfs.rename(columns={activity_column: output_column}, inplace=True)
        dfs[output_column] = [trace.split(';') for trace in dfs[activity_column]]
        return dfs[[output_column]]

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
        concurrent_nodes = set()
        arc_frequency: Dict[Tuple[str, str], int] = self.count_arc_frequency()
        for node_a, node_b in combinations(self.direct_follows_graph.keys(), 2): # check time complexity
            print(node_a, node_b)
            if node_b in self.direct_follows_graph[node_a] and node_a in self.direct_follows_graph[node_b]:
                if ((node_a, node_b) not in self.short_loops) and ((node_b, node_a) not in self.short_loops):
                    if (abs(arc_frequency[(node_a, node_b)] -
                            arc_frequency[(node_b, node_a)]))/(arc_frequency[(node_a, node_b)] +
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

    def filter_graph(self, pdfg: Dict[str, set], eta):
        
        most_frequent_edges = self.get_most_frequent_edge_for_each_node(pdfg)
        frequency_threshold = self.get_percentile_frequency(most_frequent_edges, eta)
        most_frequent_edges = self.add_edges_with_greater_threshold(frequency_threshold,
                                                                    most_frequent_edges,
                                                                    pdfg)
        filtered_edges = set()
        filtered_graph: Dict[str, set] = dict()
        for node in pdfg.keys():
            filtered_graph[node] = set()
        while len(most_frequent_edges) > 0:
            edge = self.get_most_frequent_edge_from_set(most_frequent_edges)
            node_a, node_b = edge
            if (self.arc_frequency[edge] > frequency_threshold
                    or len(filtered_graph[node_a]) == 0
                    or len(self.get_predecessors(filtered_graph, node_b)) == 0):
                filtered_edges.add(edge)
                filtered_graph[node_a].add(node_b)
            most_frequent_edges.remove(edge)
        self.filtered_graph = filtered_graph
        
      

    def get_most_frequent_edge_for_each_node(self, graph: Dict[str, set]) -> Set[Tuple[str, str]]:
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
                outgoing_edges_freq[(graph_node, outgoing_node)] = self.arc_frequency[(graph_node, outgoing_node)]

            max_outgoing_edge = max(outgoing_edges_freq, key=outgoing_edges_freq.get)
            most_frequent_edges.add(max_outgoing_edge)

            # to do predecessors
            predecessors = self.get_predecessors(graph, graph_node)
            incoming_edges_freq: Dict[Tuple[str, str], int] = dict()
            for predecessor in predecessors:
                incoming_edges_freq[predecessor, graph_node] = self.arc_frequency[(predecessor, graph_node)]
            max_incoming_edge = max(incoming_edges_freq, key=incoming_edges_freq.get)
            most_frequent_edges.add(max_incoming_edge)

        return most_frequent_edges

    def get_predecessors(self, graph: Dict[str, set], node: str) -> Set[str]:
        predecessors = set()
        for graph_node in graph.keys():
            if node in graph[graph_node]:
                predecessors.add(graph_node)
        return predecessors
    
    def get_percentile_frequency(self, most_frequent_edges: Set[Tuple[str, str]], eta) -> float:
        """
        Function to compute frequency threshold based on the most frequent edges and percentile eta
        Returns a
        :return: frequency threshold
        :rtype: float
        """
        frequencies = []
        for node_a, node_b in most_frequent_edges:
            frequencies.append(self.arc_frequency[(node_a, node_b)])
        return np.percentile(np.array(frequencies), eta)
    
    def add_edges_with_greater_threshold(self, threshold, actual_edges: Set[Tuple[str, str]],
                                         graph: Dict[str, set]) -> Set[Tuple[str, str]]:
        """
        Function to add all edges with the frequency greater than threshold to the actual set of max edges
        Returns a
        :return: updated max edges
        :rtype: Set[Tuple[str, str]]
        """
        for node, succ_set in graph.items():
            for succ in succ_set:
                if self.arc_frequency[(node, succ)] > threshold:
                    actual_edges.add((node, succ))
        return actual_edges
    
    def get_most_frequent_edge_from_set(self, edges: Set[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Function to find the most frequent edge from a given set of edges
        Returns a
        :return: edge with the highest frequency
        :rtype: Tuple[str, str]
        """
        frequencies = dict()
        for edge in edges:
            frequencies[edge] = self.arc_frequency[edge]
        return max(frequencies, key=frequencies.get)

log = SimpleLog("../logs/preprocessed/B1.csv")
log.perform_mining()
print(log.direct_follows_graph)
print("finished")

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

