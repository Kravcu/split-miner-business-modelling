import ast
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
import pandas as pd
from more_itertools import pairwise
from typing import Set, Dict, Tuple, List


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

    def parse_into_df(self) -> pd.DataFrame:
        """
            Function to parse the specified file into the dataframe.
            I'm assuming that this class is fed with preprocessed data so the .csv contains traces only
        """
        if self.log_type == LogType.CSV:
            df: pd.DataFrame = pd.read_csv(self.path, header=0, converters={'trace': ast.literal_eval})
            try:
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
                if source == target:
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


log = SimpleLog("../logs/preprocessed/phone_trace_only.csv")
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
