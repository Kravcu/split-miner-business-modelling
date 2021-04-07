import os
from enum import Enum
from pathlib import Path
import pandas as pd
from more_itertools import pairwise
from typing import Set, Dict, Tuple


class LogType(Enum):
    CSV = 0
    X = 1  # kluza wspominał coś o jakimś drugim formacie logów, dodałem od razu bo pasowałoby go też mieć


class SimpleLog:
    USED_COLUMNS = ['trace']

    def __init__(self, path):

        self.path: Path = Path(path)
        if not os.path.isfile(path):
            print(self.path)
            raise Exception("Specified path is not a file or does not exist")
        self.log_type: LogType = None
        if '.csv' in path:
            self.log_type = LogType.CSV
        elif '.x' in path:
            self.log_type = LogType.X
        self.df: pd.DataFrame = self.parse_into_df()
        self.direct_follows_graph, self.start_event_set, self.end_event_set = self.get_dfg()

    def parse_into_df(self) -> pd.DataFrame:
        """
            Function to parse the specified file into the dataframe.
            I'm assuming that this class is fed with preprocessed data so the .csv contains traces only
        """
        if self.log_type == LogType.CSV:
            df: pd.DataFrame = pd.read_csv(self.path, header=0)
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
        :rtype:
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
        self_loops: Set[str] = set()
        for event in self.direct_follows_graph:
            if event in self.direct_follows_graph[event]:
                self_loops.add(event)
        return self_loops


log = SimpleLog("../logs/preprocessed/phone_trace_only.csv")
log.parse_into_df()
print(log.df)
