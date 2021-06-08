from enum import Enum
from pathlib import Path
import pandas as pd
from typing import Union, final
from opyenxes.data_in.XUniversalParser import XUniversalParser


class LogType(Enum):
    CSV = 0
    XES = 1


class LogFile:
    """
    Class to handle all interaction with reading log file into unified Dataframe
    """

    def __init__(self, path_to_log: Union[str, Path], case_column: str = 'Case ID',
                 activity_column: str = 'Activity', start_column: str = 'Start Timestamp',
                 output_column: str = 'trace'):
        self._path: Path = self.get_path_as_object(path_to_log)
        self.check_if_valid_log_file()
        self._log_type: LogType = self.get_log_type(self._path)
        self._case_column: str = case_column
        self._activity_column: str = activity_column
        self._start_column: str = start_column
        self._output_column: str = output_column
        self._traces: pd.DataFrame = self.parse_log_file()

    def check_if_valid_log_file(self) -> None:
        """
        Checks if path to log file points to an existing file, if not raises exception.
        :return: None
        :rtype: None
        :raise AttributeError: if path does not point to file of file does not exist
        """
        if not self._path.is_file() or not self._path.exists():
            raise AttributeError("Specified path does not lead to a file")

    def parse_log_file(self) -> pd.DataFrame:
        """
        Function for users to parse log file into uniform Dataframe
        :return: Dataframe containing traces indexed by Case ID
        :rtype: pd.DataFrame
        """
        if self._log_type == LogType.CSV:
            return self._get_traces_from_csv()
        elif self._log_type == LogType.XES:
            return self._get_traces_from_xes()
        else:
            raise NotImplementedError

    @staticmethod
    def get_log_type(path: Path) -> LogType:
        """
        Function to check which type of log was provided to the constructor
        :param path: Object representing path to log file
        :type path: Path
        :return: Enum value representing log type
        :rtype: LogType
        :raise NotImplementedError: if log format is not supported
        """
        if '.csv' in str(path):
            return LogType.CSV
        elif '.xes' in str(path):
            return LogType.XES
        else:
            raise NotImplementedError

    @staticmethod
    def get_path_as_object(path_to_log: Union[str, Path]) -> Path:
        """
        Function to get log file path as Path instance.
        :param path_to_log: str path to log or Path Instance
        :type path_to_log: Union[str, Path]
        :return: Path instance of path_to_log
        :rtype: Path
        """
        if isinstance(path_to_log, Path):
            return path_to_log
        elif isinstance(path_to_log, str):
            return Path(path_to_log)

    def _get_traces_from_csv(self) -> pd.DataFrame:
        """
        Function to aggregate events into traces.
        :return: Dataframe with column _output_column indexed by _case_column
        :rtype: pd.DataFrame
        """
        df = pd.read_csv(self._path)
        dfs = df[[self._case_column, self._activity_column, self._start_column]]
        dfs = dfs.sort_values(by=[self._case_column, self._start_column]) \
            .groupby([self._case_column]) \
            .agg({self._activity_column: ';'.join})
        dfs[self._output_column] = [trace.split(';') for trace in dfs[self._activity_column]]
        return dfs[[self._output_column]]

    def _get_traces_from_xes(self) -> pd.DataFrame:
        """
        Function to aggregate events into traces.
        :return: Dataframe with column _output_column indexed by _case_column
        :rtype: pd.DataFrame
        """
        CASE_ID: final = 'concept:name'
        traces_list = []
        indexes_list = []
        with open(self._path) as log_file:
            loaded_log = XUniversalParser().parse(log_file)[0]
        for trace in loaded_log:
            event_list = []
            for event in trace:
                event_list.append(str(event.get_attributes()[CASE_ID]))
            traces_list.append(event_list)
            indexes_list.append(trace.get_attributes()[CASE_ID])
        dfs = pd.DataFrame({self._output_column: traces_list}, index=indexes_list)
        dfs.index.rename(self._case_column, inplace=True)
        return dfs[[self._output_column]]

    @property
    def traces(self) -> pd.DataFrame:
        return self._traces

    @property
    def case_column(self) -> str:
        return self._case_column

    @property
    def activity_column(self) -> str:
        return self._activity_column

    @property
    def start_column(self) -> str:
        return self._start_column

    @property
    def output_column(self) -> str:
        return self._output_column

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def log_type(self) -> LogType:
        return self._log_type
