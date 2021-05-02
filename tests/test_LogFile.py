import pytest
from pathlib import Path
from src.LogFile import LogFile, LogType
from pandas.util.testing import assert_frame_equal


@pytest.mark.parametrize("test", [
    ("../logs/raw/B1.csv"),
    ("../logs/raw/B1.xes"),

])
def test_parse_log_file(test):
    obj1 = LogFile(test)
    obj2 = LogFile(Path(test))
    assert_frame_equal(obj1.traces, obj2.traces), "Results from same file in different format is not identical"


@pytest.mark.parametrize("test,expected", [
    ("../logs/raw/B1.csv", LogType.CSV),
    ("../logs/raw/B1.xes", LogType.XES),
    (Path("../logs/raw/B1.csv"), LogType.CSV),
    (Path("../logs/raw/B1.xes"), LogType.XES)

])
def test_get_log_type(test, expected):
    obj = LogFile(test)
    assert obj.log_type == expected, "Wrong LogType assigned to log file"


@pytest.mark.parametrize("test,expected", [
    ("xxx.csv", Path("xxx.csv")),
    ("xxx.xes", Path("xxx.xes")),
    (Path("xxx.csv"), Path("xxx.csv")),
    (Path("xxx.xes"), Path("xxx.xes"))

])
def test_get_path_as_object(test, expected):
    log = LogFile.get_path_as_object(test)
    assert isinstance(log, Path), "Returned object is not instance of Path"
    assert log == expected, "Objects do not point to same element"
