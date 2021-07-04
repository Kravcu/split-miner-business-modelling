import json
from subprocess import Popen, PIPE, STDOUT
from typing import Dict, Set, List
from OrderedSet import OrderedSet

class JavaCaller:

    def make_call_and_get_formatted_result(self, dfg: Dict[str, Set[str]], flag: int) -> List[str]:
        #arg: str = self._prepare_json_for_java_call(dfg)
        stdout: List[str] = self._call_java_and_get_output(dfg, flag)
        return self._format_stdout(stdout)

    def _prepare_json_for_java_call(self, dfg: Dict[str, Set[str]]) -> str:
        """
        Function that takes a DFG dictionary and prepares it to be used as an call argument
        :param dfg: Dictionary with DFG
        :type dfg: Dict[str, Set[str]]
        :return: Formatted json string
        :rtype: str
        """
        dict_copy = {key: list(dfg[key]) for key in dfg}
        return json.dumps(dict_copy)

    def _call_java_and_get_output(self, argument: str, flag:int) -> List[str]:
        """
        Function to call Java JAR file and return captured stdout as list of string.
        :param argument: argument to call JAR file with
        :type argument: str
        :return: List of string from stdout
        :rtype: List[str]
        """
        p = Popen(['java', '-jar', 'RPSTSolver.jar', str(argument), str(flag)], stdout=PIPE, stderr=STDOUT)
        return [item.decode('utf-8').rstrip() for item in p.stdout]

    def _format_stdout(self, stdout: List[str]) -> List[str]:
        stdout = stdout[0].replace("JOIN_join_", "").replace('AND', 'and').replace('XOR', 'xor').replace('OR', 'or')
        stdout = stdout.split(", ")
        edges = OrderedSet()
        for edge in stdout:
            edges.add(tuple(edge.split('->')))
        print(edges)
        return edges

