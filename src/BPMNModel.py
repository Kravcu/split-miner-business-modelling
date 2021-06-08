from typing import Set, Tuple, Dict, List

from Graph import Graph


class BPMNModel:

    def __init__(self, start_events: Set[str], end_events: Set[str], and_events: set, xor_events: set,
                 or_events: set, edges: Set[Tuple[str, str]], tasks: set, self_loops: Set[str]) -> None:
        super().__init__()
        self.xor_events = xor_events
        self.and_events = and_events
        self.end_events = end_events
        self.tasks = tasks
        self.edges = edges
        self.or_events = or_events
        self.start_events = start_events
        self.self_loops = self_loops
        self.graph = Graph()

    def draw(self) -> None:
        """
        Function to draw visual representation of model.
        :return: None
        :rtype: None
        """
        self.graph.add_event("start")
        self.graph.add_end_event("end")
        for edge in self.edges:
            self.graph.add_edge(*edge)
        for node in self.self_loops:
            self.graph.mark_node_as_self_loop(node)
        # for elem in self.start_events:
        #     self.graph.add_edge("start", elem)

        for elem in self.end_events:
            self.graph.add_edge(elem, "end")
        self.graph.save_graph_to_image("test")

    def get_representation_for_java(self) -> Dict[str, List[str]]:
        """
        Function to prepare a representation for calling java RPSTSolver
        :return:
        :rtype:
        """
        repr_dict = {}
        for (source, target) in self.edges:
            if source not in repr_dict:
                repr_dict[source] = [target]
            else:
                if target not in repr_dict[source]:
                    repr_dict[source].append(target)
        return repr_dict
