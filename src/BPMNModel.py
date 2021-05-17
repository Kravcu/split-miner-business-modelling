from typing import Set, Tuple

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

    def draw(self):

        for edge in self.edges:
            self.graph.add_edge(*edge)
        for node in self.self_loops:
            self.graph.mark_node_as_self_loop(node)
        self.graph.add_event("start")
        self.graph.add_end_event("end")
        for elem in self.start_events:
            self.graph.add_edge("start", elem)
        for elem in self.end_events:
            self.graph.add_edge(elem,"end")
        self.graph.save_graph_to_image("test")
