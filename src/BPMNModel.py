from typing import Set, Dict, Tuple, List


class BPMNModel:

    def __init__(self, start_event: str, end_event: str, and_events: set, xor_events: set,
                 or_events: set, edges: Set[Tuple[str, str]], tasks: set) -> None:
        super().__init__()
        self.xor_events = xor_events
        self.and_events = and_events
        self.end_event = end_event
        self.tasks = tasks
        self.edges = edges
        self.or_events = or_events
        self.start_event = start_event
