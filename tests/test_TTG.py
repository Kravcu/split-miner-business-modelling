import unittest

from BPMNModel import BPMNModel
from TTG import TTG


class TestTTG(unittest.TestCase):
    def setUp(self) -> None:
        start_events = {"s"}
        end_events = {"t"}
        and_events = set()
        or_events = set()
        xor_events = set()
        edges = {("s", "t"), ("s", "u"), ("u", "v"), ("v", "u"), ("u", "t"), ("v", "t")}
        tasks = {"s", "u", "v", "t"}
        self_loops = set()
        bpmn = BPMNModel(start_events, end_events, and_events, or_events, xor_events, edges,
                         tasks, self_loops)

        self.ttg = TTG(bpmn)

    def test_normalize_TTG(self):
        expected_normalized_tasks = {"s", "split1u", "split2u", "v", "t"}
        # expected_normalized_edge_map = {'edge2': ('split2u', 'v'), 'edge3': ('split2u', 't'),
        #                                 'edge0': ('s', 'split1u'), 'edge4': ('v', 'split1u'),
        #                                 'edgeusplit': ('split1u', 'split2u'), 'edge1': ('s', 't'),
        #                                 'back_edge': ('s', 't'), 'edge5': ('v', 't')}
        self.ttg.normalize_TTG()

        self.assertEqual(expected_normalized_tasks, self.ttg.normalized_tasks)
        #self.assertEqual(expected_normalized_edge_map, self.ttg.normalized_edges_map)
