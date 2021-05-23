from typing import Set, Dict, Tuple, List

from BPMNModel import BPMNModel


class TTG:
    def __init__(self, bpmn: BPMNModel) -> None:
        super().__init__()
        self.tasks = bpmn.tasks
        self.edges, self.edges_map = self.init_TTG_from_BPMN(bpmn)
        self.normalized_tasks = set()
        self.normalized_edges = set()
        self.normalized_edges_map = dict()

    def init_TTG_from_BPMN(self, bpmn: BPMNModel) -> Tuple[Set[str], Dict[str, Tuple[str, str]]]:
        edges = set()
        edges_map = dict()
        for index, edge in enumerate(bpmn.edges):
            edge_name = f"edge{index}"
            edges.add(edge_name)
            edges_map[edge_name] = edge
        return edges, edges_map

    def normalize_TTG(self):
        nodes_to_split = self.get_nodes_to_split()
        for task in self.tasks:
            if task in nodes_to_split:
                self.split_task(task)
            else:
                self.add_task_to_normalized_form(task, nodes_to_split)

    def get_task_predecessors_edges(self, task) -> Dict[str, Tuple[str, str]]:
        predecessors = dict()
        for edge, nodes in self.edges_map.items():
            (node_a, node_b) = nodes
            if node_b == task:
                predecessors[edge] = nodes
        return predecessors

    def get_task_successors_edges(self, task) -> Dict[str, Tuple[str, str]]:
        successors = dict()
        for edge, nodes in self.edges_map.items():
            (node_a, node_b) = nodes
            if node_a == task:
                successors[edge] = nodes
        return successors

    def split_task(self, task):
        first_task_split = f"split1{task}"
        second_task_split = f"split2{task}"
        self.normalized_tasks.add(first_task_split)
        self.normalized_tasks.add(second_task_split)

        successors_edges = self.get_task_successors_edges(task)
        predecessor_edges = self.get_task_predecessors_edges(task)

        for edge, nodes in successors_edges.items():
            (node_a, node_b) = nodes
            self.normalized_edges.add(edge)
            self.normalized_edges_map[edge] = (second_task_split, node_b)

        for edge, nodes in predecessor_edges.items():
            (node_a, node_b) = nodes
            self.normalized_edges.add(edge)
            self.normalized_edges_map[edge] = (node_a, first_task_split)

        split_nodes_edge = f"edge{task}split"
        self.normalized_edges.add(split_nodes_edge)
        self.normalized_edges_map[split_nodes_edge] = (first_task_split, second_task_split)

    def add_task_to_normalized_form(self, task, nodes_to_split: Set[str]):
        self.normalized_tasks.add(task)
        successors_edges = self.get_task_successors_edges_without_nodes_to_split(task, nodes_to_split)
        predecessor_edges = self.get_task_predecessors_edges_edges_without_nodes_to_split(task, nodes_to_split)

        self.normalized_edges.update(successors_edges.keys())
        self.normalized_edges.update(predecessor_edges.keys())

        self.normalized_edges_map.update(successors_edges)
        self.normalized_edges_map.update(predecessor_edges)

    def get_nodes_to_split(self) -> Set[str]:
        nodes_to_split = set()
        for task in self.tasks:
            if len(self.get_task_predecessors_edges(task)) > 1 and len(self.get_task_successors_edges(task)) > 1:
                nodes_to_split.add(task)
        return nodes_to_split

    def get_task_successors_edges_without_nodes_to_split(self, task, nodes_to_split):
        successors = dict()
        for edge, nodes in self.edges_map.items():
            (node_a, node_b) = nodes
            if node_a == task and (node_a not in nodes_to_split and node_b not in nodes_to_split):
                successors[edge] = nodes
        return successors

    def get_task_predecessors_edges_edges_without_nodes_to_split(self, task, nodes_to_split):
        predecessors = dict()
        for edge, nodes in self.edges_map.items():
            (node_a, node_b) = nodes
            if node_b == task and (node_a not in nodes_to_split and node_b not in nodes_to_split):
                predecessors[edge] = nodes
        return predecessors





