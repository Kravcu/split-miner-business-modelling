from TTG import TTG


class TTCTree:
    def __init__(self, ttg: TTG) -> None:
        super().__init__()
        self.ttg = ttg
        self.compute_tri_connected_components()
        self.e2o = dict()

    def compute_tri_connected_components(self):
        components = []

        virtual_edge_map = self.initialize_virtual_edge_map()
        virtual_edge_map[self.ttg.back_edge] = True
        assigned_virtual_edge_map = dict.fromkeys(self.ttg.edges)
        is_hidden_map = self.initialize_virtual_edge_map()

    def initialize_virtual_edge_map(self):
        virtual_edge_map = dict()
        for edge in self.ttg.normalized_edges:
            virtual_edge_map[edge] = False
        return virtual_edge_map
