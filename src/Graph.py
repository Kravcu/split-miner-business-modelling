from typing import final, Iterable

import pygraphviz as pgv


class Graph(pgv.AGraph):
    NODE_STYLE: final = {'style': "rounded,filled", 'fillcolor': "#ffffcc", 'shape': "circle"}
    GATEWAY_STYLE: final = {'style': "filled,bold,solid",
                            'fillcolor': "#fffffc",
                            'shape': "diamond",
                            'width': ".7", 'height': ".7",
                            'fixedsize': "true",
                            'fontsize': "45", }

    def __init__(self, *args):
        super(Graph, self).__init__(strict=False, directed=True, *args)
        self.graph_attr['rankdir'] = 'LR'
        self.node_attr['shape'] = 'Mrecord'
        self.graph_attr['splines'] = 'ortho'
        self.graph_attr['nodesep'] = '0.8'
        self.edge_attr.update(penwidth='2')

    def add_event(self, name, **kwargs) -> None:
        super(Graph, self).add_node(name, **kwargs, **self.NODE_STYLE, label="", penwidth='3')

    def add_end_event(self, name, **kwargs) -> None:
        super(Graph, self).add_node(name, **kwargs, **self.NODE_STYLE, label="", penwidth='6')

    def _add_and_gateway(self, *args, **kwargs) -> None:
        super(Graph, self).add_node(*args, **kwargs, **self.GATEWAY_STYLE, label="+", )

    def _add_xor_gateway(self, *args, **kwargs) -> None:
        super(Graph, self).add_node(*args, **kwargs, **self.GATEWAY_STYLE, label="")

    def _add_or_gateway(self, *args, **kwargs) -> None:
        super(Graph, self).add_node(*args, **kwargs, **self.GATEWAY_STYLE, label="○")
    def add_custom_node(self,name,label,*args,**kwargs)-> None:
        node_label = ""
        if label == 'and':
            node_label = "+"
        elif label == 'xor':
            node_label = "×"
        elif label == 'or':
            node_label = "○"
        else:
            raise NotImplementedError
        super(Graph, self).add_node(name, **kwargs, **self.GATEWAY_STYLE, label=node_label)
    def add_edge(self, u: str, v: str = None, key=None, **kwargs) -> None:
        """
        Add edge between nodes.
        If the nodes u and v are not in the graph they will added.

        If u and v are not strings, conversion to a string will be attempted.

        The optional key argument allows assignment of a key to the
        edge.  This is especially useful to distinguish between
        parallel edges in multi-edge graphs (strict=False).
        :param u: node in graph
        :type u: str
        :param v: node in graph
        :type v: str
        :param key: key for the edge id
        :type key: str
        :param kwargs:
        :type kwargs: Dict
        :return: None
        :rtype: None
        """
        try:
            super(Graph, self).get_node(u)
        except KeyError:
            if 'and' in u:
                self.add_custom_node(u, 'and')
            elif 'xor' in u:
                self.add_custom_node(u, 'xor')
            elif 'or' in u:
                self.add_custom_node(u, 'or')
        try:
            super(Graph, self).get_node(v)
        except KeyError:
            if 'and' in v:
                self.add_custom_node(v, 'and')
            elif 'xor' in v:
                self.add_custom_node(v, 'xor')
            elif 'or' in v:
                self.add_custom_node(v, 'or')

        super(Graph, self).add_edge(u, v, key, **kwargs)

    def add_or_split(self, source: str, targets: Iterable[str], *args, **kwargs):
        """
        Adds OR split gate
        :param source: source of the gate
        :type source: str
        :param targets: targets of the gate
        :type targets: Iterable[str]
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'ORs ' + str(source) + '->' + str(targets)
        self._add_or_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(source, gateway)
        for target in targets:
            super(Graph, self).add_edge(gateway, target)
        return gateway

    def add_or_merge(self, sources: Iterable[str], target: str, *args, **kwargs):
        """
        Adds OR merge gate
        :param sources: sources of the gate
        :type sources: Iterable[str]
        :param target: target of the gate
        :type target: str
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'ORm ' + str(sources) + '->' + str(target)
        self._add_or_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(gateway, target)
        for source in sources:
            super(Graph, self).add_edge(source, gateway)
        return gateway

    def add_and_split(self, source: str, targets: Iterable[str], *args, **kwargs) -> str:
        """
        Adds AND split gate
        :param source: source of the gate
        :type source: str
        :param targets: targets of the gate
        :type targets: Iterable[str]
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'ANDs ' + str(source) + '->' + str(targets)
        self._add_and_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(source, gateway)
        for target in targets:
            super(Graph, self).add_edge(gateway, target)
        return gateway

    def add_xor_split(self, source: str, targets: Iterable[str], *args, **kwargs) -> str:
        """
        Adds XOR split gate
        :param source: source of the gate
        :type source: str
        :param targets: targets of the gate
        :type targets: Iterable[str]
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'XORs ' + str(source) + '->' + str(targets)
        self._add_xor_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(source, gateway)
        for target in targets:
            super(Graph, self).add_edge(gateway, target)
        return gateway

    def add_and_merge(self, sources, target, *args, **kwargs) -> str:
        """
        Adds AND merge gate
        :param sources: sources of the gate
        :type sources: Iterable[str]
        :param target: target of the gate
        :type target: str
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'ANDm ' + str(sources) + '->' + str(target)
        self._add_and_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(gateway, target)
        for source in sources:
            super(Graph, self).add_edge(source, gateway)
        return gateway

    def add_xor_merge(self, sources, target, *args, **kwargs) -> str:
        """
        Adds XOR merge gate
        :param sources: sources of the gate
        :type sources: Iterable[str]
        :param target: target of the gate
        :type target: str
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: handle to added gateway
        :rtype: str
        """
        gateway = 'XORm ' + str(sources) + '->' + str(target)
        self._add_xor_gateway(gateway, *args, **kwargs)
        super(Graph, self).add_edge(gateway, target)
        for source in sources:
            super(Graph, self).add_edge(source, gateway)
        return gateway

    def mark_node_as_self_loop(self, node) -> None:
        """
        Adds self loop marking to node. This function shall be called as last as it changes nodes label.
        :param node: Node to be marked
        :type node: str
        :return: None
        :rtype: None
        """
        n = super(Graph, self).get_node(node)
        n.attr['label'] = node + "\u000A\u21ba"

    def save_graph_to_image(self, name: str, extension: str = 'svg', *args, **kwargs) -> None:
        """
        Saves graph to image file
        :param name: file name
        :type name: str
        :param extension: file extension without dot
        :type extension: str
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: None
        :rtype: None
        """
        filename = f'{name}.{extension}'
        super(Graph, self).draw(filename, prog='dot', *args, **kwargs)
        print(f"Saved graph to file: '{filename}'")

    def save_graph_to_dot_file(self, name: str) -> None:
        """
        Saves graph to dot file
        :param name: file name
        :type name: str
        :return: None
        :rtype: None
        """
        filename = f'{name}.dot'
        super(Graph, self).write(filename)
        print(f"Saved graph to file: '{filename}'")
