import collections
import itertools as it


class OrderedSet(collections.OrderedDict, collections.MutableSet):
    def __init__(self, *args):
        super().__init__()
        if args:
            for arg in args:
                self.update(arg)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def remove(self, elem):
        self.discard(elem)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def union(self, *sets):
        cls = self.__class__ if isinstance(self, OrderedSet) else OrderedSet
        containers = map(list, it.chain([self], sets))
        items = it.chain.from_iterable(containers)
        return cls(items)

    def difference_update(self, *sets):
        items_to_remove = set()  # type: Set[T]
        for other in sets:
            items_as_set = set(other)  # type: Set[T]
            items_to_remove |= items_as_set
        for item in items_to_remove:
            self.discard(item)

    def intersection(self, *sets):
        cls = self.__class__ if isinstance(self, OrderedSet) else OrderedSet
        if sets:
            common = set.intersection(*map(set, sets))
            items = (item for item in self if item in common)
        else:
            items = self
        return cls(items)

    def __iter__(self):
        # Traverse the linked list in order.
        for key in self.keys():
            yield key

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    def __eq__(self, other):
        return len(self) == len(other) and sorted(list(self.keys())) == sorted(list(other.keys()))

    # def __getitem__(self, k):
    #     return list(self.keys())[k]
