from OrderedSet import OrderedSet


def test_update():
    x = OrderedSet()
    x.update([2, 3, 5])
    assert list(x.keys()) == [2, 3, 5]


def test_remove():
    x = OrderedSet([2, 3, 5])
    x.remove(5)
    assert list(x.keys()) == [2, 3]


def test_add():
    x = OrderedSet([2, 3, 5])
    x.add(6)
    assert list(x.keys()) == [2, 3, 5, 6]


def test_discard():
    x = OrderedSet([2, 3, 5])
    x.remove(5)
    assert list(x.keys()) == [2, 3]


def test_union():
    x = OrderedSet([2, 3])
    y = OrderedSet([5, 6])
    assert list((x | y).keys()) == [2, 3, 5, 6]
    assert list(x.union(y).keys()) == [2, 3, 5, 6]


def test_difference_update():
    x = OrderedSet([2, 3, 5])
    y = OrderedSet([2, 5, 6])
    x.difference_update(y)
    assert list(x.keys()) == [3]


def test_intersection():
    x = OrderedSet([2, 3, 5])
    y = OrderedSet([2, 5, 6])
    assert list((x & y).keys()) == [2, 5]
    assert list(x.intersection(y).keys()) == [2, 5]
