from cards98 import MapIndexesToNum
import pytest


def test_s2_num():
    t1 = MapIndexesToNum(2, 2)
    assert 0 == t1.get_num(0, 0)
    assert 1 == t1.get_num(1, 0)
    assert 2 == t1.get_num(0, 1)
    assert 3 == t1.get_num(1, 1)
    with pytest.raises(Exception):
        t1.get_num(2, 2)


def test_s2_map():
    t1 = MapIndexesToNum(2, 2)
    assert (0, 0) == t1.get_map(0)
    assert (1, 0) == t1.get_map(1)
    assert (0, 1) == t1.get_map(2)
    assert (1, 1) == t1.get_map(3)
    with pytest.raises(Exception):
        t1.get_map(4)


def test_s3_num():
    t1 = MapIndexesToNum(3, 3)
    assert 0 == t1.get_num(0, 0)
    assert 1 == t1.get_num(1, 0)
    assert 2 == t1.get_num(2, 0)
    assert 3 == t1.get_num(0, 1)
    assert 4 == t1.get_num(1, 1)
    assert 5 == t1.get_num(2, 1)
    assert 6 == t1.get_num(0, 2)
    assert 7 == t1.get_num(1, 2)
    assert 8 == t1.get_num(2, 2)
    with pytest.raises(Exception):
        t1.get_num(3, 3)


def test_s3_map():
    t1 = MapIndexesToNum(3, 3)
    assert (0, 0) == t1.get_map(0)
    assert (1, 0) == t1.get_map(1)
    assert (2, 0) == t1.get_map(2)
    assert (0, 1) == t1.get_map(3)
    assert (1, 1) == t1.get_map(4)
    assert (2, 1) == t1.get_map(5)
    assert (0, 2) == t1.get_map(6)
    assert (1, 2) == t1.get_map(7)
    assert (2, 2) == t1.get_map(8)
    with pytest.raises(Exception):
        t1.get_map(9)


def test_bigger_num():
    t1 = MapIndexesToNum(2, 2, 2)
    assert 0 == t1.get_num(0, 0, 0)
    assert 1 == t1.get_num(1, 0, 0)
    assert 2 == t1.get_num(0, 1, 0)
    assert 3 == t1.get_num(1, 1, 0)
    assert 4 == t1.get_num(0, 0, 1)
    assert 5 == t1.get_num(1, 0, 1)
    assert 6 == t1.get_num(0, 1, 1)
    assert 7 == t1.get_num(1, 1, 1)
    with pytest.raises(Exception):
        t1.get_num(3, 3)
    with pytest.raises(Exception):
        t1.get_num(2, 2, 3)


def test_s3_map():
    t1 = MapIndexesToNum(2, 2, 2)
    assert (0, 0, 0) == t1.get_map(0)
    assert (1, 0, 0) == t1.get_map(1)
    assert (0, 1, 0) == t1.get_map(2)
    assert (1, 1, 0) == t1.get_map(3)
    assert (0, 0, 1) == t1.get_map(4)
    assert (1, 0, 1) == t1.get_map(5)
    assert (0, 1, 1) == t1.get_map(6)
    assert (1, 1, 1) == t1.get_map(7)

    with pytest.raises(Exception):
        t1.get_map(8)


def test_size_10_num():
    t1 = MapIndexesToNum(10, 10)
    assert 0 == t1.get_num(0, 0)
    assert 10 == t1.get_num(0, 1)
    assert 11 == t1.get_num(1, 1)


def test_size_10_map():
    t1 = MapIndexesToNum(10, 10)
    assert (0, 0) == t1.get_map(0)
    assert (0, 1) == t1.get_map(10)
    assert (1, 1) == t1.get_map(11)


def test_universal_1():
    t1 = MapIndexesToNum(135, 30, 525)

    num = 40
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 140
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 1240
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 5440
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 4150
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 4430
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 123
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 4340
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
    num = 4370
    indx = t1.get_map(num)
    assert num == t1.get_num(indx)
