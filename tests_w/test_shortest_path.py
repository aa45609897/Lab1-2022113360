import pytest
import heapq
import math
from typing import Dict, Tuple, List
import math

from code_ori.lab1 import calcShortestPath  # 替换为实际模块路径
from code_ori.lib import find_shortest_path
from code_ori.lib import find_all_shortest_paths

@pytest.fixture
def sample_graph2():
    return {
        'a': {'b': 1, 'c': 4},
        'b': {'c': 2, 'd': 5},
        'c': {'d': 1},
        'd': {},
        'e': {'f': 1},  # 孤立的子图
        'f': {}
    }

def test_single_node_graph():
    """测试只有一个节点的图"""
    graph = {'a': {}}
    result = find_all_shortest_paths(graph, 'a')
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == {}

def test_unreachable_nodes(sample_graph2):
    """测试不可达节点"""
    result = find_all_shortest_paths(sample_graph2, 'a')
    assert 'e' not in result
    assert 'f' not in result
    
    result = find_all_shortest_paths(sample_graph2, 'e')
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == {'f': (['e', 'f'], 1)}

def test_empty_graph():
    """测试空图"""
    result = find_all_shortest_paths({}, 'a')
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == {}

def test_multiple_shortest_paths():
    """测试存在多条最短路径的情况"""
    graph = {
        'a': {'b': 1, 'c': 1},
        'b': {'d': 1},
        'c': {'d': 1},
        'd': {}
    }
    result = find_all_shortest_paths(graph, 'a')
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    # 我们的实现会返回其中一条路径
    assert result['d'][1] == 2  # 路径长度应为2
    assert len(result['d'][0]) == 3  # 路径节点数应为3

@pytest.fixture
def sample_graph() -> dict[str, dict[str, int]]:
    """提供测试用的有向图"""
    return {
        "a": {"b": 1, "c": 4},
        "b": {"c": 2, "d": 5},
        "c": {"d": 1},
        "d": {},
        "e": {}  # 孤立节点
    }

def test_word_not_in_graph(sample_graph):
    """测试起始词不在图中的情况"""
    result = find_shortest_path(sample_graph, "X", "d")
    print("feng1")
    print(result)
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == (None, float('inf'))

def test_target_unreachable(sample_graph):
    """测试目标词不可达的情况"""
    result = find_shortest_path(sample_graph, "e", "a")
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == (None, float('inf'))

def test_same_start_and_end():
    """测试起点和终点相同"""
    graph = {"a": {"b": 1}, "b": {}}
    result = find_shortest_path(graph, "a", "a")
    with open("writlog.txt",'a') as f:
        f.write(str(result))
        f.write("\n")
    assert result == (["a"], 0)

def test_multiple_paths_with_different_weights():
    """测试权重影响路径选择"""
    graph = {
        "a": {"b": 1, "c": 3},
        "b": {"d": 5},
        "c": {"d": 1},
        "d": {}
    }
    path, dist = find_shortest_path(graph, "a", "d")
    with open("writlog.txt",'a') as f:
        f.write(str(path))
        f.write(str(dist))
        f.write("\n")
    assert path == ["a", "c", "d"]
    assert dist == 4  # a→c(3) + c→d(1) 比 a→b→d(6)更短

def test_heap_optimization(monkeypatch):
    """测试堆优化提前终止"""
    graph = {
        "a": {"b": 1, "c": 4},
        "b": {"c": 1},
        "c": {"d": 1},
        "d": {}
    }
    
    # 模拟heapq.heappop调用次数
    pop_count = 0
    original_heappop = heapq.heappop
    
    def mock_heappop(heap):
        nonlocal pop_count
        pop_count += 1
        return original_heappop(heap)
    
    monkeypatch.setattr(heapq, 'heappop', mock_heappop)
    
    find_shortest_path(graph, "a", "d")
    assert pop_count <= 4  # 理想情况下应提前终止

# 测试用图数据结构
TEST_GRAPH = {
    "apple": {"banana": 1, "orange": 2},
    "banana": {"pear": 1, "orange": 1},
    "orange": {},
    "pear": {"banana": 1},
    "mango": {}  # 孤立节点
}

class TestCalcShortestPath:
    """calcShortestPath 函数白盒测试"""
##
    # 测试 word2 提供的情况
    def test_word2_provided_reachable(self):
        """测试可达两点间路径"""
        path, length = calcShortestPath(TEST_GRAPH, "apple", "pear")
        with open("writlog.txt",'a') as f:
            f.write(str(path))
            f.write(str(length))
            f.write("\n")
        assert path == ["apple", "banana", "pear"]
        assert length == 2

##
    def test_word2_provided_word1_not_in_graph(self):
        """测试起点不存在图中"""
        result = calcShortestPath(TEST_GRAPH, "grape", "apple")
        with open("writlog.txt",'a') as f:
            f.write(str(result))
            f.write("\n")
        assert result is None

##
    def test_word2_provided_word2_not_in_graph(self):
        """测试终点不存在图中"""
        result = calcShortestPath(TEST_GRAPH, "apple", "grape")
        with open("writlog.txt",'a') as f:
            f.write(str(result))
            f.write("\n")
        assert result == None
##
    # 测试 word2 未提供的情况
    def test_word2_not_provided(self):
        """测试 word2 未提供的情况"""
        result = calcShortestPath(TEST_GRAPH, "apple")
        with open("writlog.txt",'a') as f:
            f.write(str(result))
            f.write("\n")
        assert isinstance(result, dict)
        assert result["banana"] == (["apple", "banana"], 1)
        assert result["pear"] == (["apple", "banana", "pear"], 2)
        assert "mango" not in result  # 不可达节点不应出现
