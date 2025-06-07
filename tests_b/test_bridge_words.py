import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from code_ori.lab1 import queryBridgeWords  # 替换为实际模块路径

class TestBridgeWords:
    """测试桥接词查询功能"""

    def test_word1_missing(self, sample_graph):
        """测试word1不在图中的情况"""
        result = queryBridgeWords(sample_graph, "pear", "dog")
        # print(result)
        assert result == "No pear in the graph!"

    def test_word2_missing(self, sample_graph):
        """测试word2不在图中的情况"""
        result = queryBridgeWords(sample_graph, "apple", "elephant")
        # print(result)
        assert result == "No elephant in the graph!"

    def test_both_words_missing(self, sample_graph):
        """测试两个单词都不在图中"""
        result = queryBridgeWords(sample_graph, "pear", "elephant")
        print(result)
        with open("output.txt", "w") as f:
            f.write(result)
        assert "pear" in result and "elephant" in result

    def test_no_bridge_words(self, sample_graph):
        """测试无桥接词的情况"""
        result = queryBridgeWords(sample_graph, "apple", "cat")
        assert result == "No bridge words from apple to cat!"

    def test_single_bridge_word(self, sample_graph):
        """测试单个桥接词"""
        result = queryBridgeWords(sample_graph, "cat", "egg")
        print(result)
        assert result == "The bridge word from cat to egg is: apple"

    def test_multiple_bridge_words(self):
        """测试多个桥接词"""
        graph = {
            "apple": {"banana": 1, "zebra": 1},
            "banana": {"dog": 1},
            "zebra": {"dog": 1},
            "dog": {}
        }
        result = queryBridgeWords(graph, "apple", "dog")
        print(result)
        with open("output.txt", "a") as f:
            f.write(result)
        assert ("banana" in result and "zebra" in result and 
                ("are: banana and zebra." in result or 
                 "are: zebra and banana." in result))

    def test_direct_connection(self, sample_graph):
        """测试直接相连的单词对"""
        result = queryBridgeWords(sample_graph, "banana", "dog")
        assert result == "No bridge words from banana to dog!"

    def test_empty_graph(self):
        """测试空图情况"""
        result = queryBridgeWords({}, "apple", "dog")
        print(result)
        with open("output.txt", "a") as f:
            f.write(result)
        assert "apple" in result and "dog" in result

    def test_case_insensitive(self, sample_graph):
        """测试大小写不敏感"""
        result1 = queryBridgeWords(sample_graph, "APPLE", "DOG")
        result2 = queryBridgeWords(sample_graph, "apple", "dog")
        print(result1)
        with open("output.txt", "a") as f:
            f.write(result1)
        assert result1 == result2