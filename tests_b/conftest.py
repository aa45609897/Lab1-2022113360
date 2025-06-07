import pytest
from typing import Dict
from code_ori.lab1 import queryBridgeWords  # 替换为实际模块路径

@pytest.fixture
def sample_graph() -> Dict[str, Dict[str, int]]:
    """提供测试用的有向图"""
    return {
        "apple": {"banana": 1, "cat": 2,"egg":1},
        "banana": {"dog": 1},
        "cat": {"dog": 1,"apple":1},
        "dog": {},
        "egg":{}
    }