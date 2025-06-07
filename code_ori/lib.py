import re
import sys
import math
import time
import heapq
import random
import threading

from graphviz import Digraph
from collections import defaultdict
from typing import Any, Set, Dict, Tuple,List

def queryBridgeWords_r(graph: dict[dict],word1: str, word2: str) -> list[str]:
    """
    查找两个单词之间的桥接词
    参数:
        graph: 有向图字典 {source: {target: weight}}
        word1: 起始词
        word2: 目标词
    返回:
        结果字符串（根据不同情况返回不同提示）
    """
    # 转换为小写处理
    word1 = word1.lower()
    word2 = word2.lower()
    
    # 检查单词是否在图中
    if word1 not in graph or word2 not in graph:
        missing = []
        if word1 not in graph: missing.append(word1)
        if word2 not in graph: missing.append(word2)
        # print(f"No {' or '.join(missing)} in the graph!")
        return []
    
    # 查找桥接词
    bridge_words = []
    # 获取word1的所有直接后继节点（word3候选）
    for word3 in graph.get(word1, {}):
        # 检查word3是否能到达word2
        if word2 in graph.get(word3, {}):
            bridge_words.append(word3)
    # 处理结果
    if not bridge_words:
        # print(f"No bridge words from {word1} to {word2}!")
        pass
    else:
        # 格式化输出多个桥接词的情况
        if len(bridge_words) == 1:
            # print(f"The bridge word from {word1} to {word2} is: {bridge_words[0]}")
            pass
            
        else:
            front = ", ".join(bridge_words[:-1])
            last = bridge_words[-1]
            # print(f"The bridge words from {word1} to {word2} are: {front} and {last}.")

    return bridge_words

def find_shortest_path(graph: dict, word1: str, word2: str) -> tuple:
    """
    计算两个单词之间的最短路径
    参数:
        graph: 有向图 {source: {target: weight}}
        word1: 起始词
        word2: 目标词
    返回:
        (路径列表, 路径长度) 或 (None, inf) 如果不可达
    """
    word1 = word1.lower()
    word2 = word2.lower()
    
    # Dijkstra算法实现
    distances = {node: float('inf') for node in graph}
    print(distances[word2])
    distances[word1] = 0
    previous = {node: None for node in graph}
    heap = [(0, word1)]
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current == word2:  # 提前终止优化
            break
        if current_dist > distances[current]:
            continue
            
        for neighbor, weight in graph.get(current, {}).items():
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))
    
    # 构建路径
    if distances[word2] == float('inf'):
        return None, float('inf')
    
    path = []
    current = word2
    while current:
        path.insert(0, current)
        current = previous[current]
    return path, distances[word2]

def find_all_shortest_paths(graph: dict, word1: str) -> list:
    """
    计算起始词到所有其他可达单词的最短路径
    参数:
        graph: 有向图
        word1: 起始词
    返回:
        {目标词: (路径列表, 路径长度)}
    """
    word1 = word1.lower()
    
    # Dijkstra算法（计算到所有节点的最短路径）
    distances = {node: float('inf') for node in graph}
    distances[word1] = 0
    previous = {node: None for node in graph}
    heap = [(0, word1)]
    
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > distances[current]:
            continue
            
        for neighbor, weight in graph.get(current, {}).items():
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))
    
    # 构建所有路径
    result = {}
    for node in graph:
        if node != word1 and distances[node] != float('inf'):
            path = []
            current = node
            while current:
                path.insert(0, current)
                current = previous[current]
            result[node] = (path, distances[node])
    return result




def clean_text(text):
    """
    文本清洗函数：
    1. 将换行符/回车符替换为空格
    2. 将所有非字母字符替换为空格
    3. 合并连续空格
    4. 返回元组：(处理后的纯文本字符串, 小写单词列表)
    """
    # 替换所有换行符为空格
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 移除非字母字符（只保留a-z和A-Z），其他都替换为空格
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # 将多个连续空格合并为单个空格
    text = re.sub(r' +', ' ', text)
    
    # 转换为全小写
    lower_text = text.lower()
    
    # 分割为单词列表
    words = lower_text.strip().split()
    
    return lower_text.strip(), words  # 返回处理后的文本和单词列表

def process_file(filename):
    """处理输入文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            return clean_text(content)
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时出错: {e}")
        sys.exit(1)


def build_graph(words):
    """
    构建有向图：
    - 节点：单词（小写）
    - 边A→B：A和B在文本中相邻出现的次数
    - 返回：字典形式的图结构
    """
    graph = defaultdict(dict)
    
    for i in range(len(words)-1):
        current_word = words[i]
        next_word = words[i+1]
        
        # 更新边权重
        if next_word in graph[current_word]:
            graph[current_word][next_word] += 1
        else:
            graph[current_word][next_word] = 1
    
    return dict(graph)


def compute_pagerank(graph: dict, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6, 
                   initial_weights: dict = None) -> dict:
    """
    PageRank算法实现
    
    参数:
        graph: 有向图 {source: {target: weight}}
        d: 阻尼系数(默认0.85)
        max_iter: 最大迭代次数
        tol: 收敛阈值
        initial_weights: 初始权重字典(可选)
    返回:
        节点PR值字典 {node: PR_value}
    """
    # 收集所有节点（包括那些只作为目标节点的）
    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets.keys())
    nodes = list(all_nodes)
    N = len(nodes)
    
    # 构建出链和入链图
    outgoing_links = defaultdict(set)
    incoming_links = defaultdict(set)
    
    for src, targets in graph.items():
        for tgt in targets:
            outgoing_links[src].add(tgt)
            incoming_links[tgt].add(src)
    
    # 处理悬挂节点（没有出链的节点）
    dangling_nodes = [node for node in nodes if not outgoing_links.get(node)]
    
    # 初始化PR值
    if initial_weights:
        total = sum(initial_weights.values())
        pr = {n: (initial_weights.get(n, 0))/total for n in nodes}
    else:
        pr = {n: 1/N for n in nodes}
    
    # 迭代计算
    for _ in range(max_iter):
        new_pr = {}
        dangling_sum = sum(pr[node] for node in dangling_nodes) / N
        
        for node in nodes:
            # 来自其他节点的贡献
            incoming_sum = sum(pr[src]/len(outgoing_links[src]) 
                          for src in incoming_links[node])
            
            new_pr[node] = (1-d)/N + d * (incoming_sum + dangling_sum)
        
        # 检查收敛
        diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        if diff < tol:
            break
        pr = new_pr
    
    return pr

def compute_tfidf_weights(documents: list) -> dict:
    """
    计算TF-IDF权重作为初始PR值
    参数:
        documents: 分词后的文档列表 [[word1, word2,...], ...]
    返回:
        {word: tfidf_weight}
    """
    # 计算TF
    tf = defaultdict(dict)
    doc_count = len(documents)
    for doc_id, doc in enumerate(documents):
        word_count = len(doc)
        for word in set(doc):
            tf[word][doc_id] = doc.count(word) / word_count
    
    # 计算IDF
    idf = {}
    for word in tf:
        idf[word] = math.log(doc_count / (1 + len(tf[word])))
    
    # 计算TF-IDF
    tfidf = defaultdict(float)
    for word in tf:
        tfidf[word] = sum(tf_val * idf[word] for tf_val in tf[word].values())
    
    return dict(tfidf)