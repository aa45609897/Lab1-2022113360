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

def random_walk(graph: Dict[str, Dict[str, int]], 
                 start_word: str = None,
                 output_file: str = "random_walk.txt",
                 interactive: bool = False) -> Tuple[List[str], Set[Tuple[str, str]]]:
    """
    改进版随机游走（线程控制+非阻塞输入检测）
    
    参数:
        graph: 有向图 {source: {target: weight}}
        start_word: 起始词（None则随机选择）
        output_file: 结果输出文件
        interactive: 是否启用交互式停止
    返回:
        (访问节点列表, 访问边集合)
    """
    if not graph:
        print("图为空，无法进行随机游走")
        return [], set()

    # 初始化游走状态
    current_node = random.choice(list(graph.keys())) if start_word is None else start_word
    visited_nodes = [current_node]
    visited_edges = set()
    stop_flag = False  # 线程共享标志位

    # 输入监听线程
    def input_listener():
        nonlocal stop_flag
        if interactive:
            print("\n提示: 按回车键可随时终止游走...")
            input()  # 阻塞等待回车
            stop_flag = True
    sleep_time = 0
    if interactive:
        listener_thread = threading.Thread(target=input_listener)
        listener_thread.daemon = True  # 设为守护线程
        listener_thread.start()
        sleep_time = 1

    print(f"随机游走开始，起点: {current_node}")

    # 主游走循环
    while not stop_flag:
        # 检查终止条件
        if current_node not in graph or not graph[current_node]:
            print(f"\n终止原因: 节点 {current_node} 无出边")
            break

        # 按权重随机选择下一节点
        targets = graph[current_node]
        next_node = random.choices(
            list(targets.keys()),
            weights=list(targets.values()),
            k=1
        )[0]
        
        # 检查边重复
        edge = (current_node, next_node)
        if edge in visited_edges:
            print(f"\n终止原因: 重复边 {edge}")
            break

        # 更新状态
        visited_edges.add(edge)
        visited_nodes.append(next_node)
        current_node = next_node
        
        # 打印当前进度
        print(f"{edge[0]} → {edge[1]}")
        time.sleep(sleep_time)  # 控制游走速度

    # 结果处理
    print("\n\n随机游走结果:")
    print(" -> ".join(visited_nodes))
    print(f"遍历节点数: {len(visited_nodes)}")
    print(f"遍历边数: {len(visited_edges)}")

    # 文件输出
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Random Walk Result:\n")
        f.write(" -> ".join(visited_nodes))
        f.write(f"\n\nTotal nodes: {len(visited_nodes)}\n")
        f.write(f"Total edges: {len(visited_edges)}\n")
        f.write("\nEdge sequence:\n")
        for i, edge in enumerate(visited_edges, 1):
            f.write(f"{i}. {edge[0]} -> {edge[1]}\n")

    return visited_nodes, visited_edges


def calPageRank(graph: dict, word: str = None, d: float = 0.85, use_tfidf: bool = False) -> float:
    """
    计算PageRank的接口函数
    参数:
        graph: 有向图
        word: 查询的单词(None时返回全部)
        d: 阻尼系数
        use_tfidf: 是否使用TF-IDF优化初始值
    返回:
        单个PR值或完整PR字典
    """
    # 可选TF-IDF初始化
    initial_weights = None
    if use_tfidf:
        # 将图转换为文档形式(每个节点的出边作为文档)
        documents = [[src] + list(targets.keys()) for src, targets in graph.items()]
        initial_weights = compute_tfidf_weights(documents)
    
    pr = compute_pagerank(graph, d=d, initial_weights=initial_weights)
    
    if word:
        word = word.lower()
        return pr.get(word, 0)
    else:
        return pr

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

def calcShortestPath(graph: dict, word1: str, word2: str = None) -> tuple:
    """
    统一接口函数
    参数:
        graph: 有向图
        word1: 起始词
        word2: 可选目标词
    返回:
        如果word2提供: (路径, 长度) 或 (None, inf)
        如果word2未提供: {目标词: (路径, 长度)}
    """
    if word1 in graph:
        if word2 is not None:
            if word2 in graph:
                return find_shortest_path(graph, word1, word2)
            else:
                return None
        else:
            return find_all_shortest_paths(graph, word1)
    else:
        return None

def generateNewText(inputText: str, graph: dict) -> str:
    """
    根据bridge word生成新文本
    参数:
        inputText: 用户输入文本
        graph: 已构建的有向图
    返回:
        插入桥接词后的新文本
    """
    text,words = clean_text(inputText)
    if len(words) < 2:
        return inputText
    # print( words )
    output = []
    for i in range(len(words)-1):
        word1, word2 = words[i], words[i+1]
        output.append(word1)
        # 查询桥接词
        result = queryBridgeWords(graph, word1, word2)
        
        # 解析结果并随机选择桥接词
        if len(result)==1:
            bridge = result[0]
            output.append(bridge)
        elif len(result)>1:
            bridges = result
            print(bridges)
            chosen = random.choice(bridges)
            output.append(chosen)
    
    output.append(words[-1])

    # print(output)
    
    # 重建文本（简单实现，实际需要更复杂的文本重建逻辑）
    return " ".join(output)


def showDirectedGraph(G: dict[dict], output_file = None) -> None:
    """
    展示有向图：
    1. 在CLI上以文本格式展示
    2. 可选生成图形文件（需要安装graphviz）
    
    参数:
        G: 有向图字典 {source: {target: weight}}
        output_file: 图形输出文件名（不包含扩展名），为None则不生成图形
    """
    # CLI文本展示
    print("\n有向图结构（文本格式）:")
    print("=" * 50)
    for src, edges in G.items():
        if edges:
            connections = [f"{dest}({weight})" for dest, weight in edges.items()]
            print(f"{src.ljust(15)} → {', '.join(connections)}")
        else:
            print(f"{src.ljust(15)} → (无出边)")
    print("=" * 50)
    
    # 图形可视化（可选）
    if output_file:
        try:
            dot = Digraph(comment='Directed Graph')
            
            # 添加所有节点（自动去重）
            all_nodes = set(G.keys())
            for edges in G.values():
                all_nodes.update(edges.keys())
            for node in all_nodes:
                dot.node(node)
            
            # 添加边
            for src, edges in G.items():
                for dest, weight in edges.items():
                    dot.edge(src, dest, label=str(weight))
            
            # 保存并渲染
            dot.render(output_file, format='png', cleanup=True)
            print(f"\n图形已保存为 {output_file}.png")
        except Exception as e:
            print(f"\n无法生成图形文件（请确保已安装graphviz）: {e}")


def queryBridgeWords(graph: dict[dict],word1: str, word2: str) -> list[str]:
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
        print(f"No {' or '.join(missing)} in the graph!")
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
        print(f"No bridge words from {word1} to {word2}!")
    else:
        # 格式化输出多个桥接词的情况
        if len(bridge_words) == 1:
            print(f"The bridge word from {word1} to {word2} is: {bridge_words[0]}")
            
        else:
            front = ", ".join(bridge_words[:-1])
            last = bridge_words[-1]
            print(f"The bridge words from {word1} to {word2} are: {front} and {last}.")

    return bridge_words




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