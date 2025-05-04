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

from lib import *

#return Void的话除了使用指针就只能在函数里打印了
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


def queryBridgeWords(graph: dict[dict],word1: str, word2: str) -> str:
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
        return(f"No bridge words from {word1} to {word2}!")
    else:
        # 格式化输出多个桥接词的情况
        if len(bridge_words) == 1:
            return(f"The bridge word from {word1} to {word2} is: {bridge_words[0]}")
            
        else:
            front = ", ".join(bridge_words[:-1])
            last = bridge_words[-1]
            return(f"The bridge words from {word1} to {word2} are: {front} and {last}.")


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
        result = queryBridgeWords_r(graph, word1, word2)
        
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
#python的float就是double
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

#randomWalk想要交互停止只能允许输入
def randomWalk(graph: Dict[str, Dict[str, int]], 
                 start_word: str = None,
                 output_file: str = "random_walk.txt",
                 interactive: bool = False) -> str:
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

    return " -> ".join(visited_nodes)



def main(input_file: str) -> None:
    """主程序入口，接收用户输入文件，生成图，并允许用户选择后续各项功能"""
    print(f"Processing input file: {input_file}")
    # 输入并预处理文本
    text, words = process_file(input_file)
    print(f"文本预处理完成，共处理 {len(words)} 个单词")
    
    # 根据words生成有向图
    graph = build_graph(words)
    print(f"有向图构建完成，包含 {len(graph)} 个节点")
    
    
    while True:
        #菜单
        print("\n可用功能：")
        print("1. 展示有向图")
        print("2. 查询桥接词")
        print("3. 生成新文本")
        print("4. 计算最短路径")
        print("5. 计算PageRank")
        print("6. 随机游走")
        print("q. 退出")
        choice = input("\n请输入功能编号(1-6)或q退出: ").strip().lower()
        
        if choice == 'q':
            break
            
        elif choice == '1':
            output_file = input("请输入输出文件名(可选，直接回车跳过): ").strip()
            if output_file:
                showDirectedGraph(graph, output_file)
            else:
                showDirectedGraph(graph)
                
        elif choice == '2':
            word1 = input("请输入第一个单词: ").strip()
            word2 = input("请输入第二个单词: ").strip()
            res = queryBridgeWords(graph, word1, word2)
            print(res)
            
        elif choice == '3':
            input_text = input("请输入要插入桥接词的文本: ").strip()
            print("生成文本:", generateNewText(input_text, graph))
            
        elif choice == '4':
            word1 = input("请输入起始单词: ").strip()
            word2 = input("请输入目标单词(可选，直接回车计算到所有节点): ").strip()
            if word2:
                res = calcShortestPath(graph, word1, word2)
                if res:
                    path,dist = res
                    print(f"最短路径: {' -> '.join(path)} (长度: {dist})")
                else:
                    print("路径不存在")
            else:
                paths = calcShortestPath(graph, word1)
                if paths:
                    for target, (path, dist) in paths.items():
                        print(f"到 {target}: {' -> '.join(path)} (长度: {dist})")
                else:
                    print("路径不存在")
                    
        elif choice == '5':
            word = input("请输入要查询的单词(可选，直接回车显示全部): ").strip()
            use_tfidf = input("使用TF-IDF优化?(y/n): ").lower() == 'y'
            if word:
                print(f"PR值: {calPageRank(graph, word, 0.85, use_tfidf):.4f}")
            else:
                pr = calPageRank(graph, d=0.85, use_tfidf=use_tfidf)
                for w, val in sorted(pr.items(), key=lambda x: -x[1])[:10]:
                    print(f"{w}: {val:.4f}")
                    
        elif choice == '6':
            start_node = input("请输入起始节点(可选，直接回车随机选择): ").strip()
            output_file = input("请输入输出文件名(默认为random_walk.txt): ").strip() or "random_walk.txt"
            hand_stop = input("使用手工停止（回车默认不使用）?(y/n): ").lower() == 'y'
            if start_node and start_node not in graph:
                print(f"错误: 节点 '{start_node}' 不存在")
                continue
            randomWalk(graph, start_word=start_node if start_node else None, 
                       output_file=output_file, interactive=hand_stop)
            
        else:
            print("无效输入，请重新选择")


if __name__ == '__main__':

    # 检查是否提供了输入文件参数
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")

        sys.exit(1)
        # input_filename = "test.txt"
    else:
        # 获取输入文件路径并调用main函数
        input_filename = sys.argv[1]
    main(input_filename)