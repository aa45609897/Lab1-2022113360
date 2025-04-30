import sys
from lib import *


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
            queryBridgeWords(graph, word1, word2)
            
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
            if start_node and start_node not in graph:
                print(f"错误: 节点 '{start_node}' 不存在")
                continue
            random_walk(graph, start_word=start_node if start_node else None, 
                       output_file=output_file, interactive=True)
            
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