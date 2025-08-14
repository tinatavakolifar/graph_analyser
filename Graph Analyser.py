from collections import defaultdict, deque  # دیک با لیست پیش‌فرض و صف سریع برای
from sys import exit  # استفاده برای خروج از برنامه
import csv  # خواندن و پردازش فایل CSV
import heapq  # صف اولویت برای الگوریتم دیکسترا

import networkx as nx  # رسم و عملیات پیشرفته روی گراف
import matplotlib.pyplot as plt  # رسم و ذخیره تصویر گراف


def print_menu():  # نمایش منوی اصلی برنامه
    print(''' MENU  \n
---------------------------------------------------------------------\n
1. Display basic graph information (graph type) \n
2. Calculate the degree of a specified node \n
3. Check graph connectivity and find possible paths between two nodes \n
4. Detect cycles in the graph \n
5. Perform topological sorting (for directed acyclic graphs) \n
6. Find connected components (for undirected graphs) \n
7. Visualize the graph and save the output image \n
8. Re-enter graph details \n
9. Exit the program \n
--------------------------------------------------------------------- \n
Tip: Press 0 anytime to see the menu again\n
''')

def get_graph_info(): # دریافت اطلاعات گراف از کاربر
    edge_list = []
    print('''GRAPH ANALYSER \n
        This analyser can process up to 100 nodes and 500 edges.''')

    while True:
        try:
            manual_CSV = input('Would you rather manual or CSV input (m/c): ')
            if manual_CSV not in ['m', 'c']:
                print('Invalid input.')
        except ValueError:
            print("Enter either 'm' or 'c'.")
            continue

        try:
            graph_type = input(
                "Is it directed? 'y' if directed, 'n' if not: "
            ).lower()
            is_directed = True if graph_type == 'y' else False

            graph_weight = input(
                "Is the graph weighted? 'y' if weighted, 'n' if not: "
            ).lower()
            is_weighted = True if graph_weight == 'y' else False

            if graph_weight not in ['y', 'n'] or graph_type not in ['y', 'n']:
                print('Invalid input for either directness or weight.')
        except ValueError:
            print('Answer only with y or n.')
            continue

        try:
            if manual_CSV == 'm':  # دریافت تعداد یال ها و گره ها
                node_num = int(
                    input('How many nodes are in the graph? (from 0 to 100)')
                )
                edge_num = int(
                    input('How many edges are in the graph? (from 0 to 500)')
                )
                if not (0 <= node_num <= 100 and 0 <= edge_num <= 500):
                    print('Invalid range.')
                    continue
        except ValueError:
            print('Enter integers.')
            continue

        try:
            for _ in range(edge_num):
                u, v = map(
                    int,
                    input(
                        'Enter two nodes connected by an edge (e.g. "1 2"): '
                    ).strip().split()
                )
                if not (1 <= u <= node_num and 1 <= v <= node_num):
                    print(
                        f'Invalid input. Please enter integers between 1 and {node_num}.'
                    )
                    continue

                if is_weighted:
                    w = int(
                        input(
                            'Please enter the weight of the edge connecting the nodes: '
                        )
                    )
                    edge_list.append((u, v, w))
                    if not is_directed:
                        edge_list.append((v, u, w))
                else:
                    edge_list.append((u, v))
                    if not is_directed:
                        edge_list.append((v, u))
            break
        except ValueError:  # برای تایپ نادرست ورودی
            print(
                'Invalid input. Please enter two integers separated by a space.'
            )

    if manual_CSV == 'c':
        print('''The file must have columns like:
        - for weighted:   u, v, w
        - for unweighted: u, v''')

        while True:
            file_path = input("Enter CSV file path: ")
            try:
                with open(file_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    edge_list = []
                    for row in reader:
                        try:
                            if is_weighted:
                                if len(row) != 3:
                                    raise ValueError(
                                        f'Expected 3 values per row (u, v, w), got {len(row)}'
                                    )
                                u, v, w = map(int, row)
                                edge_list.append((u, v, w))
                                if not is_directed:
                                    edge_list.append((v, u, w))
                            else:
                                if len(row) != 2:
                                    raise ValueError(
                                        f'Expected 2 values per row (u, v), got {len(row)}'
                                    )
                                u, v = map(int, row)
                                edge_list.append((u, v))
                                if not is_directed:
                                    edge_list.append((v, u))
                        except ValueError as error:
                            print(f'CSV row format error: {row} -> {error}')
                            raise

                if not edge_list:
                    print("CSV is empty. Please enter a valid file.")
                    continue
                node_num = max(max(edge[0], edge[1]) for edge in edge_list)
                if node_num > 100:
                    print(
                        'Maximum allowed nodes is 100. Please provide a valid CSV.'
                    )
                    continue
                edge_num = len(edge_list) // (1 if is_directed else 2)
                return (
                    edge_list,
                    node_num,
                    edge_num,
                    is_directed,
                    is_weighted,
                )
            except FileNotFoundError:
                print(f'File not found: {file_path}. Please try again.')
            except ValueError:
                print(
                    'There was a problem with your CSV file format. Please re-enter the file.'
                )

def get_graph_simplicity(edge_list, is_directed): # بررسی سادگی گراف
    seen_edges = set()

    for edge in edge_list:
        u, v = edge[0], edge[1]

        if u == v:
            return 'Not simple (has self-loop)'

        if is_directed:
            if (u, v) in seen_edges:
                return 'Multigraph'
            seen_edges.add((u, v))
        else:
            edgeKey = tuple(sorted((u, v)))
            if edgeKey in seen_edges:
                return 'Multigraph'
            else:
                seen_edges.add(edgeKey)
    return 'Simple'

def get_node_degree(node_num, edge_list, is_directed): #درجه گره‌ و گره‌ ایزوله
    degree_list = []                                   #  و امکان تشکیل حلقه
    makes_cycle = True 

    connected_nodes = set()
    for edge in edge_list:
        connected_nodes.add(edge[0])
        connected_nodes.add(edge[1])

    for node in range(1, node_num + 1):
        if is_directed:
            degree_in = degree_out = 0
            for u, v, *_ in edge_list:
                if u == node:
                    degree_out += 1
                if v == node:
                    degree_in += 1
            degree_list.append(
                (node, degree_in, degree_out, degree_in + degree_out)
            )
            if degree_in != 1 or degree_out != 1:
                makes_cycle = False
        else:
            degree = sum(
                1 for u, v, *_ in edge_list if u == node or v == node
            )
            degree_list.append((node, degree))
            if degree != 2:
                makes_cycle = False

    isolated_list = [
        n for n in range(1, node_num + 1) if n not in connected_nodes
    ]
    isolated_num = len(isolated_list)
    return makes_cycle, isolated_num, degree_list, isolated_list

def get_neighbor_weight_dict(edge_list, is_directed, is_weighted): # دیکشنری
    neighbors_dict = defaultdict(list)                             
    for edge in edge_list:
        u, v = edge[:2]
        if is_weighted:
            w = edge[2]
            neighbors_dict[u].append((v, w))
            if not is_directed:
                neighbors_dict[v].append((u, w))
        else:
            neighbors_dict[u].append(v)
            if not is_directed:
                neighbors_dict[v].append(u)
    return neighbors_dict

def detect_cycle(neighbors_dict, node_num, is_directed): # وجود حلقه در گراف 
    visited = set()
    recStack = set()

    def dfs(node, parent=None):
        visited.add(node)
        if is_directed:
            recStack.add(node)
        for neighbor, *_ in neighbors_dict.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif (
                (is_directed and neighbor in recStack)
                or (not is_directed and neighbor != parent)
            ):
                return True
        if is_directed:
            recStack.remove(node)
        return False

    for node in range(1, node_num + 1):
        if node not in visited and dfs(node):
            return True
    return False

def connected_BFS(neighbors_dict, start, node_num): # بررسی همبندی گراف
    visited = set([start])
    queue = deque([start]) 

    while queue:
        current = queue.popleft() 
        for neighbor, *_ in neighbors_dict.get(current, []): 
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    is_connected = len(visited) == node_num
    return is_connected, visited

def all_paths_BFS(neighbors_dict, start, goal, visited = None): #مسیرهای ساده 
    if visited is None:
        visited = set()
    if start == goal:
        return 1
    visited.add(start)
    count = 0
    for neighbor, *_ in neighbors_dict.get(start, []):
        if neighbor not in visited:
            count += all_paths_BFS(neighbors_dict, neighbor, goal, visited)
    visited.remove(start)
    return count

def shortest_path(neighbors_dict, start, goal, is_weighted = False): #کوتاه‌ترین 
    if is_weighted:                                                  # مسیر
        heap = [(0, start, [start])]  # (مسیر تا اینجا، گره فعلی، وزن تا اینجا)
        visited = {}
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == goal:
                return path, cost
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neighbor, weight in neighbors_dict.get(node, []):
                new_cost = cost + weight
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
        return None, float('inf')
    else:
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path, len(path) - 1
            for neighbor in neighbors_dict.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, None

def topological_sort(neighbors_dict, node_num): # توپولوژیک
    in_degree = {node: 0 for node in range(1, node_num + 1)}

    for node in neighbors_dict: 
        for neighbor, *_ in neighbors_dict[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []

    while queue:
        current = queue.popleft()
        topo_order.append(current)
        for neighbor, *_ in neighbors_dict.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) == node_num:
        return topo_order
    else:
        return None  # به علت وجود حلقه

def connected_components_undirected(neighbors_dict, node_num): #  مؤلفه‌ متصل 
    visited, components = set(), []
    def dfs(start):
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                for neighbor, *_ in neighbors_dict.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return component

    for node in range(1, node_num + 1):
        if node not in visited:
            comp = dfs(node)
            components.append(comp)

    return components

def weakly_connected_components_directed(neighbors_dict, node_num): #  متصل ضعیف
    undirected_dict = defaultdict(list)

    for node in neighbors_dict:
        for neighbor, *_ in neighbors_dict[node]:
            undirected_dict[node].append(neighbor)
            undirected_dict[neighbor].append(node)  
    return weakly_connected_components_directed(undirected_dict, node_num)

def strongly_connected_components(neighbors_dict, node_num): # متصل قوی 
    index = 0
    stack = []
    indices = {}
    lowlink = {}
    onstack = set()
    sccs = []

    def strong_connect(v):
        nonlocal index
        indices[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w, *_ in neighbors_dict.get(v, []):
            if w not in indices:
                strong_connect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(1, node_num + 1):
        if v not in indices:
            strong_connect(v)

    return sccs

def visualize_graph(edge_list, is_directed, is_weighted): # رسم و ذخیره گراف 
    G = nx.DiGraph() if is_directed else nx.Graph()
    for edge in edge_list:
        if is_weighted:
            u, v, w = edge
            G.add_edge(u, v, weight=w)
        else:
            u, v = edge[:2]
            G.add_edge(u, v)
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        node_size=500,
        font_size=12,
    )  # رسم گراف
    if is_weighted:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=nx.get_edge_attributes(G, 'weight')
        )
    plt.savefig("graph_visualization.png")
    plt.show()

def setupGraph(): # آماده‌سازی گراف
    (
        edge_list,
        node_num,
        edge_num,
        is_directed,
        is_weighted,
    ) = get_graph_info()
    is_simple = get_graph_simplicity(edge_list, is_directed)
    (
        makes_cycle,
        isolated_num,
        degree_list,
        isolated_list,
    ) = get_node_degree(node_num, edge_list, is_directed)
    neighbors_dict = get_neighbor_weight_dict(
        edge_list, is_directed, is_weighted
    )
    is_connected, visited = connected_BFS(neighbors_dict, 1, node_num)
    return (
        node_num,
        edge_num,
        is_directed,
        is_weighted,
        edge_list,
        is_simple,
        makes_cycle,
        isolated_num,
        degree_list,
        isolated_list,
        neighbors_dict,
        is_connected,
    )

def main(): # حلقه اصلی برنامه و مدیریت منو
    (
        node_num,
        edge_num,
        is_directed,
        is_weighted,
        edge_list,
        is_simple,
        makes_cycle,
        isolated_num,
        degree_list,
        isolated_list,
        neighbors_dict,
        is_connected,
    ) = setupGraph()

    while True:
        try:
            option = int(input('Select desired option (0 - 9): '))
            if not (0 <= option < 10):
                print('''Invalid option. 
                     Please choose a number between the options above.''')
        except ValueError:
            print('''Invalid option.
                 Please choose a number between the options above.''')
            continue

        match option:
            case 0:
                print_menu()

            case 1:
                type_list = []
                if node_num == 1 and edge_num == 0:
                    type_list.append('Point')
                elif node_num == 0 and len(edge_list) == 0:
                    type_list.append('Null')
                elif node_num == 2 and edge_num == 1:
                    type_list.append('Line')
                if is_directed and edge_num == node_num * (node_num - 1):
                    type_list.append('Complete')
                elif not is_directed and (
                    edge_num == node_num * (node_num - 1) // 2
                ):
                    type_list.append('Complete')
                if 'Simple' in is_simple:
                    type_list.append('Simple')
                elif 'Multigraph' in is_simple:
                    type_list.append('Multigraph')
                elif 'not simple' in is_simple.lower():
                    type_list.append(is_simple)
                if makes_cycle:
                    type_list.append('Cycle')
                if node_num - 1 == edge_num and isolated_num == 0:
                    if not detect_cycle(neighbors_dict, node_num, is_directed):
                        type_list.append('Tree')
                print('Graph type: ', ', '.join(type_list))
                featureDescriptions = {
                    'Point': 'Graph has a single node.',
                    'Null': 'Graph has no nodes.',
                    'Line': 'Graph has two nodes connected by one edge.',
                    'Complete': (
                        'Every pair of distinct nodes is connected by an edge.'
                    ),
                    'Simple': (
                        'Graph has no loops or multiple edges between the same nodes.'
                    ),
                    'Multigraph': (
                        'Graph has multiple edges between the same pair of nodes.'
                    ),
                    'Cycle': 'Graph contains at least one cycle.',
                    'Tree': 'Graph is connected and has no cycles.',
                }
                for feature in type_list:
                    print(
                        f' - {feature}: {featureDescriptions.get(feature, "")}'
                    )

            case 2:
                try:
                    wanted_node = int(input('Enter node number: '))
                    for node in degree_list:
                        if wanted_node == node[0]:
                            if is_directed:
                                print(
                                    f'In-degree: {node[1]}, '
                                    f'out-degree: {node[2]}, '
                                    f'Total degree: {node[3]}'
                                )
                            else:
                                print(f'Degree: {node[1]}')
                except ValueError:
                    print('Enter an integer.')

            case 3:
                if is_connected:
                    print('The graph is connected')
                else:
                    print("The graph isn't connected")
                try:
                    start_node = int(input('Enter the start node: '))
                    end_node = int(input('Enter the end node: '))
                    if not (
                        1 <= start_node <= node_num
                        and 1 <= end_node <= node_num
                    ):
                        print('Invalid input. nodes must be in input range.')
                    else:
                        path_count = all_paths_BFS(
                            neighbors_dict, start_node, end_node
                        )
                        print(f'''There are {path_count} possible simple 
                              paths from {start_node} to {end_node}.''')
                        
                        path, *_ = shortest_path(
                            neighbors_dict, start_node, end_node
                        )
                        if path:
                            print('Shortest path:', ' -> '.join(map(str, path)))
                        else:
                            print(
                                f"there's no path between {start_node} and {end_node}."
                            )
                except ValueError:
                    print(
                        'Invalid input. Please input integers. Hit enter to continue.'
                    )
            case 4:
                if detect_cycle(neighbors_dict, node_num, is_directed):
                    print('Cycle detected in the graph.')
                else:
                    print('No cycles found.')

            case 5:
                if not is_directed:
                    print('Topological sorting only works for directed graphs.')
                else:
                    if detect_cycle(neighbors_dict, node_num, is_directed):
                        print(
                            'Graph has a cycle; topological sorting is not possible.'
                        )
                    else:
                        topo = topological_sort(neighbors_dict, node_num)
                        if topo:
                            print('Topological order:', ' -> '.join(map(str, topo)))
                        else:
                            print('Topological sorting failed.')

            case 6:
                if is_directed:
                    scc = strongly_connected_components(
                        neighbors_dict, node_num
                    )
                    print('Strongly Connected Components: ', scc)
                    for i, comp in enumerate(scc, 1):
                        print(f"Component {i}: {comp}")
                    wcc = weakly_connected_components_directed(
                        neighbors_dict, node_num
                    )
                    print("\nWeakly Connected Components:")
                    for i, comp in enumerate(wcc, 1):
                        print(f"Component {i}: {comp}")
                else:
                    print(
                        'Components:',
                        connected_components_undirected(
                            neighbors_dict, node_num
                        ),
                    )

            case 7:
                visualize_graph(edge_list, is_directed, is_weighted)

            case 8:
                (
                    node_num,
                    edge_num,
                    is_directed,
                    is_weighted,
                    edge_list,
                    is_simple,
                    makes_cycle,
                    isolated_num,
                    degree_list,
                    isolated_list,
                    neighbors_dict,
                    is_connected,
                ) = setupGraph()
                print("Graph details re-entered successfully.")

            case 9:
                print('Stopping the program...')
                exit()


if __name__ == '__main__':
    main()