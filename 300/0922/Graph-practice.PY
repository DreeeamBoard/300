# HW1 과제 코드

# college_len = len(edges['college']) # 59835

# college_list = []

# init_time = 0
# t = 30 # arriving time

# for i in range(college_len) :
#     if edges['college'][i][2] <= t :
#         college_list.append(edges['college'][i])

# node_list = []
# num_node_list = []

# for i in range(college_len) :
#     if edges['college'][i][-1] <= t :
#         node_list.append(edges['college'][i][0])
#         node_list.append(edges['college'][i][1])

#         node_list = list(set(node_list)) #  중복 제거
#         num_node = len(node_list)
#         # num_node_list.append(num_node)
        
# num_node
# # 시간대별로 num_node를 저장하는 코드 추가해야함
# # 현재는 time t까지 총 number of nodes를 출력함

# # number_of_nodes[0] : the number of nodes that appeared until 0th timestamp = 104
# # number_of_nodes[1] : the number of nodes that appeared until 1th timestamp = 427

# # edge

# from collections import defaultdict
# dict = defaultdict(list)
# edge_list = []

# for i in range(10) :
#     if edges['college'][i][-1] <= t :
#         dict[edges['college'][i][0]].append(edges['college'][i][1])

#         edge_list = list(set(dict.values()))

def get_number_of_nodes_edges(edges):
    """
    Compute the number of nodes and edges at each timestamp.

    Input:
    edges: a list of (u, v, t) where of an edge from u to v at time t

    Output:
    number_of_nodes: list of number of nodes 
    number_of_edges: list of number of edges
    e.g., number_of_nodes[i] is the number of nodes that appeared until the i-th timestamp
    """

    ########### TODO ##########
    
    init_time = 0

    number_of_nodes = []
    cur_nodes = []
    total_nodes = []

    cur_edges = 0
    number_of_edges = []

    for i in edges :
        if i[2] != init_time :
            init_time += 1
            total_nodes += cur_nodes
            number_of_nodes.append(len(list(set(total_nodes)))) #  remove duplicates
            number_of_edges.append(cur_edges)
            cur_nodes = [] # re-initialization
        
        cur_nodes.append(i[0])
        cur_nodes.append(i[1])
        cur_edges += 1
    
    total_nodes += cur_nodes
    number_of_nodes.append(len(list(set(total_nodes))))
    number_of_edges.append(cur_edges)
    
    ########### TODO ##########

    return number_of_nodes, number_of_edges

    nodes_by_depth = [] # depth에 따른 nodes
    init_time = 0
    
    for i in range(d+1) : 
        cur_nodes = [] # current nodes initialization
        for j in range(b**i) : # b^i
            cur_nodes.append(init_time)
            init_time += 1
        nodes_by_depth.append(cur_nodes)
    
    
    child_to_parent = {} # 특정 child의 parent
    cur_parents = [0]
    
    for i in range(d) :
        new_parents = []
        for parent in cur_parents :
            for j in range(1, b+1): # b = 2이면 j = 1,2
                child = parent * b + j
                child_to_parent[child] = parent # 딕셔너리의 key는 child, value는 parent 
                new_parents.append(child) # 이 child들은 후에 새로운 parents가 되므로
        cur_parents = new_parents