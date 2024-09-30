import copy
import random
import time
import numpy as np
from getopt import getopt
import sys

start=time.time()
args = getopt(sys.argv, "-s:-t:")[1]
file_path = args[1]
timeout = int(args[3])
seed = int(args[5])
random.seed(seed)
# path="D:\PythonProject\CARP\CARP_samples\egl-s1-A.dat"
# path="D:\PythonProject\CARP\sample.dat"
path = file_path
with open(path, "r") as f:
    content = f.readlines()
# print (content)
NAME = content[0].split(":")[1]
VERTICES_NUMBER = (int)(content[1].split(":")[1])
DEPOT = (int)(content[2].split(":")[1])
REQUIRED_EDGES = (int)(content[3].split(":")[1])
NON_REQUIRED_EDGES = (int)(content[4].split(":")[1])
VEHICLES = (int)(content[5].split(":")[1])
CAPACITY = (int)(content[6].split(":")[1])
RequiredEdgeCost = (int)(content[7].split(":")[1])
EdgeNumber = REQUIRED_EDGES + NON_REQUIRED_EDGES


class edge:
    def __init__(self, node1, node2, cost, demand, index):
        self.node1 = node1
        self.node2 = node2
        self.cost = cost
        self.demand = demand
        self.index = index


# class node:
#     visited
#     children
edges = []
nodeToEdge = {}
nodeToIndex = {}
for i in range(EdgeNumber):
    content[9 + i] = content[9 + i].split()
    Edge = edge((int)(content[9 + i][0]) - 1, (int)(content[9 + i][1]) - 1, (int)(content[9 + i][2]),
                (int)(content[9 + i][3]), i)
    nodeToEdge[(Edge.node1, Edge.node2)] = Edge
    nodeToEdge[(Edge.node2, Edge.node1)] = Edge
    # nodeToIndex[(Edge.node1, Edge.node2)] = 2 * i
    # nodeToIndex[(Edge.node2, Edge.node1)] = 2 * i + 1
    edges.append(Edge)
# for i in range(EdgeNumber):
#     print(edges[i].node1 + 1, end=" ")
#     print(edges[i].node2 + 1, end=" ")
#     print(edges[i].cost, end=" ")
#     print(edges[i].demand)
# print(nodeToEdge.get((0,1)))
# print(nodeToEdge.get((0,1)).node1)
MAX = 1 << 32 - 1
distance_matrix = np.ones((VERTICES_NUMBER, VERTICES_NUMBER), dtype=int) * MAX
for Edge in edges:
    node1 = Edge.node1
    node2 = Edge.node2
    distance_matrix[node1][node2] = Edge.cost
    distance_matrix[node2][node1] = Edge.cost


def dijkstra(matrix, start_node):
    matrix_length = len(matrix)  # 矩阵一维数组的长度，即节点的个数

    used_node = [False] * matrix_length  # 访问过的节点数组

    distance = [MAX] * matrix_length  # 最短路径距离数组

    distance[start_node] = 0  # 初始化，将起始节点的最短路径修改成0

    # 将访问节点中未访问的个数作为循环值，其实也可以用个点长度代替。
    while used_node.count(False):
        min_value = MAX
        min_value_index = -1

        # 在最短路径节点中找到最小值，已经访问过的不在参与循环。
        # 得到最小值下标，每循环一次肯定有一个最小值
        for index in range(matrix_length):
            if not used_node[index] and distance[index] <= min_value:
                min_value = distance[index]
                min_value_index = index

        # 将访问节点数组对应的值修改成True，标志其已经访问过了
        used_node[min_value_index] = True

        # 更新distance数组。
        # 以B点为例：distance[x] 起始点达到B点的距离。
        # distance[min_value_index] + matrix[min_value_index][index] 是起始点经过某点达到B点的距离，比较两个值，取较小的那个。
        for index in range(matrix_length):
            distance[index] = min(distance[index], distance[min_value_index] + matrix[min_value_index][index])

    return distance


# dijkstra(distance_matrix,0)
Shortest_Distance = []
for i in range(VERTICES_NUMBER):
    Shortest_Distance.append(dijkstra(distance_matrix, i))

import copy


# Path Scanning
# print(VERTICES_NUMBER)
# print(Shortest_Distance)
# Rule:
# 1) maximize the distance from the task to the depot;
# 2) minimize the distance from the task to the depot;
# 3) maximize the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t, respectively;
# 4) minimize the term dem(t)/sc(t);
# 5) use rule 1) if the vehicle is less than half- full, otherwise use rule 2)
# better returns whether u1 is better than u2
def better(u1, u2, load, rule):
    node1 = u1.node1
    node2 = u2.node1
    distance1 = Shortest_Distance[node1][DEPOT - 1]
    distance2 = Shortest_Distance[node2][DEPOT - 1]
    if rule == 1:
        if distance1 >= distance2:
            return True
        else:
            return False
    elif rule == 2:
        if distance1 <= distance2:
            return True
        else:
            return False
    elif rule == 3:
        if u1.demand / u1.cost >= u2.demand / u2.cost:
            return True
        else:
            return False
    elif rule == 4:
        if u1.demand / u1.cost <= u2.demand / u2.cost:
            return True
        else:
            return False
    else:
        if load <= CAPACITY / 2:
            return True
        else:
            return False


demand_arc=[]
count=0
for u in edges:
    if u.demand!=0:
        nodeToIndex[(u.node1,u.node2)]=2*count
        nodeToIndex[(u.node2,u.node1)]=2*count+1
        demand_arc.append((u.node1,u.node2))
        demand_arc.append((u.node2,u.node1))
        count=count+1


# 需求边集free

def path_scanning(demand_arc, rule, VEHICLES):
    free = copy.deepcopy(demand_arc)
    k = 0
    routines = []
    costs = []
    while len(free) != 0 and k < VEHICLES:
        routine = []
        k = k + 1
        load = 0
        cost = 0
        i = DEPOT - 1
        d = 0
        # d=MAX
        while len(free) != 0 and d != MAX:
            d = MAX
            for item in free:
                u = nodeToEdge.get(item)
                if load + u.demand <= CAPACITY:
                    distance = Shortest_Distance[i][item[0]]
                    if distance < d:
                        d = distance
                        chosen_arc = item
                    elif distance == d and better(u, nodeToEdge.get(chosen_arc), load, rule):
                        chosen_arc = item

            if d != MAX:
                chosen_edge = nodeToEdge.get(chosen_arc)
                routine.append(chosen_arc)
                free.remove(chosen_arc)
                free.remove((chosen_arc[1], chosen_arc[0]))
                load = load + chosen_edge.demand
                cost = cost + d + chosen_edge.cost
                i = chosen_arc[1]
        cost = cost + Shortest_Distance[i][DEPOT - 1]
        routines.append(routine)
        costs.append(cost)
    # if len(free)!=0:
    #     invalid_routine.append(routines)
    # else:
    #     valid_routine.append(routines)
    return routines, sum(costs)


def fitness(routines):
    cost = 0
    for i in range(len(routines)):
        start = DEPOT - 1
        end= DEPOT-1
        for j in range(len(routines[i])):
            begin = routines[i][j][0]
            end = routines[i][j][1]
            # print(begin, end)
            cost = cost + Shortest_Distance[start][begin] + nodeToEdge.get((begin, end)).cost
            start = end
        cost = cost + Shortest_Distance[end][DEPOT - 1]
    return cost


# road=[[(1, 8), (8, 3), (3, 4), (4, 11), (11, 12), (10, 1)],
#  [(1, 11), (11, 6), (6, 4), (4, 1), (9, 1)],
#  [(1, 2), (2, 4), (4, 7), (7, 2), (2, 5), (5, 1)],
#  [(6, 5), (5, 7), (2, 3), (3, 9), (9, 8), (8, 12), (12, 10), (10, 9)]]
# fitness(road)

def random_pick():
    free=copy.deepcopy(demand_arc)
    random.shuffle(free)
    chosen=np.zeros(len(free))
    k=0
    routines=[]
    costs=[]
    while len(free)!=0 and k<VEHICLES:
        routine=[]
        k=k+1
        load=0
        cost=0
        i=DEPOT-1
        # d=MAX
        for item in free:
            index=nodeToIndex.get(item)
            if chosen[index]==0:
                u=nodeToEdge.get(item)
                if load+u.demand<=CAPACITY:
                    chosen_arc=item
                    chosen_edge=nodeToEdge.get(chosen_arc)
                    routine.append(chosen_arc)
                    load=load+chosen_edge.demand
                    cost=cost+Shortest_Distance[i][item[0]]+chosen_edge.cost
                    i=chosen_arc[1]
                    chosen[nodeToIndex.get(item)]=1
                    chosen[nodeToIndex.get((item[1],item[0]))]=1
        cost=cost+Shortest_Distance[i][DEPOT-1]
        routines.append(routine)
        costs.append(cost)
    # if len(free)!=0:
    #     invalid_routine.append(routines)
    # else:
    #     valid_routine.append(routines)
    return routines, sum(costs)


#判断是否没有做完任务
def is_Valid(routines):
    tasks = []
    for r in routines:
        for t in r:
            tasks.append(t)
            tasks.append((t[1], t[0]))
    task_set = set(tasks)
    return len(tasks) == len(task_set) and len(tasks) / 2 == REQUIRED_EDGES

# valid_routine=[]
# invalid_routine=[]
def init_population(pop_size):
    """
    Randomly initialize a population for genetic algorithm
        pop_size  :  Number of individuals in population
        gene_pool   :  List of possible values for individuals
        state_length:  The length of each individual
    """
    # global valid_routine
    # global invalid_routine
    population = []
    for i in range(5):
        solution,cost=path_scanning(demand_arc,i,VEHICLES)
        population.append(solution)
        # if is_Valid(solution):
        #     valid_routine.append(solution)
        # else:
        #     invalid_routine.append(solution)
    # for _ in range(pop_size-5):
    #     solution,cost=random_pick()
    #     if is_Valid(solution):
    #         valid_routine.append(solution)
    #     else:
    #         invalid_routine.append(solution)
    while len(population)<pop_size:
        solution,cost=random_pick()
        if is_Valid(solution):
            population.append(solution)
    # population=valid_routine+invalid_routine
    return population

population=init_population(20)




def is_overloaded(route):
    cnt = 0
    for item in route:
        u = nodeToEdge.get(item)
        cnt = cnt + u.demand
    if cnt <= CAPACITY:
        return False
    else:
        return True


def single_insertion(routine):
    minimum = fitness(routine)
    chosen = routine
    routines = copy.deepcopy(routine)
    route=[]
    while len(route)==0:
     route=random.choice(routines)
    task = random.choice(route)
    task2=(task[1],task[0])
    route.remove(task)
    for i in range(len(routines)):
        for j in range(len(routines[i])):
            routines[i].insert(j, task)
            if not is_overloaded(routines[i]) :
                f = fitness(routines)
                if f < minimum:
                    chosen = copy.deepcopy(routines)
                    minimum =f
            routines[i].remove(task)

    for i in range(len(routines)):
        for j in range(len(routines[i])):
            routines[i].insert(j, task2)
            if not is_overloaded(routines[i]) :
                f = fitness(routines)
                if f < minimum:
                    chosen = copy.deepcopy(routines)
                    minimum =f
            routines[i].remove(task2)

    return chosen
# print(population[2])
# print(fitness(population[2]))
# print(fitness(single_insertion(population[2])))
def double_insertion(routine):
    minimum = fitness(routine)
    chosen = routine
    routines = copy.deepcopy(routine)
    route = []
    while len(route) < 2:
        route = random.choice(routines)
    index = random.randint(0,len(route)-2)
    task = [route[index],route[index+1]]
    # print(task)
    task2 = [(task[1][1],task[1][0]),(task[0][1],task[0][0])]
    # print(task2)
    route.remove(task[0])
    route.remove(task[1])
    for i in range(len(routines)):
        for j in range(len(routines[i])):
            routines[i].insert(j, task[0])
            routines[i].insert(j+1,task[1])
            if not is_overloaded(routines[i]):
                f = fitness(routines)
                # print(f)
                if f < minimum:
                    chosen = copy.deepcopy(routines)
                    minimum = f
            routines[i].remove(task[0])
            routines[i].remove(task[1])

    for i in range(len(routines)):
        for j in range(len(routines[i])):
            routines[i].insert(j, task2[0])
            routines[i].insert(j+1,task2[1])
            if not is_overloaded(routines[i]):
                f = fitness(routines)
                # print(f)
                if f < minimum:
                    chosen = copy.deepcopy(routines)
                    minimum = f
            routines[i].remove(task2[0])
            routines[i].remove(task2[1])
    return chosen


#输入一个方案，任选两辆车的子图进行PS，然后返回新的方案
def Merge_Split(routine):
    routines = copy.deepcopy(routine)
    #随机选取两个小车
    route1 = random.choice(routines)
    routines.remove(route1)
    route2 = random.choice(routines)
    routines.remove(route2)
    #对两个小车构成的route进行PS
    route = []
    for item in route1:
        route.append(item)
        route.append((item[1], item[0]))
    for item in route2:
        route.append(item)
        route.append((item[1], item[0]))
    #选择5个规则下面最好的
    minimum = sys.maxsize
    chosen_routine = None
    for i in range(5):
        solution, cost = path_scanning(route, i, 2)
        if cost < minimum:
            chosen_routine = solution
            minimum = cost
    routines=routines+chosen_routine
    if fitness(routines)<fitness(routine):
        return routines
    else:
        return routine

# print(fitness(population[0]))
# print(fitness(Merge_Split(population[0])))

from time import perf_counter
mid=time.time()
pre_time=mid-start
start_time = end_time = perf_counter()
timeout=timeout-pre_time
while end_time - start_time < timeout - 2:
    individual = random.choice(population)
    new_individual = single_insertion(individual)
    new_individual = double_insertion(new_individual)
    # print(new_individual)
    new_individual = Merge_Split(new_individual)
    if is_Valid(new_individual):
        population.remove(individual)
        population.append(new_individual)
    end_time = perf_counter()


minimum=sys.maxsize
Solution=[]
for item in population:
    if is_Valid(item):
        f=fitness(item)
        # print(f)
        if f<minimum:
            minimum=f
            Solution=item

print("s", end=" ")
for i in range(len(Solution)):
    print(0, end=",")
    for j in range(len(Solution[i])):
        Solution[i][j] = (Solution[i][j][0] + 1, Solution[i][j][1] + 1)
        print(Solution[i][j], end=",")
    if (i == len(Solution) - 1 and j == len(Solution[i]) - 1):
        print(0, end="")
    else:
        print(0, end=",")
print()
print("q", end=" ")
print(minimum)
end=time.time()
print(end-start)

