import time
import numpy as np
from getopt import getopt
import sys
args = getopt(sys.argv, "-s:-t:")[1]
file_path = args[1]
timeout = int(args[3])
seed = int(args[5])
# path="D:\PythonProject\CARP\CARP_samples\gdb10.dat"
# path="D:\PythonProject\CARP\sample.dat"
path=file_path
with open(path,"r") as f:
    content=f.readlines()
# print (content)
NAME=content[0].split(":")[1]
VERTICES_NUMBER=(int)(content[1].split(":")[1])
DEPOT=(int)(content[2].split(":")[1])
REQUIRED_EDGES=(int)(content[3].split(":")[1])
NON_REQUIRED_EDGES=(int)(content[4].split(":")[1])
VEHICLES=(int)(content[5].split(":")[1])
CAPACITY=(int)(content[6].split(":")[1])
RequiredEdgeCost=(int)(content[7].split(":")[1])
EdgeNumber=REQUIRED_EDGES+NON_REQUIRED_EDGES

class edge:
    def __init__(self,node1,node2,cost,demand):
        self.node1=node1
        self.node2=node2
        self.cost=cost
        self.demand=demand

edges=[]
nodeToEdge={}
for i in range(EdgeNumber):
    content[9+i]=content[9+i].split()
    Edge=edge((int)(content[9+i][0])-1,(int)(content[9+i][1])-1,(int)(content[9+i][2]),(int)(content[9+i][3]))
    nodeToEdge[(Edge.node1,Edge.node2)]=Edge
    nodeToEdge[(Edge.node2,Edge.node1)]=Edge
    edges.append(Edge)
# for i in range(EdgeNumber):
#     print(edges[i].node1+1,end=" ")
#     print(edges[i].node2+1,end=" ")
#     print(edges[i].cost,end=" ")
#     print(edges[i].demand)
# print(nodeToEdge.get((0,1)))
# print(nodeToEdge.get((0,1)).node1)
MAX = 1<<32-1
distance_matrix=np.ones((VERTICES_NUMBER,VERTICES_NUMBER),dtype=int)*MAX
for Edge in edges:
    node1=Edge.node1
    node2=Edge.node2
    distance_matrix[node1][node2]=Edge.cost
    distance_matrix[node2][node1]=Edge.cost

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
Shortest_Distance=[]
for i in range(VERTICES_NUMBER):
    Shortest_Distance.append(dijkstra(distance_matrix,i))
#Path Scanning
# print(VERTICES_NUMBER)
# print(Shortest_Distance)
#Rule:
# 1) maximize the distance from the task to the depot;
# 2) minimize the distance from the task to the depot;
# 3) maximize the term dem(t)/sc(t), where dem(t) and sc(t) are demand and serving cost of task t, respectively;
# 4) minimize the term dem(t)/sc(t);
# 5) use rule 1) if the vehicle is less than half- full, otherwise use rule 2)
#better returns whether u1 is better than u2
def better(u1,u2,load,rule):
    node1=u1.node1
    node2=u2.node1
    distance1=Shortest_Distance[node1][DEPOT-1]
    distance2=Shortest_Distance[node2][DEPOT-1]
    if rule==1:
        if distance1 >=distance2:
            return True
        else:
            return False
    elif rule==2:
        if distance1 <=distance2:
            return True
        else:
            return False
    elif rule==3:
        if u1.demand/u1.cost >=u2.demand/u2.cost:
            return True
        else:
            return False
    elif rule==4:
         if u1.demand/u1.cost <=u2.demand/u2.cost:
            return True
         else:
            return False
    else:
        if load<=CAPACITY/2:
            return True
        else:
            return False



#需求边集free
def path_scanning(rule):
    free=[]
    k=0
    for u in edges:
        if u.demand!=0:
            free.append((u.node1,u.node2))
            free.append((u.node2,u.node1))
    routines=[]
    costs=[]
    while len(free)!=0 and k<VEHICLES:
        routine=[]
        k=k+1
        load=0
        cost=0
        i=DEPOT-1
        d=0
        # d=MAX
        while len(free)!=0 and d!=MAX :
            d=MAX
            for item in free :
                u=nodeToEdge.get(item)
                if load+u.demand<=CAPACITY:
                    distance=Shortest_Distance[i][item[0]]
                    if distance<d:
                        d=distance
                        chosen_arc=item
                    elif distance==d and better(u,nodeToEdge.get(chosen_arc),load,rule):
                        chosen_arc=item

            if d!=MAX:
                chosen_edge=nodeToEdge.get(chosen_arc)
                routine.append(chosen_arc)
                free.remove(chosen_arc)
                free.remove((chosen_arc[1],chosen_arc[0]))
                load=load+chosen_edge.demand
                cost=cost+d+chosen_edge.cost
                i=chosen_arc[1]
        cost=cost+Shortest_Distance[i][DEPOT-1]
        routines.append(routine)
        costs.append(cost)
    if len(free)!=0:
        return -1
    return routines, sum(costs)

# print(routines)
# print(sum(costs))
Solution, Cost = [], MAX
for i in range(5):
    solution,cost=path_scanning(i)
    if cost<Cost:
        Solution=solution
        Cost=cost
print("s",end=" ")
for i in range(len(Solution)):
    print(0,end=",")
    for j in range(len(Solution[i])):
        Solution[i][j]=(Solution[i][j][0]+1,Solution[i][j][1]+1)
        print(Solution[i][j],end=",")
    if(i==len(Solution)-1 and j==len(Solution[i])-1):
        print(0,end="")
    else:
        print(0,end=",")
print()
print("q",end=" ")
print(Cost)
# for i in range(EdgeNumber):
#     print(edges[i].node1,end=" ")
#     print(edges[i].node2,end=" ")
#     print(edges[i].cost,end=" ")
#     print(edges[i].demand)

# print(NAME)
# print(VERTICES_NUMBER)
# print(DEPOT)
# print(REQUIRED_EDGES)
# print(NON_REQUIRED_EDGES)
# print(VEHICLES)
# print(CAPACITY)
# print(RequiredEdgeCost)

# content[1]
run_time=time.time()-start
# for i in range(EdgeNumber):
#     print(edges[i].node1,end=" ")
#     print(edges[i].node2,end=" ")
#     print(edges[i].cost,end=" ")
#     print(edges[i].demand)

# print(NAME)
# print(VERTICES_NUMBER)
# print(DEPOT)
# print(REQUIRED_EDGES)
# print(NON_REQUIRED_EDGES)
# print(VEHICLES)
# print(CAPACITY)
# print(RequiredEdgeCost)

# content[1]