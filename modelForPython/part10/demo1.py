'''
画邻接矩阵
0 9 2 4 7
9 0 3 4 0
2 3 0 8 4
4 4 8 0 6
7 0 4 6 0
顶点号从1开始
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# a=np.zeros((5,5))
# a[0,1:5]=[9, 2, 4, 7]; a[1,2:4]=[3,4]
# a[2,[3,4]]=[8, 4]; #输入邻接矩阵的上三角元素
# a[3,4]=6; print(a); np.savetxt("Pdata10_2.txt",a) #保存邻接矩阵供以后使用
# i,j=np.nonzero(a)  #提取顶点的编号
# w=a[i,j]  #提出a中的非零元素
# edges=list(zip(i,j,w))
# G=nx.Graph()
# G.add_weighted_edges_from(edges)
# key=range(5); s=[str(i+1) for i in range(5)]
# s=dict(zip(key,s))  #构造用于顶点标注的字符字典
# plt.rc('font',size=18)
# plt.subplot(121); nx.draw(G,font_weight='bold',labels=s)
# plt.subplot(122); pos=nx.shell_layout(G)  #布局设置
# nx.draw_networkx(G,pos,node_size=260,labels=s)
# w = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,font_size=12,edge_labels=w) #标注权重
# plt.savefig("figure10_2.png", dpi=500); plt.show()

'''
用列表画无向图
'''
# import networkx as nx
# import pylab as plt
# import numpy as np
# List=[(1,2,9),(1,3,2),(1,4,4),(1,5,7),
#   (2,3,3),(2,4,4),(3,4,8),(3,5,4),(4,5,6)]
# G=nx.Graph()
# G.add_nodes_from(range(1,6))
# G.add_weighted_edges_from(List)
# pos=nx.shell_layout(G)
# w = nx.get_edge_attributes(G,'weight')
# nx.draw(G, pos,with_labels=True, font_weight='bold',font_size=12)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=w)
# plt.show()

'''
画有向图
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# G=nx.DiGraph()
# List=[(1,2),(1,3),(2,3),(3,2),(3,5),(4,2),(4,6),
#       (5,2),(5,4),(5,6),(6,5)]
# G.add_nodes_from(range(1,7))
# G.add_edges_from(List)
# plt.rc('font',size=16)
# pos=nx.shell_layout(G) 
# nx.draw(G,pos,with_labels=True, font_weight='bold',node_color='r')
# plt.savefig("figure10_3.png", dpi=500); plt.show()


'''
查看图的各种信息
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# a=np.loadtxt("Pdata10_2.txt")
# G=nx.Graph(a)     #利用邻接矩阵构造赋权无向图
# print("图的顶点集为：", G.nodes(),"\n边集为：", G.edges())
# print("邻接表为：", list(G.adjacency()))  #显示图的邻接表
# print("列表字典为：", nx.to_dict_of_lists(G)) 
# B=nx.to_numpy_matrix(G)  #从图G中导出邻接矩阵B，这里B=a
# C=nx.to_scipy_sparse_matrix(G)  #从图G中导出稀疏矩阵C

'''
求最短路径,Dijkstra算法
输入为一个邻接矩阵,顶点号从0开始
'''
# import numpy as np
# inf=np.inf
# def Dijkstra_all_minpath( matr,start): #matr为邻接矩阵的数组，start表示起点
#     n=len( matr) #该图的节点数
#     dis=[]; temp=[]
#     dis.extend(matr[start])  #添加数组matr的start行元素
#     temp.extend(matr[start]) #添加矩阵matr的start行元素
#     temp[start] = inf  #临时数组会把处理过的节点的值变成inf
#     visited=[start]  #start已处理
#     parent=[start]*n   #用于画路径，记录此路径中该节点的父节点
#     while len(visited)<n:
#         i= temp.index(min(temp)) #找最小权值的节点的坐标
#         temp[i]=inf
#         for j in range(n):
#             if j not in visited:
#                 if (dis[i]+ matr[i][j])<dis[j]:
#                     dis[j] = temp[j] =dis[i]+ matr[i][j]
#                     parent[j]=i  #说明父节点是i
#         visited.append(i)  #该索引已经处理了
#         path=[]  #用于画路径
#         path.append(str(i))
#         k=i
#         while(parent[k]!=start):  #找该节点的父节点添加到path，直到父节点是start
#             path.append(str(parent[k]))
#             k=parent[k]
#         path.append(str(start))
#         path.reverse()   #path反序产生路径
#         print(str(i)+':','->'.join(path))  #打印路径
#     return dis
# a=[[0,1,2,inf,7,inf,4,8],[1,0,2,3,inf,inf,inf,7],
#   [2,2,0,1,5,inf,inf,inf],[inf,3,1,0,3,6,inf,inf],
#   [7,inf,5,3,0,4,3,inf],[inf,inf,inf,6,4,0,6,4],
#   [4,inf,inf,inf,3,6,0,2],[8,7,inf,inf,inf,4,2,0]]
# d=Dijkstra_all_minpath(a,3)
# print("v3到所有顶点的最短距离为：",d)


'''
调用networkx的最短路径函数
输入为列表
'''
# import numpy as np
# import networkx as nx
# List=[(0,1,1),(0,2,2),(0,4,7),(0,6,4),(0,7,8),(1,2,2),(1,3,3),
#       (1,7,7),(2,3,1),(2,4,5),(3,4,3),(3,5,6),(4,5,4),(4,6,3),
#       (5,6,6),(5,7,4),(6,7,2)]
# G=nx.Graph()
# G.add_weighted_edges_from(List)
# A=nx.to_numpy_matrix(G, nodelist=range(8))  #导出邻接矩阵
# np.savetxt('Pdata10_6.txt',A)
# p=nx.dijkstra_path(G, source=3, target=7, weight='weight')  #求最短路径；
# d=nx.dijkstra_path_length(G, 3, 7, weight='weight') #求最短距离
# print("最短路径为：",p,"；最短距离为：",d)


'''
Floyd算法,也是算最短路径,只不过是输出所有顶点对的信息
路由矩阵Rk用来记录两点间路径的前驱后继关系
'''
# import numpy as np
# def floyd(graph):
#     m = len(graph)
#     dis = graph
#     path = np.zeros((m, m))  #路由矩阵初始化
#     for k in range(m):
#         for i in range(m):
#             for j in range(m):
#                 if dis[i][k] + dis[k][j] < dis[i][j]:
#                     dis[i][j] = dis[i][k] + dis[k][j]
#                     path[i][j] = k
#     return dis, path
# inf=np.inf
# a=np.array([[0,1,2,inf,7,inf,4,8],[1,0,2,3,inf,inf,inf,7],
#   [2,2,0,1,5,inf,inf,inf],[inf,3,1,0,3,6,inf,inf],
#   [7,inf,5,3,0,4,3,inf],[inf,inf,inf,6,4,0,6,4],
#   [4,inf,inf,inf,3,6,0,2],[8,7,inf,inf,inf,4,2,0]])  #输入邻接矩阵
# dis, path=floyd(a)
# print("所有顶点对之间的最短距离为：\n", dis, '\n',"路由矩阵为：\n",path)


'''
调用networkx来求所有顶点对的最短路径
'''
# import numpy as np
# import networkx as nx
# a=np.loadtxt("Pdata10_6.txt")
# G=nx.Graph(a)     #利用邻接矩阵构造赋权无向图
# d=nx.shortest_path_length(G,weight='weight')  #返回值是可迭代类型
# Ld=dict(d)  #转换为字典类型
# print("顶点对之间的距离为：",Ld)  #显示所有顶点对之间的最短距离
# print("顶点0到顶点4的最短距离为:",Ld[0][4])  #显示一对顶点之间的最短距离
# m,n=a.shape; dd=np.zeros((m,n))
# for i in range(m):
#     for j in range(n): dd[i,j]=Ld[i][j]
# print("顶点对之间最短距离的数组表示为：\n",dd)  #显示所有顶点对之间最短距离
# np.savetxt('Pdata10_8.txt',dd) #把最短距离数组保存到文本文件中
# p=nx.shortest_path(G, weight='weight')  #返回值是可迭代类型
# dp=dict(p)  #转换为字典类型
# print("\n顶点对之间的最短路径为：", dp)
# print("顶点0到顶点4的最短路径为：",dp[0][4])


'''
处理价值问题,优化策略,如对于价值减少,购入新机器,维修费用的考量问题
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# p=[25,26,28,31]; a=[10,14,18,26]; r=[20,16,13,11];
# b=np.zeros((5,5)); #邻接矩阵（非数学上的邻接矩阵）初始化
# for i in range(5):
#     for j in range(i+1,5):
#         b[i,j]=p[i]+np.sum(a[0:j-i])-r[j-i-1];
# print(b)
# G=nx.DiGraph(b)
# p=nx.dijkstra_path(G, source=0, target=4, weight='weight')  #求最短路径；
# print("最短路径为:",np.array(p)+1)  #python下标从0开始
# d=nx.dijkstra_path_length(G, 0, 4, weight='weight') #求最短距离
# print("所求的费用最小值为：",d)
# s=dict(zip(range(5),range(1,6))) #构造用于顶点标注的标号字典
# plt.rc('font',size=16)
# pos=nx.shell_layout(G)  #设置布局
# w=nx.get_edge_attributes(G,'weight')
# nx.draw(G,pos,font_weight='bold',labels=s,node_color='r')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=w)
# path_edges=list(zip(p,p[1:]))
# nx.draw_networkx_edges(G,pos,edgelist=path_edges,
#             edge_color='r',width=3)
# plt.savefig("figure10_9.png",pdi=500); plt.show()


'''
处理给定一组点,选取一点,使得各个点到其距离最短的问题
'''
# import numpy as np
# import networkx as nx
# List=[(1,2,20),(1,5,15),(2,3,20),(2,4,40),
#       (2,5,25),(3,4,30),(3,5,10),(5,6,15)]
# G=nx.Graph()
# G.add_nodes_from(range(1,7))
# G.add_weighted_edges_from(List)
# c=dict(nx.shortest_path_length(G,weight='weight'))
# d=np.zeros((6,6))
# for i in range(1,7):
#     for j in range(1,7): d[i-1,j-1]=c[i][j]
# print(d)
# q=np.array([80,90,30,20,60,10])
# m=d@q  #计算运力，这里使用矩阵乘法
# mm=m.min()  #求运力的最小值
# ind=np.where(m==mm)[0]+1  #python下标从0开始，np.where返回值为元组
# print("运力m=",m,'\n最小运力mm=',mm,"\n选矿厂的设置位置为：",ind)


'''
求最小生成树
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# L=[(1,2,8),(1,3,4),(1,5,2),(2,3,4),(3,4,2),(3,5,1),(4,5,5)]
# b=nx.Graph()
# b.add_nodes_from(range(1,6))
# b.add_weighted_edges_from(L)
# T=nx.minimum_spanning_tree(b)  #返回可迭代对象
# w=nx.get_edge_attributes(T,'weight') #提取字典数据
# TL=sum(w.values())  #计算最小生成树的长度
# print("最小生成树为:",w)
# print("最小生成树的长度为：",TL)
# pos=nx.shell_layout(b)
# nx.draw(T,pos,node_size=280,with_labels=True,node_color='r')
# nx.draw_networkx_edge_labels(T,pos,edge_labels=w)
# plt.show()


'''
用最小生成树来处理,连通各个节点,并使路线最短
'''
# import numpy as np
# import networkx as nx
# import pandas as pd
# import pylab as plt
# a=pd.read_excel("Pdata10_14.xlsx",header=None)
# b=a.values; b[np.isnan(b)]=0
# c=np.zeros((8,8))  #邻接矩阵初始化
# c[0:7,1:8]=b  #构造图的邻接矩阵
# G=nx.Graph(c)
# T=nx.minimum_spanning_tree(G)  #返回可迭代对象
# d=nx.to_numpy_matrix(T)  #返回最小生成树的邻接矩阵
# print("邻接矩阵c=\n",d)
# W=d.sum()/2+5  #求油管长度
# print("油管长度W=",W)
# s=dict(zip(range(8),range(1,9))) #构造用于顶点标注的标号字典
# plt.rc('font',size=16); pos=nx.shell_layout(G)
# nx.draw(T,pos,node_size=280,labels=s,node_color='r')
# w=nx.get_edge_attributes(T,'weight')
# nx.draw_networkx_edge_labels(T,pos,edge_labels=w)
# plt.savefig('figure10_14.png'); plt.show()


'''
给定一个矩阵,ij代表第i个员工对应第j项工作的效益,
计算各种匹配的可能的最大效益
'''
# import numpy as np
# import networkx as nx
# from networkx.algorithms.matching import max_weight_matching
# a=np.array([[3,5,5,4,1],[2,2,0,2,2],[2,4,4,1,0],
#             [0,2,2,1,0],[1,2,1,3,3]])
# b=np.zeros((10,10)); b[0:5,5:]=a; G=nx.Graph(b)
# s0=max_weight_matching(G)  #返回值为（人员，工作）的集合
# s=[sorted(w) for w in s0]
# L1=[x[0] for x in s]; L1=np.array(L1)+1  #人员编号
# L2=[x[1] for x in s]; L2=np.array(L2)-4  #工作编号
# c=a[L1-1,L2-1]  #提取对应的效益
# d=c.sum()  #计算总的效益
# print("工作分配对应关系为：\n人员编号：",L1)
# print("工作编号：", L2); print("总的效益为：",d)


'''
求图的最大流
'''
# import numpy as np
# import networkx as nx
# import pylab as plt
# L=[(1,2,5),(1,3,3),(2,4,2),(3,2,1),(3,5,4),
#    (4,3,1),(4,5,3),(4,6,2),(5,6,5)]
# G=nx.DiGraph()
# for k in range(len(L)):
#     G.add_edge(L[k][0]-1,L[k][1]-1, capacity=L[k][2])
# value, flow_dict= nx.maximum_flow(G, 0, 5)
# print("最大流的流量为：",value)
# print("最大流为：", flow_dict)
# n = len(flow_dict)
# adj_mat = np.zeros((n, n), dtype=int)
# for i, adj in flow_dict.items():
#     for j, weight in adj.items():
#         adj_mat[i,j] = weight
# print("最大流的邻接矩阵为：\n",adj_mat)
# ni,nj=np.nonzero(adj_mat)  #非零弧的两端点编号
# key=range(n)
# s=['v'+str(i+1) for i in range(n)]
# s=dict(zip(key,s)) #构造用于顶点标注的字符字典
# plt.rc('font',size=16)
# pos=nx.shell_layout(G)  #设置布局
# w=nx.get_edge_attributes(G,'capacity')
# nx.draw(G,pos,font_weight='bold',labels=s,node_color='r')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=w)
# path_edges=list(zip(ni,nj))
# nx.draw_networkx_edges(G,pos,edgelist=path_edges,
#             edge_color='r',width=3)
# plt.show()


'''
求图的最小流
'''
# import numpy as np
# import networkx as nx
# L=[(1,2,5,3),(1,3,3,6),(2,4,2,8),(3,2,1,2),(3,5,4,2),
#    (4,3,1,1),(4,5,3,4),(4,6,2,10),(5,6,5,2)]
# G=nx.DiGraph()
# for k in range(len(L)):
#     G.add_edge(L[k][0]-1,L[k][1]-1, capacity=L[k][2], weight=L[k][3])
# mincostFlow=nx.max_flow_min_cost(G,0,5)
# print("所求流为：",mincostFlow)
# mincost=nx.cost_of_flow(G, mincostFlow)
# print("最小费用为：", mincost)
# flow_mat=np.zeros((6,6),dtype=int)
# for i,adj in mincostFlow.items():
#     for j,f in adj.items():
#         flow_mat[i,j]=f
# print("最小费用最大流的邻接矩阵为：\n",flow_mat)













