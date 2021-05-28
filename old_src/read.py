import pandas as pd
import networkx as nx

students = pd.read_csv("../data/6-student_merged.csv", sep=";", dtype={'from_hei': "string", 'to_hei': "string", 'level': 'category', 'language:': "category",
                       'year': "category", 'gender': "category", 'nationality': "category", 'previous': "category", 'qualification': "category", 'languageprep': "category"})

print(students.dtypes)

#print(students)

G = nx.Graph()

#Get nodes (hei codes)
nodes_out = set(students.from_hei.unique())
nodes_in = set(students.from_hei.unique())
nodes = nodes_out.union(nodes_in)
print(len(nodes))

G.add_nodes_from(nodes_for_adding = list(nodes))

#Get edges
'''
Weighted?
Directed?
'''
#print(nodes)
grouped_st = students.groupby(['from_hei', 'to_hei'])#.size()
#print(grouped_st)
for key, el in grouped_st:
    #print(len(el)
    #Node for key[0], key[1], len(el)
    G.add_edge(key[0], key[1], weight=len(el))

#draw Graph
nx.draw(G)



