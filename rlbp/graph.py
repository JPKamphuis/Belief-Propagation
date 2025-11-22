import numpy  as np
import igraph as ig
import pyvis.network as net
import re
import webbrowser
from pathlib import Path
from factor import *

class factor_graph:
    def __init__(self):
        self._graph = ig.Graph()
    
    # ----------------------- Factor node functions ---------
    def add_factor_node(self, f_name, factor_):
        if (self.get_node_status(f_name) != False) or (f_name in factor_.get_variables()):
            raise Exception('Invalid factor name')
        if type(factor_) is not factor:
            raise Exception('Invalid factor_')
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == 'factor':
                raise Exception('Invalid factor')
        
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 1)
        # Create variables
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == False:
                self.__create_variable_node(v_name)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Add node and corresponding edges
        self.__create_factor_node(f_name, factor_)
        
    def change_factor_distribution(self, f_name, factor_):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        if set(factor_.get_variables()) != set(self._graph.vs[self._graph.neighbors(f_name)]['name']):
            raise Exception('invalid factor distribution')
        
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 0)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Set data
        self._graph.vs.find(name=f_name)['factor_'] = factor_


    def remove_factor(self, f_name, remove_zero_degree=False):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        
        neighbors = self._graph.neighbors(f_name, mode="out")
        self._graph.delete_vertices(f_name)
        
        if remove_zero_degree:
            for v_name in neighbors:
                if self._graph.vs.find(v_name).degree() == 0:
                    self.remove_variable(v_name)


    def __create_factor_node(self, f_name, factor_):
        # Create node
        self._graph.add_vertex(f_name)
        self._graph.vs.find(name=f_name)['is_factor'] = True
        self._graph.vs.find(name=f_name)['factor_']   = factor_
        
        # Create corresponding edges
        start = self._graph.vs.find(name=f_name).index
        edge_list = [tuple([start, self._graph.vs.find(name=i).index]) for i in factor_.get_variables()]
        self._graph.add_edges(edge_list)

    def apply_evidence(self, evidence_dict):
        for s, dist in evidence_dict.items():
            vname = f"V{s}"     # variable node name
            fname = f"E{s}"     # evidence factor name

            # Build unary evidence factor
            ev_factor = factor([vname], np.array(dist))

            # If the evidence factor already exists, update it
            if self.get_node_status(fname) == "factor":
                self.change_factor_distribution(fname, ev_factor)
            else:
                # Otherwise create a new unary factor node
                self.add_factor_node(fname, ev_factor)
    
    # ----------------------- Rank functions -------
    def __check_variable_ranks(self, f_name, factor_, allowded_v_degree):
        for counter, v_name in enumerate(factor_.get_variables()):
            if (self.get_node_status(v_name) == 'variable') and (not factor_.is_none()):
                if     (self._graph.vs.find(name=v_name)['rank'] != factor_.get_shape()[counter]) \
                and (self._graph.vs.find(name=v_name)['rank'] != None) \
                and (self._graph.vs.find(v_name).degree() > allowded_v_degree):
                    raise Exception('Invalid shape of factor_')


    def __set_variable_ranks(self, f_name, factor_):
        for counter, v_name in enumerate(factor_.get_variables()):
            if factor_.is_none():
                self._graph.vs.find(name=v_name)['rank'] = None
            else:
                self._graph.vs.find(name=v_name)['rank'] = factor_.get_shape()[counter]


        
    # ----------------------- Variable node functions -------
    def add_variable_node(self, v_name):
        if self.get_node_status(v_name) != False:
            raise Exception('Node already exists')
        self.__create_variable_node(v_name)


    def remove_variable(self, v_name):
        if self.get_node_status(v_name) != 'variable':
            raise Exception('Invalid variable name')
        if self._graph.vs.find(v_name).degree() != 0:
            raise Exception('Can not delete variables with degree >0')
        self._graph.delete_vertices(self._graph.vs.find(v_name).index)  


    def __create_variable_node(self, v_name, rank=None):
        self._graph.add_vertex(v_name)
        self._graph.vs.find(name=v_name)['is_factor'] = False
        self._graph.vs.find(name=v_name)['rank'] = rank


    # ----------------------- Info --------------------------
    def get_node_status(self, name):
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            if self._graph.vs.find(name=name)['is_factor'] == True:
                return 'factor'
            else:
                return 'variable'


    
    # ----------------------- Graph structure ---------------
    def get_graph(self):
        return self._graph


    def is_connected(self):
        return self._graph.is_connected()


    def is_loop(self):
        return any(self._graph.is_loop())

# ----------------------- Utility functions -----------------
def string2factor_graph(str_):
    res_factor_graph = factor_graph()
    
    str_ = [i.split('(') for i in str_.split(')') if i != '']
    for i in range(len(str_)):
        str_[i][1] = str_[i][1].split(',')
        
    for i in str_:
        res_factor_graph.add_factor_node(i[0], factor(i[1]))
    
    return res_factor_graph

def plot_factor_graph(x):
    graph = net.Network(notebook=True, width="100%")
    graph.toggle_physics(False)
    
    # Vertices
    label = x.get_graph().vs['name']
    color = ['#2E2E2E' if i is True else '#F2F2F2' for i in x.get_graph().vs['is_factor']]
    graph.add_nodes(range(len(x.get_graph().vs)), label=label, color=color)
    
    # Edges
    graph.add_edges(x.get_graph().get_edgelist())
    
    return graph.show("./graph.html")

# def plot_grid_factor_graph(fg):
#     net_vis = net.Network(notebook=True, width="100%")
#     net_vis.toggle_physics(False)

#     g = fg.get_graph()
#     vs = g.vs
#     edges = g.get_edgelist()

#     # Compute positions
#     pos = {}
#     rows, cols = 0, 0

#     # First, find rows and cols from variable node names
#     for v in vs:
#         name = v['name']
#         if not v['is_factor']:
#             m = re.match(r"V(\d+)(\d+)", name)
#             if m:
#                 r, c = int(m.group(1)), int(m.group(2))
#                 pos[name] = (c, r)  # x=c, y=r
#                 rows = max(rows, r+1)
#                 cols = max(cols, c+1)

#     # Place factor nodes midway between their connected variables
#     for v in vs:
#         if v['is_factor']:
#             f_name = v['name']
#             # get connected nodes
#             connected = [vs[e.source]['name'] if vs[e.source]['name'] != f_name else vs[e.target]['name']
#                          for e in g.es if f_name in [vs[e.source]['name'], vs[e.target]['name']]]
#             if len(connected) == 2:
#                 x1, y1 = pos[connected[0]]
#                 x2, y2 = pos[connected[1]]
#                 pos[f_name] = ((x1 + x2)/2, (y1 + y2)/2)
#             else:
#                 pos[f_name] = (0, 0)

#     # Add nodes
#     for i, v in enumerate(vs):
#         name = v['name']
#         label = name
#         color = '#2E2E2E' if not v['is_factor'] else '#F2F2F2'
#         x, y = pos[name]
#         # scale y to invert (optional, so V_0_0 is bottom-left)
#         net_vis.add_node(i, label=label, color=color, x=x*150, y=y*150)

#     # Add edges
#     net_vis.add_edges(edges)

#     tmp_file = Path("./graph.html")
#     webbrowser.open(tmp_file.resolve().as_uri())
#     return

# def create_grid_world_factor_graph(rows, cols, alpha, beta, num_actions=4):
#     """
#     Create a grid-world factor graph with a lattice structure.
#     Each cell is a variable node with `num_actions` categories (default 4 for up, down, left, right).
#     Pairwise factors connect neighboring cells to encourage similar actions using Potts model.
#     """
#     fg = factor_graph()

#     # Helper to get variable names
#     def varname(r, c):
#         return f"V{r}{c}"

#     # Create all variable nodes first with known rank
#     for r in range(rows):
#         for c in range(cols):
#             fg.add_variable_node(varname(r, c))
#             fg._graph.vs.find(name=varname(r, c))["rank"] = num_actions

#     # Potts smoothing factor
#     pairwise_dist = beta * np.ones((4,4))
#     np.fill_diagonal(pairwise_dist, alpha)
    
#     # Add factors for horizontal and vertical neighbors
#     for r in range(rows):
#         for c in range(cols):
#             v = varname(r, c)

#             # Right neighbor
#             if c + 1 < cols:
#                 v2 = varname(r, c+1)
#                 f_name = f"F{r}{c}_{r}{c+1}"
#                 fg.add_factor_node(f_name, factor([v, v2], pairwise_dist))

#             # Down neighbor
#             if r + 1 < rows:
#                 v2 = varname(r+1, c)
#                 f_name = f"F{r}{c}_{r+1}{c}"
#                 fg.add_factor_node(f_name, factor([v, v2], pairwise_dist))
#     return fg

def print_graph_info(fg):
    g = fg.get_graph()
    for v in g.vs:
        name = v['name']
        if v['is_factor']:
            f = v['factor_']
            print(f"Factor node '{name}':")
            print(f"  Variables: {f.get_variables()}")
            print(f"  Distribution:\n{f.get_distribution()}")
        else:
            print(f"Variable node '{name}':")
            print(f"  Rank (number of categories): {v['rank']}")
    print(f"Total nodes: {len(g.vs)}")
    print(f"Total edges: {len(g.es)}")