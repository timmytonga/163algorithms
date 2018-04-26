from collections import defaultdict
import math 

class GraphException(Exception):
    def __init__(self, message=""):
        print("ERROR: " + message)

class Graph:
    ''' Represent a graph object. Undirected by default. Initialize empty graph '''
######################## SETUP AND UTILITIES ############################### 
    def __init__(self, graph=defaultdict(set), weightDict={}, vertices=set(),transitionMatrix=[], isDirected = False ):
        ''' Input edgeDict --> get vertices from there.... '''
        self.directed = isDirected  # directed or not
        self.isWeighted = False     # default non weighted
        if len(weightDict) > 0:
            self.isWeighted = True
        self.vertices = vertices    # {'A', 'B', 'C', ... }
        self.graph = graph          # {'A':['B','C'], ... } 
        self.weights = weightDict   # {('A','B'): 24, ... }
        for i in self.graph:
            self.vertices.add(i)

    def add_vertex(self, v):
        if v in self.graph:
            raise GraphException("add_vertex: Vertex " + v + "already in graph")
        self.vertices.add(v)
        self.graph[v] = {}
    
    def add_edge(self, source, dest):
        if source not in self.graph:
            self.add_vertex(source)
        if dest not in self.graph:
            self.add_vertex(dest)
        self.graph[source].append(dest)
        if not self.directed:
            self.graph[dest].append(source)

    def degree(self, vertex):
        ''' return the degree (in + out deg) of that vertex '''
        pass

    def in_degree(self, vertex):
        '''Remove the number of indegree of a vertex
            only for directed graph... return degree for undirected '''
        pass

    def out_degree(self, vertex):
        ''' Remove the number of outdegree edges of a vertex '''
        pass

    ##################### ALGORITHMS #############################
    ''' See NOTES regarding algorithms runnability on directed vs. undirected graphs '''
    def mst_prim(self, seed):
        pass

    def mst_boruvka(self):
        pass

    def mst_kruskal(self):
        pass

    def dfs(self, start):
        ''' return a list of reachable vertices from start '''
        if start not in self.graph:
            raise GraphException("dfs invalid start vertex")
        visited = set()
        def visit(w):
            visited.add(w)
            for dest in self.graph[w]:
                if dest not in visited:
                    visit(dest)
        visit(start)
        return visited

    def tarjan_strongly_connected_components(self):
        ''' return a list of sets of strongly connected components'''
        pass

    def bfs(self, start):
        ''' Return all reachable vertex from start using shortest path
            using bfs traversal '''
        if start not in self.graph:
            raise GraphException("Invalid start vertex")
        reached = { start }
        q = [start]
        while len(q) != 0:
            v = q.pop(0)
            for dest in self.graph[v]:
                if dest not in reached:
                    reached.add(dest)
                    q.append(dest)
        return reached     

    def topological_sort(self):
        ''' Directed graphs only
        Return the reversed postordering of a graph '''
        visited = set()
        postorder = [] 
        def visit(v):
            visited.add(v)
            for w in self.graph[v]:
                if w not in visited:
                    visit(w)
            postorder.append(v)     # this adds 
        for v in self.graph:
            if v not in visited:
                visit(v)
        return postorder[::-1]   # we want to return the reversed list for toporder

    def weight(self, u, v):
        ''' return the weight of an edge if exists in the provided dict
        else raises an error '''
        if (u,v) in self.weights:
            return self.weights[u,v]
        elif (v,u) in self.weights:
            return self.weights[v,u]
        else:
            raise GraphException("Weight does not exist for edge " + u + " " + v)

    def relax(self, u, v , D, P):
        ''' relax given an edge (u,v) and update D and P accordingly
        D is the distance to v and P is the previous vertex to v '''
        if D[u] + self.weight(u,v) < D[v]:
            D[v] = D[u] + self.weight(u,v)
            P[v] = u

    def dag_shortest_path(self, start):
        if start not in self.graph:
            raise GraphException("Invalid start vertex!")
        inf = math.inf
        D = {} # D[v] = distance from start to vertex v 
        P = {} # P[v] = predecessor of v on that path
        # initialize all the initial distances to be infinity 
        for vertex in self.graph:
            P[vertex] = None
            D[vertex] = inf
            if vertex == start:
                D[vertex] = 0
        toporder = self.topological_sort() # this graph in topological order
        for u in toporder:
            for v in self.graph[u]: # for each edge uv... relax
                if D[u] + self.weight(u,v) < D[v]: # this is relax
                    D[v] = D[u] + self.weight(u,v)
                    P[v] = u
        return D, P 
        
        
    def dijkstra(self, start):
        ''' Compute shortest path from a given start vertex
        NOTE: NO NEGATIVE EDGE WEIGHTS -- use bellman_ford instead '''
        inf = math.inf
        D = {} # D[v] = distance from start to vertex v 
        P = {} # P[v] = predecessor of v on that path
        # initialize all the initial distances to be infinity 
        for vertex in self.graph:
            P[vertex] = None
            D[vertex] = inf
            if vertex == start:
                D[vertex] = 0
        return D, P 
        

    def bellman_ford(self, start):
        ''' Compute shortest path from a given start vertex
        NO NEGATIVE CYCLES ''' 
        pass

    def astar(self, start, goal, h):
        ''' compute shortest path from start to goal FASTER than Dijkstra
        h is a heuristic function that returns the estimated cost to a point from start ''' 
        pass

    def johnson(self):
        ''' All pairs shortest paths '''
        pass

    #################################### OTHER UTILITIES ####################
    def draw_graph(self):
        ''' somehow draw a graph... maybe on a new window with canvas '''
        pass
    def __str__(self):
        return str(self.graph)
    
    def construct_path(self, start, end, P):
        u = None
        q = [end]
        v = end
        while u != start:
            v = P[v]
            q.insert(0,v)
            if v == start:
                return q
            elif v == None:
                raise GraphException("cannot find path...")
            
        

def random_graph(numberVertices, edgeDensity, directed=False, weights=False):
    ''' make a random graph with given number of vertices and number of edges
    as a fraction of a complete graph. vertices are named '1', '2', ...
    NOTE: edgeDensity must be between 0 and 1
    Return a graph object.'''
    pass

if __name__=="__main__":
    ''' test algorithms here '''
    graph1 = { 'a' : ['e','b','c'],
               'b' : ['e','d'],
               'c' : ['b'],
               'd' : [],
               'e' : [],
               'f' : ['c', 'd', 'h'],
               'g' : ['f','h'],
               'h' : []}
    weightedGraph1 = { 's': ['a','d','h'],
                       'a': ['b'],
                       'b': ['c','f'],
                       'c': ['g'],
                       'd': ['e'],
                       'e': ['c'],
                       'f': ['g'],
                       'h': ['e','i'],
                       'i': ['f','j'],
                       'j': ['end'],
                       'g': ['end'],
                       'end': []
                    }
    weightedGraph1weights = {('s','a'):3, ('s','d'):1, ('s','h'):2,
                            ('a','b'):4, ('b','c'):1, ('b','f'):2,
                            ('c','g'):7, ('d','e'):3, ('e','c'):5,
                            ('f','g'):2, ('g', 'end'):2, ('h','e'):4,
                            ('h','i'):6, ('i', 'f'):3, ('i','j'):4,
                            ('j','end'):1}
    g1 = Graph(graph1)
    wg2 = Graph(weightedGraph1, weightedGraph1weights, isDirected=True)
    print(g1.topological_sort())
