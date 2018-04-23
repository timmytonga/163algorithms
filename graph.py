from collection import defaultdict

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
        self.vertices = vertices    # {'A', 'B', 'C', ... }
        self.graph = graph          # {'A':{'B','C'}, ... } 
        self.weights = weightDict   # {('A','B'): 24, ... }

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
        self.graph[source].add(dest)
        if not self.directed:
            self.graph[dest].add(source)

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
        if start not in self.vertices:
            raise GraphException("dfs invalid start vertex")
        visited = set()
        def visit(w):
            visited.add(w)
            for dest in self.graph[w]:
                if dest not in visited:
                    visit(dest)
        visit(start)
        return visited
                    
    def bfs(self, start):
        ''' Return all reachable vertex from start using shortest path
            using bfs traversal '''
        reached = { start }
        q = [start]
        while len(q) != 0:
            v = q.pop()
            for dest in self.graph[v]:
                if dest not in reached:
                    reached.add(dest)
        return reached     

    def topological_sort(self):
        ''' Directed graphs only '''
        pass

    def dijkstra(self, start):
        ''' Compute shortest path from a given start vertex
        NOTE: NO NEGATIVE EDGE WEIGHTS -- use bellman_ford instead ''' 
        pass

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
