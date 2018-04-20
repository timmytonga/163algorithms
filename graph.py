from collection import defaultdict

class GraphException(Exception):
    def __init__(self, message=""):
        print("ERROR: " + message)

class Graph:
    ''' Represent a graph object. Undirected by default. Initialize empty graph '''
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
        ''' Return all reachable vertex from start using shortest path  '''
        reached = { start }
        q = [start]
        while len(q) != 0:
            v = q.pop()
            for dest in self.graph[v]:
                if dest not in reached:
                    reached.add(dest)
                    

    def all_pairs_shortest(self):
        pass

    def topological_sort(self):
        pass

    def shortest_path_dijkstra(self, start):
        pass

    def bellman_ford(self, start):
        pass

    def mst_prim(self, seed):
        pass

    def mst_boruvka(self):
        pass

    def mst_kruskal(self):
        pass
    
