# by Tim Nguyen (for CS163 UCI)
# NOTE: WILL ONLY WORK WITH PYTHON 3.0+ 

from collections import defaultdict
import math
import heapq # for priority queue 

class GraphException(Exception):
    def __init__(self, message=""):
        print("ERROR: " + message)

class Graph:
    ''' Represent a graph object. Undirected by default. Initialize empty graph '''
######################## SETUP AND UTILITIES ############################### 
    def __init__(self, graph=defaultdict(list), weightDict={}, vertices=set(),transitionMatrix=[],
                 isDirected = False):
        ''' Input edgeDict --> get vertices from there.... '''
        # initialize
        self.directed = isDirected  # directed or not
        self.isWeighted = False     # is weighted determined by weightDict
        self.graph = graph          # {'A':['B','C'], ... } 
        self.weights = weightDict   # {('A','B'): 24, ... }
        # setup:
        self.vertices = set(self.graph.keys())
        if not self.directed:       # if not directed then make undirected
            self.makeUndirected()
        if len(weightDict) > 0:
            self.isWeighted = True

    def makeUndirected(self):
        ''' make edges go both ways -- unweighted '''
        for v in self.graph:
            for w in self.graph[v]:
                if v not in self.graph[w]:
                    self.graph[w].append(v)
            
        
    def weight(self, u, v):
        ''' return the weight of an edge if exists in the provided dict
        else raises an error '''
        if (u,v) in self.weights:
            return self.weights[u,v]
        elif not self.directed and (v,u) in self.weights:
            return self.weights[v,u]
        else:
            raise GraphException("Weight does not exist for edge " + u + " " + v)

    def add_vertex(self, v):
        if v in self.graph:
            raise GraphException("add_vertex: Vertex " + v + "already in graph")
        self.vertices.add(v)
        self.graph[v] = {}
    
    def add_edge(self, source, dest):
        if source in self.graph and dest in self.graph[source]:
            raise GraphException("Edge " + source + "," + dest+ "already in graph.")
        if source not in self.graph:
            self.add_vertex(source)
            self.graph[source] = []
        if dest not in self.graph:
            self.add_vertex(dest)
            self.graph[dest] = []
        self.graph[source].append(dest)
        if not self.directed:
            self.graph[dest].append(source)

    def remove_edge(self, v, w):
        ''' remove an edge '''
        if v not in self.graph:
            # raise exception?
            return
        self.graph[v].remove(w) # raise an exception if no edge
        if not self.directed:
            self.graph[w].remove(v)
        
    def remove_vertex(self, v):
        if v not in self.graph:
            # raise exception?
            return
        self.vertices.remove(v)
        del self.graph[v]
        for p in self.weights:
            if v in p:
                del self.weights[p]
        
    def neighbors(self,v):
        ''' return a set of neighbors vertices of v for undirected only '''
        return set(self.graph[v])
    
    def degree(self, vertex):
        ''' return the degree (in + out deg) of that vertex '''
        result = 0
        for i in self.graph[vertex]: # for destination in graph
            result+=1
        if not self.directed:
            return result
        for v in self.graph:
            if vertex in self.graph[v]:
                result +=1
        return result 

    def in_degree(self, vertex):
        '''Remove the number of indegree of a vertex
            only for directed graph... return degree for undirected '''
        result = 0
        for v in self.graph:
            if vertex in self.graph[v]:
                result +=1
        return result 

    def out_degree(self, vertex):
        ''' Remove the number of outdegree edges of a vertex '''
        result = 0
        for i in self.graph[vertex]: # for destination in graph
            result+=1
        return result

    ##################### ALGORITHMS #############################
    ''' See NOTES regarding algorithms runnability on directed vs. undirected graphs '''
    def mst_prim(self, seed):
        ''' return a minimum spanning tree graph '''
        if self.directed:
            raise GraphException("cannot run PrimMST on directed graph")
        if seed not in self.graph:
            raise GraphException("Invalid seed vertex")
        
        g = Graph(isDirected=True) # initialize an undirected graph
        g.add_vertex(seed)
        avail_edges = {} # dict of available edges
        for edge in self.graph[seed]:
            pass
        #### TO DO ### !!!! 

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
                    D[v] = D[u] + self.weight(u,v) # update shortest dist
                    P[v] = u                       # update parent 
        return D, P 
        
    def dijkstra(self, start): # O(m log n) for heap because loop thru edges m times and logn each loop
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
        ### Initialize priority Q #### 
        Q = [(0, start)]
        heapq.heapify(Q) # Q is now a min heap of D
        final = set()           # final set of vertices to keep track
        l = len(D)              # length of D to compare final
        ### Dijkstra Magic ### 
        while len(final) != l:          # this will be executed n times
            dv, v = heapq.heappop(Q)    # dv is distance of v and v is smallest vertex
            final.add(v)                # add v to final set 
            for w in self.graph[v]:     # RELAX all edges v-w
                if D[v] + self.weight(v,w) < D[w]: 
                    D[w] = D[v] + self.weight(v,w)
                    P[w] = v
                    heapq.heappush(Q, (D[w], w)) # modify priority queue 
        return D, P 

        
    def relax(self, u, v , D, P):
        ''' relax given an edge (u,v) and update D and P accordingly
        D is the distance to v and P is the previous vertex to v '''
        if D[u] + self.weight(u,v) < D[v]:
            D[v] = D[u] + self.weight(u,v)
            P[v] = u
            
    def bellman_ford(self, start): ## IMPLEMENT VARIATIONS TO OBTAIN BETTER PERFORMANCE 
        ''' Compute shortest path from a given start vertex
        NO NEGATIVE CYCLES ''' 
        inf = math.inf
        D = {} # D[v] = distance from start to vertex v 
        P = {} # P[v] = predecessor of v on that path
        # initialize all the initial distances to be infinity 
        for vertex in self.graph:
            P[vertex] = None
            D[vertex] = inf
            if vertex == start:
                D[vertex] = 0
        for i in range(n-1): # relax all edges n-1 times
            for u in self.graph: # can improve by stop when no change after relax
                for v in self.graph[u]:
                    self.relax(u,v,D,P)
        return D, P

    def astar(self, start, goal, h):
        ''' compute shortest path from start to goal FASTER than Dijkstra
        h is a heuristic function that returns the estimated cost to a point from start ''' 
        # 
        pass

    def johnson(self):
        ''' All pairs shortest paths
        Initialize a vertex, s, that is reachable to all other vertices with edge 0
        Compute the shortest path from s to all other vertices
        Use that to turn all paths/edge weights positive
        Run Dijkstra on all to find all pairs shortest path '''
        pass

    def widest_path(self,start):
        ''' Return D and P where D[u] returns the widest path from start to u
        P[u] returns the previous node on the widest path '''
        inf = math.inf
        D = {} # D[v] = widest path from start to vertex v 
        P = {} # P[v] = predecessor of v on that path
        # initialize all the initial distances to be infinity 
        for vertex in self.graph:
            P[vertex] = None
            D[vertex] = -inf
            if vertex == start:
                D[vertex] = inf
        ### Initialize priority Q #### 
        Q = [(0, start)]
        final = set()           # final set of vertices to keep track
        l = len(D)              # length of D to compare final
        ### Dijkstra Magic ### 
        while len(final) != l:          # this will be executed n times
            dv, v = Q.pop(0)            # dv is distance of v and v is smallest vertex
            final.add(v)                # add v to final set 
            for w in self.graph[v]:     # RELAX all edges v-w from v
                takeWidth = min( D[v], self.weight(v,w) )  # the width of the path if were to take v-w
                if takeWidth > D[w]: 
                    D[w] = takeWidth
                    P[w] = v
                    Q.append( (D[w], w)) # modify priority queue 
        return D, P 


    def schulze_voting(self):
        #todo
        pass

    def euler_tour(self):
        #todo
        pass

    def dynamic_tsp(self,start):
        #todo
        pass

    def mst_tsp_2approx(self):
        #todo
        pass

    def tsp_christofides(self):
        #todo 1.5 approx
        pass

    def elimination_ordering(self):
        ''' return a list of elimination ordering of a graph '''
        pass
    
    def degeneracy_ordering(self):
        ''' return the degeneracy ordering of a graph as a list '''
        pass

    
    def bron_kerbosch_nopivot(self):
        ''' List all maximal cliques -> return a list of sets of vertices that form
            maximal cliques '''
        result = [] # list of sets of vertices that form maximal clique
        def bk(R, P, X):
            ''' R, P, X are sets of vertices: 
                R = recursive clique we are building
                P = potential vertices we might be able to add to R not yet in clique
                X = excluding vertices -> we already tried and can't add to R -> use for checking maximal'''
            print("Hello I am bk")      # for hw to track recursive calls
            if len(P) == 0 and len(X) == 0:
                result.append(R)
            for v in P.copy():
                #print("DEBUG: " + v + "'s turn.")
                nv = self.neighbors(v) # set of neighbors of V
                # recursive call
                bk(R.union({v}), P.intersection(nv), X.intersection(nv))
                # move v from P to X 
                P.remove(v)
                X.add(v)
        # start by running bk(empty, all vertices, empty)
        vertices = self.vertices.copy()
        bk(set(), vertices, set()) # start with empty, allvertices, empty
        return result
                

    #################################### OTHER UTILITIES ####################
    def draw_graph(self):
        ''' somehow draw a graph... maybe on a new window with canvas '''
        pass
    
    def __str__(self):
        ''' Improve this somehow... right now just print dict '''
        return str(self.graph)


class NetworkFlow(Graph):
    ''' Network flow class to solve network flow problems '''
    def __init__(self, graph={}, source='s', sink='t'):
        ''' Input is a dict with vertices and destination with tuple value
        indicating flow and capacities. Leave flow = 0 for default
        e.g.: a->b with flow 2, capacity 10 is: self.graph['a']['b'] = (2,10)'''
        self.graph = graph
        self.source = source
        self.sink = sink


    def getCap(self, u, v):
        ''' return capacity of edge u->v'''
        return self.graph[u][v][1]

    def getFlow(self, u, v):
        ''' return flow amount of edge u->v'''
        return self.graph[u][v][0]

    def isValidVertex(self, u):
        ''' Check:  (1) Flow <= capacity
                    (2) In flow == out flow'''
        outFlow = 0
        inFlow = 0
        for v in self.graph[u]:
            flow, cap = self.graph[u][v]
            if flow > cap or flow < 0:
                return False
            outFlow += flow
        for v in self.graph:
            if u in self.graph[v] and v != u:
                inFlow += self.graph[v][u][1]
        return outFlow == inFlow
            
    def isValidFlow(self):
        ''' return True if given config of graph is valid '''
        for v in self.graph:
            if v != self.source and v!= self.sink:
                if not isValidVertex(v):
                    return False
        return True

    def findMaxFlow(self):
        ''' return the maxflow of this graph '''
        pass

    def findMinCut(self):
        ''' return mincut ??? '''
        pass

    
def construct_path(start, end, P):
    ''' Take P and 2 vertices and give the intermediary vertices '''
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

def random_graph(numberVertices, edgeDensity, directed=False, weighted=False, weightsRange=(1,10)):
    ''' make a random graph with given number of vertices and number of edges
    as a fraction of a complete graph. vertices are named '1', '2', ...
    NOTE: edgeDensity must be between 0 and 1
    Return a graph object.'''
    if edgeDensity not in range(0,1):
        raise GraphException("ERROR: edgeDensity must be between 0 and 1")
    # TO DO 


def make_graph_from_weight_dict(weightDict, directed=True):
    ''' given a weight dict return a graph object with everything initialized
    no vertices without edges ''' 
    result = Graph(weightDict=weightDict, isDirected=directed)
    edges = list(weightDict.keys())
    for edge in edges: # for edge in weight dict
        result.add_edge(edge[0], edge[1])
    return result

if __name__=="__main__":
    ''' test algorithms here '''
    # some examples ...
    # complete K4 graphs with vertex a-d
    K4 = { 'a' : ['d','b','c'],
           'b' : ['a' , 'c', 'd'],
           'c' : ['a','b','d'],
           'd' : ['a','b','c']}
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

    weightedGraph2 = {'a':['b', 'c'],
                      'b':['c','d','e'],
                      'c':['b','d','e'],
                      'd':[],
                      'e':['d']}
    weightedGraph2weights = {('a','b'):4, ('a','c'):2, ('b','c'):3, ('b','d'):2,
                             ('b','e'):3, ('c','b'):1, ('c','e'):5, ('c','d'):4,
                             ('e','d'):1}

    flow1 = { ('s', 'a'):6, ('s','b'):2, ('a','c'):3, ('a','d'):5, ('b','c'):7, ('b','d'):4,
              ('c','t'):8, ('d','t'):1}

    k4 = Graph(K4)
    fg1 = make_graph_from_weight_dict(flow1)
    g1 = Graph(graph1)
    wg1 = Graph(weightedGraph1, weightedGraph1weights, isDirected=True)
    wg2 = Graph(weightedGraph2, weightedGraph2weights, isDirected=True)
    #d, p  = wg1.dag_shortest_path('s')
    #print(wg1.construct_path('s','end',p))
    #print(g1.topological_sort())
