from collections import deque

def f(l):
    if len(l) == 0:
        return l
    else:
        pivot = l[0]
        left = [x for x in l[1:] if x <= l[0] ]
        right = [x for x in l[1:] if x > l[0] ]
        return f(left) + [pivot] + f(right)



class C:
    def __init__(self, a=1):
        self.a = a

    def calc_double_a(self):
        return self.a * 2

c = C()
print(c.calc_double_a())
print(f([3,3,5,1,2,35,34]))


edges = []
visited = []
def explore(v):
    visited.append(v)
    for (x,y) in edges:
        if x == v:
            if not (y in visited):
                return explore(y)


def bread_first_search(vertices, edges, s):
    visited = {}
    dist = {}
    for u in vertices:
        visited[u] = False
    dist[s] = 0
    front = deque()
    front.append(s)
    while len(front) > 0:
        current_v = front.popleft()
        visited[current_v] = True
        for (x, y) in edges:
            if x == current_v:
                if not visited[y]:
                    dist[y] = dist[current_v] + 1
                    front.append(y)


def relax(u, v):
    if dist[v] > dist[u] + w[(u, v)]:
        dist[v] = dist[u] + w[(u, v)]


def dijkstra(s):
    dist = {}
    prev = {}
    for u in V:
        dist[u] = None
        prev[u] = None
    front = deque()
    front.append(s)
    while len(front) > 0:
        u = extract_min(front)
        for (x, y) in edges:
            if u == x:
                if dist[y] > dist[x] + w(x,y):
                    dist[y] = dist[x] + w(x,y)
                    change_priority(y)



