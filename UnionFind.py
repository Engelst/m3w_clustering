class UnionFind:
    def __init__(self, n):
        self._id = list(range(n))
        self._sz = [1] * n
        self._count = n

    def find(self, p):
        while p != self._id[p]:
            self._id[p] = self._id[self._id[p]]  # Path compression
            p = self._id[p]
        return p

    def union(self, p, q):
        i = self.find(p)
        j = self.find(q)
        if i == j:
            return

        # Make smaller root point to larger one
        if self._sz[i] < self._sz[j]:
            self._id[i] = j
            self._sz[j] += self._sz[i]
        else:
            self._id[j] = i
            self._sz[i] += self._sz[j]
        self._count -= 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def count(self):
        return self._count