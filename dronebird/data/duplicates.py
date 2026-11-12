"""Finding near-duplicate frames and grouping them.

Author: Mariah Habak

Frames from the same video look almost identical, and the dataset does
not always say which video a frame comes from. We compare every pair of
images through a small colour thumbnail (16 x 16 x 3, values in [0, 1]):

    difference(i, j) = root-mean-square of (thumb_i - thumb_j)

computed for all pairs at once with torch.cdist, in chunks. Pairs below
a threshold, and images that share a source id from their file name,
are joined into groups with union-find. A group never crosses two
partitions.

(We first tried a difference hash, dHash, but on smooth skies its bits
depend on noise, so two near-identical frames got very different
hashes.)
"""

import numpy as np
import torch


class UnionFind:
    def __init__(self, n: int):
        self.__parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.__parent[root] != root:
            root = self.__parent[root]
        while self.__parent[i] != root:  # path compression
            self.__parent[i], i = root, self.__parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.__parent[max(ra, rb)] = min(ra, rb)


class NearDuplicateGrouper:
    """Groups images whose thumbnails are closer than a threshold."""

    def __init__(self, max_rms_difference: float, chunk: int = 1024):
        self.__threshold = max_rms_difference
        self.__chunk = chunk
        self.__pairs = 0

    @property
    def confirmed_pairs(self) -> int:
        return self.__pairs

    def close_pairs(self, thumbnails: torch.Tensor):
        """Index pairs (i < j) with RMS thumbnail difference <= threshold."""
        flat = thumbnails.flatten(start_dim=1).float()
        dims = flat.shape[1]
        limit = self.__threshold * (dims ** 0.5)  # RMS -> Euclidean distance
        for start in range(0, len(flat), self.__chunk):
            block = flat[start: start + self.__chunk]
            distance = torch.cdist(block, flat)
            rows, cols = torch.nonzero(distance <= limit, as_tuple=True)
            rows = rows + start
            keep = cols > rows
            yield from zip(rows[keep].tolist(), cols[keep].tolist())

    def group(self, thumbnails: torch.Tensor, sources: list) -> list:
        """Group id per image (0..G-1)."""
        n = len(thumbnails)
        uf = UnionFind(n)
        self.__pairs = 0
        for i, j in self.close_pairs(thumbnails):
            uf.union(i, j)
            self.__pairs += 1
        first_by_source = {}
        for i, source in enumerate(sources):
            if source:
                if source in first_by_source:
                    uf.union(first_by_source[source], i)
                else:
                    first_by_source[source] = i
        roots = {}
        return [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]


def group_report(groups: list) -> dict:
    sizes = np.bincount(np.asarray(groups))
    shared = sizes[sizes > 1]
    return {"images": int(len(groups)), "groups": int(len(sizes)),
            "groups_with_duplicates": int(len(shared)),
            "images_in_duplicate_groups": int(shared.sum()),
            "largest_group": int(sizes.max()) if len(sizes) else 0}
