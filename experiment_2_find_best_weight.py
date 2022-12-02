import copy
import torch
import numpy as np

from anytree import Node, RenderTree

model = torch.load("./compressed_base_model_weight.pt")["model"]

"""
print(model) 하면 아래와 같이 나옴

Sequential(
  (feature_extractor): Sequential(
    (layer1): Sequential(
      (conv): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU(inplace=True)
      (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (layer2): Sequential(
      (conv): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU(inplace=True)
      (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (layer3): Sequential(
      (conv): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU(inplace=True)
      (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    )
  (classifier): Sequential(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (in_linear): Linear(in_features=144, out_features=100, bias=True)
    (relu): ReLU(inplace=True)
    (out_linear): Linear(in_features=100, out_features=10, bias=True)
    (output): Softmax(dim=0)
  )
)
"""

weights = model.feature_extractor.layer2.conv.weight
bias = model.feature_extractor.layer2.conv.bias

weights = weights.detach().cpu().numpy()
bias = bias.detach().cpu().numpy()

weights = copy.deepcopy(weights)
bias = copy.deepcopy(bias)


def find_weight_relations(weight):
    left_to_right = (np.pad(weight[:, 1:], ((0, 0), (0, 1))) - weight[:, :])[
        :, :2
    ]  # left -> right
    top_to_bottom = ((np.pad(weight[1:, :], ((0, 1), (0, 0)))) - weight[:, :])[
        :2
    ]  # top -> bottom

    return [
        [i * 3 + 1 + j, j + 2 + i * 3, left_to_right[i, j]]
        for i in range(3)
        for j in range(2)
    ] + [
        [i * 3 + 1 + j, (i + 1) * 3 + j + 1, top_to_bottom[i, j]]
        for i in range(2)
        for j in range(3)
    ]


"""
kruscal
"""

parent = dict()
rank = dict()


# vertice 초기화
def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0


# 해당 vertice의 최상위 정점을 찾는다
def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]


# 두 정점을 연결한다
def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]:
                rank[root2] += 1


def kruskal(graph):
    minimum_spanning_tree = []

    # 초기화
    for vertice in graph["vertices"]:
        make_set(vertice)

    # 간선 weight 기반 sorting
    edges = graph["edges"]
    edges.sort()

    # 간선 연결 (사이클 없게)
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.append(edge)

    return minimum_spanning_tree


if __name__ == "__main__":
    find_result = ""

    """
    왜 8과 4이냐?
    weight.shape == [8, 4, 3, 3]
    """
    for in_ in range(8):
        for out_ in range(4):
            print(f"=================== <weight[{in_}, {out_}]> ===================")
            find_result += (
                f"=================== <weight[{in_}, {out_}]> ===================\n"
            )
            feed_weight = weights[in_, out_]

            w = find_weight_relations(feed_weight)
            connections = sorted(w, reverse=False, key=lambda x: x[-1])
            connections = list(map(lambda x: [x[2], str(x[0]), str(x[1])], connections))

            graph = {"vertices": list(map(str, range(1, 10))), "edges": connections}

            res = kruskal(graph)

            nodes = [
                Node(i, weight=feed_weight[(i - 1) // 3, (i - 1) % 3])
                for i in range(1, 10)
            ]

            for r in res:
                start, end = map(int, r[1:])
                nodes[end - 1].parent = nodes[start - 1]

            # mst가 space에 몇개 있는지
            roots = {x.root.name for x in nodes}

            for root in roots:
                for pre, fill, node in RenderTree(nodes[root - 1]):
                    find_result += f"{pre}{node.name}\n"
                    print("%s%s" % (pre, node.name))

    with open("layer_2_weight_trees_kruskal", "w") as f:
        f.write(find_result)
