import copy
import torch
import numpy as np
import heapq

from anytree import Node, RenderTree
from collections import defaultdict

model = torch.load("./compressed_base_model_weight_cifar100.pt")["model"]

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


def find_weight_relations(weight):	# 임의로 edge 설정
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
prim
"""

def init_graph(graph):	# 각 정점 별로 연결된 정점 리스트 생성
  edges = defaultdict(list)
  for weight, node_u, node_v in graph["edges"]:
    edges[node_u].append((weight, node_u, node_v))
  return edges

def prim(graph):
  mst = list()
  visited_vertex = set()
  start_node = '1'	# 임의로 시작 노드 지정
  
  edges = init_graph(graph)
  # print("edges", edges)
  
  visited_vertex.add(start_node)	# 시작 노드 방문
  candidate_edges = edges[start_node]
  heapq.heapify(candidate_edges)	# 후보 엣지들을 최소 힙에 추가
  # print("candidate", candidate_edges)
  
  while candidate_edges:
    cur_weight, cur_u, cur_v = heapq.heappop(candidate_edges)	# 가중치가 제일 작은 엣지 선택
    if cur_v not in visited_vertex:	# 다음 노드가 방문하지 않은 노드이면
      visited_vertex.add(cur_v)	# 방문 처리 후 mst에 넣음
      mst.append((cur_weight, cur_u, cur_v))
      
      for edge in edges[cur_v]:
        if edge[2] not in visited_vertex:
          heapq.heappush(candidate_edges, edge)
          
          
  print("minimum spanning tree\n", mst)
  return mst


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
            connections = list(map(lambda x: [x[2], str(x[0]), str(x[1])], w))
            graph = {"vertices": list(map(str, range(1, 10))), "edges": connections}
            
            res = prim(graph)

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

    with open("layer_2_weight_trees_prim", "w") as f:
        f.write(find_result)
