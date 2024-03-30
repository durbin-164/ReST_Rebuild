from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class MemoryBuffer:
    def __init__(self, buffer_size=30, match_threshold: float = 0.5):
        self.memory = defaultdict(lambda: defaultdict(lambda: deque(maxlen=buffer_size)))
        self.buffer_size = buffer_size
        self.match_threshold = match_threshold

    def update(self, embeds: List[List[Tuple[int, int, Tensor]]], nodes: List[List[int]], label: int):
        for n in nodes:
            if n < len(embeds):
                embed = embeds[n]
                for fid, cid, em in embed:
                    self.memory[label][cid].appendleft(em)

    def get_existing_node_ids(self, embeds: List[List[Tuple[int, int, Tensor]]], unassigned_nodes: List[List[int]],
                              assigned_ids: List[int]):
        unassigned_memory_label = set(self.memory.keys()) - set(assigned_ids)
        # unassigned_memory_label = set(self.memory.keys())

        if not unassigned_memory_label:
            return {}, unassigned_nodes

        results = []
        for nodes in unassigned_nodes:
            nodes = tuple(nodes)
            node_embeds = self.get_nodes_embed(embeds, nodes)
            results += self.get_similarity(node_embeds, unassigned_memory_label, nodes)

        unique_nodes_and_score = self.find_unique_nodes_with_max_score(results)
        new_assigned_nodes_label, new_unassigned_nodes = self.get_assign_and_unassigned_nodes(unique_nodes_and_score)
        return new_assigned_nodes_label, new_unassigned_nodes

    def get_nodes_embed(self, embeds: List[List[Tuple[int, int, Tensor]]], nodes: Tuple):
        node_embeds = []
        for n in nodes:
            if n < len(embeds):
                for f, c, em in embeds[n]:
                    node_embeds.append(em)

        return node_embeds

    def get_similarity(self, node_embeds: List[Tensor], unassigned_memory_label: Set[int], nodes: Tuple):
        results = []
        for label in unassigned_memory_label:
            max_score = 0
            m_embeds = self.get_memory_values_for_label(label)
            for n_em in node_embeds:
                for m_em in m_embeds:
                    score = F.cosine_similarity(n_em.unsqueeze(0), m_em.unsqueeze(0)).item()
                    max_score = max(score, max_score)

            results.append((nodes, label, max_score))

        return results

    def find_unique_nodes_with_max_score(self, data: List[Tuple]):
        max_score_nodes = {}
        sorted_data = sorted(data, key=lambda x: x[2])
        unique_label = set()

        # Iterate over the data
        for nodes, label, score in sorted_data:
            if label not in unique_label and nodes not in max_score_nodes:
                max_score_nodes[nodes] = (label, score)
                unique_label.add(label)

        return max_score_nodes

    def get_assign_and_unassigned_nodes(self, unique_nodes_score: Dict[int, Tuple[Set[int], float]]):
        new_assigned_nodes_label = {}
        new_unassigned_nodes = []
        for nodes, (label, score) in unique_nodes_score.items():
            if score >= self.match_threshold:
                new_assigned_nodes_label[nodes] = label
            else:
                new_unassigned_nodes.append(nodes)

        return new_assigned_nodes_label, new_unassigned_nodes

    def get_memory_values_for_label(self, label: int):
        values_list = []
        for deque_obj in self.memory[label].values():
            values_list.extend(deque_obj)
        return values_list
