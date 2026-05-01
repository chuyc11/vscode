from typing import List, Set, Dict
from collections import defaultdict, deque
from schema import ClaimGraphDraft

def validate_claim_dag(draft: ClaimGraphDraft) -> None:
    drafts = draft.drafts
    temp_ids = [d.temp_id for d in drafts]
    
    # Check duplicate temp_id
    if len(temp_ids) != len(set(temp_ids)):
        raise ValueError("Duplicate temp_id found in drafts.")
    
    temp_id_set = set(temp_ids)
    
    # Check missing parent and self-reference
    for d in drafts:
        for parent in d.parent_temp_ids:
            if parent not in temp_id_set:
                raise ValueError(f"Missing parent temp_id: {parent}")
            if parent == d.temp_id:
                raise ValueError(f"Self-reference in temp_id: {d.temp_id}")
    
    # Check cycles
    graph = defaultdict(list)
    indegree = {d.temp_id: 0 for d in drafts}
    for d in drafts:
        for parent in d.parent_temp_ids:
            graph[parent].append(d.temp_id)
            indegree[d.temp_id] += 1
    
    queue = deque([tid for tid in indegree if indegree[tid] == 0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    
    if visited != len(drafts):
        raise ValueError("Cycle detected in claim graph.")
    
    # Check max depth (simple BFS depth)
    if drafts:
        depths = {d.temp_id: 0 for d in drafts}
        queue = deque([(d.temp_id, 0) for d in drafts if not d.parent_temp_ids])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for child in graph[node]:
                depths[child] = max(depths[child], depth + 1)
                queue.append((child, depth + 1))
        if max_depth > 10:  # arbitrary max depth
            raise ValueError(f"Max depth exceeded: {max_depth}")