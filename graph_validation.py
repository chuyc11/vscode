"""Graph validation — pure-Python topological checks for ClaimGraphDraft.

Enhanced with:
- Cycle detection (topological sort)
- Max depth enforcement
- Disconnected component detection
- Fan-in / fan-out anomaly detection
- Topological ordering export for downstream batch analysis
"""

from typing import List, Set, Dict
from collections import defaultdict, deque
from schema import ClaimGraphDraft


def validate_claim_dag(draft: ClaimGraphDraft) -> None:
    """Validate ClaimGraphDraft is a well-formed DAG.

    Raises ValueError on any structural violation.
    """
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

    # Build adjacency: parent -> child
    graph: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {d.temp_id: 0 for d in drafts}
    for d in drafts:
        for parent in d.parent_temp_ids:
            graph[parent].append(d.temp_id)
            indegree[d.temp_id] += 1

    # Cycle detection via Kahn's algorithm
    queue = deque([tid for tid in indegree if indegree[tid] == 0])
    topo_order: List[str] = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(topo_order) != len(drafts):
        raise ValueError("Cycle detected in claim graph.")

    # Max depth check (BFS from roots)
    if drafts:
        depths: Dict[str, int] = {d.temp_id: 0 for d in drafts}
        roots = [d.temp_id for d in drafts if not d.parent_temp_ids]
        queue = deque([(r, 0) for r in roots])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for child in graph[node]:
                depths[child] = max(depths[child], depth + 1)
                queue.append((child, depth + 1))
        if max_depth > 10:
            raise ValueError(f"Max depth exceeded: {max_depth}")


def get_topological_order(draft: ClaimGraphDraft) -> List[str]:
    """Return topological ordering of temp_ids (roots first).

    Useful for downstream batch analysis — process nodes in dependency order.
    """
    drafts = draft.drafts
    graph: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {d.temp_id: 0 for d in drafts}
    for d in drafts:
        for parent in d.parent_temp_ids:
            graph[parent].append(d.temp_id)
            indegree[d.temp_id] += 1

    queue = deque([tid for tid in indegree if indegree[tid] == 0])
    topo_order: List[str] = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return topo_order


def detect_disconnected_components(draft: ClaimGraphDraft) -> List[List[str]]:
    """Find disconnected components in the claim graph.

    Returns a list of components, each being a list of temp_ids.
    A single component means the graph is fully connected.
    """
    drafts = draft.drafts
    if not drafts:
        return []

    # Build undirected adjacency
    adj: Dict[str, Set[str]] = defaultdict(set)
    for d in drafts:
        for parent in d.parent_temp_ids:
            adj[d.temp_id].add(parent)
            adj[parent].add(d.temp_id)

    visited: Set[str] = set()
    components: List[List[str]] = []

    for d in drafts:
        if d.temp_id in visited:
            continue
        # BFS to find component
        component = []
        queue = deque([d.temp_id])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    return components


def detect_fan_anomalies(
    draft: ClaimGraphDraft,
    max_fan_in: int = 5,
    max_fan_out: int = 8,
) -> List[str]:
    """Detect fan-in / fan-out anomalies in the claim graph.

    Returns a list of warning strings for nodes with excessive connections.
    """
    drafts = draft.drafts
    children_count: Dict[str, int] = defaultdict(int)
    parents_count: Dict[str, int] = defaultdict(int)

    for d in drafts:
        parents_count[d.temp_id] = len(d.parent_temp_ids)
        for parent in d.parent_temp_ids:
            children_count[parent] += 1

    warnings = []
    for d in drafts:
        if parents_count[d.temp_id] > max_fan_in:
            warnings.append(
                f"Node '{d.temp_id}' has {parents_count[d.temp_id]} parents "
                f"(max_fan_in={max_fan_in}) — possible over-aggregation."
            )
        if children_count[d.temp_id] > max_fan_out:
            warnings.append(
                f"Node '{d.temp_id}' has {children_count[d.temp_id]} children "
                f"(max_fan_out={max_fan_out}) — possible over-fan-out."
            )

    return warnings
