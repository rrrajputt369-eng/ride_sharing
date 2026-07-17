"""
Ride-Sharing Insertion Heuristic
=================================

Direct translation of the provided pseudocode:

  - ExecuteRideSharingSystem : greedily inserts each incoming ride request into
    the current route at whichever (pickup_slot, drop_slot) position increases
    total route distance the least, while respecting vehicle capacity and each
    passenger's maximum detour ("flex") tolerance.

  - ValidateDynamicConstraints : given a candidate route and the manifest of
    requests riding on it, checks capacity at every node and checks that every
    passenger's actual travel distance does not exceed their allowed distance
    (their direct distance + their flex budget).

Locations are represented as hashable values (e.g. strings, ints, or (x, y)
tuples). Distance is computed with a pluggable `distance_fn` (defaults to
Euclidean distance when locations are (x, y) tuples).
"""

import time
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Request:
    id: Any
    pickup: Any
    drop: Any
    dist: float   # the passenger's own direct pickup->drop distance
    flex: float   # extra detour distance the passenger is willing to tolerate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_distance(a: Any, b: Any) -> float:
    """Euclidean distance assuming locations are (x, y) tuples."""
    return math.dist(a, b)


def get_current_time_ms() -> float:
    return time.time() * 1000.0


def compute_route_distance(route: List[Any], distance_fn: Callable[[Any, Any], float]) -> float:
    total = 0.0
    for k in range(1, len(route)):
        total += distance_fn(route[k - 1], route[k])
    return total


# ---------------------------------------------------------------------------
# ValidateDynamicConstraints
# ---------------------------------------------------------------------------

def validate_dynamic_constraints(
    route: List[Any],
    manifest: Dict[Any, Request],
    capacity: int,
    distance_fn: Callable[[Any, Any], float] = _default_distance,
) -> Tuple[bool, float, List[int]]:
    """
    Returns (is_valid, total_distance_or_inf, passenger_counts_per_node).
    """
    current_passengers = 0
    passenger_counts: List[int] = []
    location_indices: Dict[Any, int] = {}

    # Pre-map the chronological index of every physical location in the route.
    # Duplicates: the LAST occurrence wins (matches pseudocode's "most recent index").
    for idx, node in enumerate(route):
        location_indices[node] = idx

    # Step 1 (OPTIMIZED): precompute each location's net passenger delta once,
    # in O(R), instead of re-scanning the whole manifest at every route node
    # (which was O(M*R)). A single O(M) pass over the route then applies the
    # precomputed delta at each stop, so this step is O(M+R) overall.
    delta: Dict[Any, int] = {}
    for req in manifest.values():
        delta[req.pickup] = delta.get(req.pickup, 0) + 1
        delta[req.drop] = delta.get(req.drop, 0) - 1

    for node in route:
        current_passengers += delta.get(node, 0)
        passenger_counts.append(current_passengers)

        if current_passengers > capacity:
            return False, float("inf"), []

    # Step 2: cumulative distances along the route.
    cumulative_dist = [0.0] * len(route)
    for k in range(1, len(route)):
        cumulative_dist[k] = cumulative_dist[k - 1] + distance_fn(route[k - 1], route[k])

    # Step 3: chronology + flexibility checks for every request in the manifest.
    for req in manifest.values():
        p_pos = location_indices.get(req.pickup, -1)
        d_pos = location_indices.get(req.drop, -1)

        if p_pos == -1 or d_pos == -1 or p_pos >= d_pos:
            return False, float("inf"), []

        actual_travel_dist = cumulative_dist[d_pos] - cumulative_dist[p_pos]
        max_allowed_dist = req.dist + req.flex

        # EPSILON guards against floating-point rounding noise (e.g. summing
        # cumulative distances can differ from a direct distance call by a
        # few ulps). Without it, exact-boundary cases (actual == max_allowed,
        # most commonly flex == 0) can be wrongly rejected -- or, depending on
        # which side the rounding lands on, wrongly accepted.
        EPSILON = 1e-6
        if actual_travel_dist > max_allowed_dist + EPSILON:
            return False, float("inf"), []

    total_distance = cumulative_dist[-1]
    return True, total_distance, passenger_counts


# ---------------------------------------------------------------------------
# ExecuteRideSharingSystem
# ---------------------------------------------------------------------------

def execute_ride_sharing_system(
    source: Any,
    destination: Any,
    capacity: int,
    max_passengers_hint: int,  # M in the pseudocode; kept for signature parity, unused directly
    requests: List[Request],
    distance_fn: Callable[[Any, Any], float] = _default_distance,
) -> Tuple[List[Any], float, float, List[int]]:
    """
    Greedily inserts each request into the route at the cheapest valid
    (pickup_slot, drop_slot) position.

    Returns (final_route, final_distance, elapsed_ms, passenger_counts).
    """
    start_time = get_current_time_ms()

    # Anchor the global boundaries. Source is index 0, Destination is index 1.
    route: List[Any] = [source, destination]
    current_visited_idx = 0
    accepted_manifest: Dict[Any, Request] = {}

    for new_request in requests:
        best_route = None
        min_cost_increase = float("inf")
        base_route_cost = compute_route_distance(route, distance_fn)

        # Internal bounds: slots open AFTER current vehicle position
        # but strictly BEFORE the final Destination node (len - 1).
        start_slot = current_visited_idx + 1
        end_slot = len(route) - 1

        test_manifest = dict(accepted_manifest)
        test_manifest[new_request.id] = new_request

        # O(M^2) slot exploration within the intermediate subpath.
        for i in range(start_slot, end_slot + 1):
            for j in range(i, end_slot + 1):

                # Construct candidate sequence strictly inside the global boundaries.
                # Inclusive pseudocode slice Route[a...b] == Python Route[a:b+1].
                candidate_route = (
                    route[0:i]
                    + [new_request.pickup]
                    + route[i:j]
                    + [new_request.drop]
                    + route[j:len(route)]
                )

                # O(M+R) constraint validation check (see validate_dynamic_constraints).
                is_valid, candidate_cost, _ = validate_dynamic_constraints(
                    candidate_route, test_manifest, capacity, distance_fn
                )

                if is_valid:
                    cost_increase = candidate_cost - base_route_cost
                    if cost_increase < min_cost_increase:
                        min_cost_increase = cost_increase
                        best_route = candidate_route

        # If a valid path was found, commit the route update.
        if best_route is not None:
            route = best_route
            accepted_manifest[new_request.id] = new_request
        else:
            print(f"Request {new_request.id} rejected.")

    final_distance = compute_route_distance(route, distance_fn)
    _, _, passenger_counts = validate_dynamic_constraints(
        route, accepted_manifest, capacity, distance_fn
    )
    end_time = get_current_time_ms()

    return route, final_distance, (end_time - start_time), passenger_counts


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Locations as (x, y) coordinates.
    SOURCE = (0, 0)
    DEST = (10, 0)

    requests = [
        Request(id="R1", pickup=(2, 0), drop=(5, 0), dist=3, flex=2),
        Request(id="R2", pickup=(1, 0), drop=(4, 0), dist=3, flex=1),
        Request(id="R3", pickup=(6, 0), drop=(9, 0), dist=3, flex=0),
        Request(id="R4", pickup=(8, 0), drop=(3, 0), dist=5, flex=1),
    ]

    route, final_distance, elapsed_ms, passenger_counts = execute_ride_sharing_system(
        SOURCE, DEST, capacity=3, max_passengers_hint=3, requests=requests
    )

    print("\nFinal route:", route)
    print("Final distance:", final_distance)
    print("Elapsed (ms):", elapsed_ms)
    print("Passenger counts per stop:", passenger_counts)
