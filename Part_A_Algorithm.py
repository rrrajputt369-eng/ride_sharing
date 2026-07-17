import math
import random

class UpgradedAdaptiveHybridSA:
    def __init__(self, locations, distance_matrix, max_budget, threshold_factor=0.9, k_decay=0.05):
        """
        :param locations: List of dicts, e.g., [{'id': 0, 'score': 10, 'category': 'A'}, ...]
                          Location at index 0 is always the fixed START depot.
        :param distance_matrix: 2D matrix where distance_matrix[i][j] is the distance from i to j.
        :param max_budget: Total allowed distance budget.
        :param threshold_factor: Category repetition penalty multiplier (e.g., 0.9).
        :param k_decay: Exponential decay constant for cumulative distance.
        """
        self.locations = locations
        self.distance_matrix = distance_matrix
        self.max_budget = max_budget
        self.threshold_factor = threshold_factor
        self.k_decay = k_decay
        self.N = len(locations)

    def calculate_route_score_and_distance(self, route):
        """
        Evaluates a sequence under true non-linear exponential decay 
        and cumulative category penalties.
        """
        if not route or route[0] != 0:
            return -float('inf'), 0

        total_distance = 0
        total_score = 0
        category_counts = {}

        # The starting node score is collected immediately
        start_cat = self.locations[0]['category']
        category_counts[start_cat] = 1
        total_score += self.locations[0]['score']

        for i in range(1, len(route)):
            u, v = route[i-1], route[i]
            dist = self.distance_matrix[u][v]
            total_distance += dist
            
            if total_distance > self.max_budget:
                return -float('inf'), total_distance  # Hard budget breach

            loc = self.locations[v]
            cat = loc['category']
            
            # Category repetition penalty management
            count = category_counts.get(cat, 0)
            penalty = self.threshold_factor ** count
            category_counts[cat] = count + 1

            # Cumulative distance exponential decay multiplier
            decay_multiplier = math.exp(-self.k_decay * total_distance)
            total_score += loc['score'] * penalty * decay_multiplier

        return total_score, total_distance

    def run_dynamic_beam_search(self):
        """
        Stage 1: Dynamic Beam Search Initialization
        Scales beam width 'k' aggressively to preserve high-value skeleton anchors.
        """
        # Your custom optimized dynamic beam width formula
        beam_width = max(3, (self.N // 5) + 2)
        
        start_cat = self.locations[0]['category']
        # Tracks paths as: (path_sequence, cumulative_distance, category_counts_dict)
        beams = [([0], 0, {start_cat: 1})]
        best_overall_route = [0]
        best_overall_score = self.locations[0]['score']

        for step in range(1, self.N):
            candidates = []
            for route, total_dist, cat_counts in beams:
                last_node = route[-1]
                
                for next_node in range(1, self.N):
                    if next_node in route:
                        continue
                        
                    step_dist = self.distance_matrix[last_node][next_node]
                    new_dist = total_dist + step_dist
                    
                    if new_dist > self.max_budget:
                        continue

                    # Calculate local S/d1 value-density
                    loc = self.locations[next_node]
                    cat = loc['category']
                    count = cat_counts.get(cat, 0)
                    penalty = self.threshold_factor ** count
                    
                    approx_decay = math.exp(-self.k_decay * new_dist)
                    effective_score = loc['score'] * penalty * approx_decay
                    
                    # Density Score metric: S / d1
                    density_metric = effective_score / max(0.1, step_dist)
                    
                    new_route = route + [next_node]
                    new_counts = cat_counts.copy()
                    new_counts[cat] = count + 1
                    
                    candidates.append((density_metric, new_route, new_dist, new_counts))
            
            if not candidates:
                break
                
            # Sort candidates by value-density metric descending and keep top 'beam_width' paths
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [(cand[1], cand[2], cand[3]) for cand in candidates[:beam_width]]
            
            # Check if any path encountered sets a new record
            for cand in candidates:
                r_score, _ = self.calculate_route_score_and_distance(cand[1])
                if r_score > best_overall_score:
                    best_overall_score = r_score
                    best_overall_route = cand[1]

        return best_overall_route

    def mutate_route(self, route):
        """
        Generates stochastic neighborhood variations including node swaps, 
        sequence inversions (2-opt style), and unvisited node toggles.
        """
        new_route = list(route)
        if len(new_route) <= 1:
            return new_route

        mutation_type = random.choice(['swap', 'invert', 'toggle'])
        
        if mutation_type == 'swap' and len(new_route) > 2:
            idx1, idx2 = random.sample(range(1, len(new_route)), 2)
            new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
            
        elif mutation_type == 'invert' and len(new_route) > 3:
            idx1, idx2 = sorted(random.sample(range(1, len(new_route)), 2))
            new_route[idx1:idx2+1] = reversed(new_route[idx1:idx2+1])
            
        elif mutation_type == 'toggle':
            visited_set = set(new_route)
            unvisited = [i for i in range(1, self.N) if i not in visited_set]
            
            if unvisited and (len(new_route) == 1 or random.random() < 0.5):
                insert_node = random.choice(unvisited)
                insert_pos = random.randint(1, len(new_route))
                new_route.insert(insert_pos, insert_node)
            elif len(new_route) > 1:
                delete_idx = random.randint(1, len(new_route) - 1)
                new_route.pop(delete_idx)
                
        return new_route

    def optimize(self):
        """
        Stage 2: Adaptive Local Simulated Annealing Polish
        Warm-starts using the dynamic beam path, utilizing an updated cooling schedule 
        and higher iteration cap to eliminate micro-lags.
        """
        # Warm-start initialization
        current_route = self.run_dynamic_beam_search()
        current_score, _ = self.calculate_route_score_and_distance(current_route)
        
        best_route = list(current_route)
        best_score = current_score
        
        T = 1.0
        alpha_fast = 0.98        # Slowed down from 0.95 to maximize exploration
        alpha_cushion = 0.993    # Precision low-temperature fine-tuning
        max_iterations = 5000    # Expanded from 2500 to guarantee convergence
        
        for iteration in range(max_iterations):
            # Dynamic adaptive cooling transitions
            if T > 0.1:
                T *= alpha_fast
            else:
                T *= alpha_cushion
                
            if T < 1e-6:
                break
                
            candidate_route = self.mutate_route(current_route)
            cand_score, _ = self.calculate_route_score_and_distance(candidate_route)
            
            # Hard boundary validation check
            if cand_score == -float('inf'):
                continue
                
            delta = cand_score - current_score
            
            # Metropolis-Hastings Acceptance Criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                current_route = candidate_route
                current_score = cand_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_route = list(current_route)
                    
        return best_route, best_score

# ==========================================
# Execution Demo
# ==========================================
if __name__ == "__main__":
    # Test layout mapping to check functionality
    test_locations = [
        {'id': 0, 'score': 0, 'category': 'A'},
        {'id': 1, 'score': 40, 'category': 'A'},
        {'id': 2, 'score': 30, 'category': 'B'},
        {'id': 3, 'score': 50, 'category': 'A'},
        {'id': 4, 'score': 25, 'category': 'B'}
    ]
    
    test_matrix = [
        [0.0,  5.0,  6.0, 12.0, 13.0],
        [5.0,  0.0,  2.0,  9.0, 10.0],
        [6.0,  2.0,  0.0,  8.0,  7.0],
        [12.0,  9.0,  8.0,  0.0,  4.0],
        [13.0, 10.0,  7.0,  4.0,  0.0]
    ]
    
    solver = UpgradedAdaptiveHybridSA(
        locations=test_locations,
        distance_matrix=test_matrix,
        max_budget=25.0,
        threshold_factor=0.5,
        k_decay=0.1
    )
    
    final_path, final_score = solver.optimize()
    print("=== FINAL UPGRADED ALGORITHM RUN ===")
    print(f"Optimal Path Sequence : {final_path}")
    print(f"Maximized Score Result : {final_score:.3f}")
