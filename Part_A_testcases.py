#Just run this file to test the testcases

import math
import random

class IITDharwadSightseeingSolver:
    def __init__(self, locations, distance_matrix, max_budget, category_threshold):
        """
        Implementation conforming strictly to IIT Dharwad Part A guidelines.
        """
        self.locations = locations
        self.distance_matrix = distance_matrix
        self.max_budget = max_budget
        self.n_threshold = category_threshold
        self.k_decay = 0.1  # Set strictly to 0.1 per specification
        self.N = len(locations)

    def calculate_route_score_and_distance(self, route):
        """
        Calculates exact cumulative exponential decay scores and distances.
        Ensures route starts at 0 and ends at N-1.
        """
        if not route or route[0] != 0 or route[-1] != self.N - 1:
            return -float('inf'), 0

        total_distance = 0
        total_score = 0
        category_counts = {}

        for i in range(1, len(route)):
            u, v = route[i-1], route[i]
            dist = self.distance_matrix[u][v]
            
            # Cumulative distance BEFORE reaching location 'v' is total_distance
            loc = self.locations[v]
            cat = loc['category']
            
            # Start and End nodes do not yield satisfaction points
            if cat not in ['START', 'END']:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Apply 10% penalty if the category count EXCEEDS threshold n
                penalty = 0.9 if category_counts[cat] > self.n_threshold else 1.0
                
                # S_eff = S * e^(-k * d)
                decay_multiplier = math.exp(-self.k_decay * total_distance)
                total_score += loc['score'] * penalty * decay_multiplier
            
            total_distance += dist
            if total_distance > self.max_budget:
                return -float('inf'), total_distance  # Hard budget constraint breach

        return total_score, total_distance

    def run_dynamic_beam_search(self):
        """
        Stage 1: Dynamic Beam Search Initialization
        Optimized to search paths ending at N-1.
        """
        beam_width = max(3, (self.N // 5) + 2)
        
        # Structure: (current_route, total_distance, category_counts)
        beams = [([0], 0, {})]
        best_route = [0, self.N - 1]
        best_score, _ = self.calculate_route_score_and_distance(best_route)

        for step in range(1, self.N):
            candidates = []
            for route, total_dist, cat_counts in beams:
                last_node = route[-1]
                
                # Check option to close path directly to destination
                dest_dist = self.distance_matrix[last_node][self.N - 1]
                if total_dist + dest_dist <= self.max_budget:
                    test_route = route + [self.N - 1]
                    r_score, _ = self.calculate_route_score_and_distance(test_route)
                    if r_score > best_score:
                        best_score = r_score
                        best_route = test_route

                for next_node in range(1, self.N - 1):
                    if next_node in route:
                        continue
                        
                    step_dist = self.distance_matrix[last_node][next_node]
                    new_dist = total_dist + step_dist
                    
                    if new_dist + self.distance_matrix[next_node][self.N - 1] > self.max_budget:
                        continue

                    loc = self.locations[next_node]
                    cat = loc['category']
                    count = cat_counts.get(cat, 0) + 1
                    penalty = 0.9 if count > self.n_threshold else 1.0
                    
                    approx_decay = math.exp(-self.k_decay * total_dist)
                    effective_score = loc['score'] * penalty * approx_decay
                    density_metric = effective_score / max(0.1, step_dist)
                    
                    new_route = route + [next_node]
                    new_counts = cat_counts.copy()
                    new_counts[cat] = count
                    
                    candidates.append((density_metric, new_route, new_dist, new_counts))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [(cand[1], cand[2], cand[3]) for cand in candidates[:beam_width]]

        return best_route

    def mutate_route(self, route):
        """
        Stage 2 Helper: Generates neighborhood mutations preserving fixed endpoints.
        """
        if len(route) <= 2:
            # If path is only [Start, End], mutation can only insert an unvisited node
            unvisited = [i for i in range(1, self.N - 1)]
            if unvisited:
                new_route = [0, random.choice(unvisited), self.N - 1]
                return new_route
            return list(route)

        new_route = list(route)
        intermediates = new_route[1:-1]
        mutation_type = random.choice(['swap', 'invert', 'toggle'])
        
        if mutation_type == 'swap' and len(intermediates) >= 2:
            idx1, idx2 = random.sample(range(0, len(intermediates)), 2)
            intermediates[idx1], intermediates[idx2] = intermediates[idx2], intermediates[idx1]
            
        elif mutation_type == 'invert' and len(intermediates) >= 2:
            idx1, idx2 = sorted(random.sample(range(0, len(intermediates)), 2))
            intermediates[idx1:idx2+1] = reversed(intermediates[idx1:idx2+1])
            
        elif mutation_type == 'toggle':
            visited_set = set(intermediates)
            unvisited = [i for i in range(1, self.N - 1) if i not in visited_set]
            
            if unvisited and (len(intermediates) == 0 or random.random() < 0.5):
                insert_node = random.choice(unvisited)
                insert_pos = random.randint(0, len(intermediates))
                intermediates.insert(insert_pos, insert_node)
            elif len(intermediates) > 0:
                intermediates.pop(random.randint(0, len(intermediates) - 1))
                
        return [0] + intermediates + [self.N - 1]

    def optimize(self):
        """
        Polishes the warm-start path using the tuned SA routine.
        """
        current_route = self.run_dynamic_beam_search()
        current_score, _ = self.calculate_route_score_and_distance(current_route)
        
        best_route = list(current_route)
        best_score = current_score
        
        T = 1.0
        alpha_fast = 0.98        
        alpha_cushion = 0.993    
        max_iterations = 5000    
        
        for iteration in range(max_iterations):
            T *= alpha_fast if T > 0.1 else alpha_cushion
            if T < 1e-6:
                break
                
            candidate_route = self.mutate_route(current_route)
            cand_score, _ = self.calculate_route_score_and_distance(candidate_route)
            
            if cand_score == -float('inf'):
                continue
                
            delta = cand_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_route = candidate_route
                current_score = cand_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_route = list(current_route)
                    
        return best_route, best_score


# =====================================================================
# AUTOMATED DATA INJECTION LOADER FOR PUBLIC TEST CASES 1 - 6
# =====================================================================
if __name__ == "__main__":
    random.seed(42)

    public_test_suite = {
        "Public Test Case 1 (Easy)": {
            "locations": [
                {'id': 0, 'name': 'Start', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Museum', 'score': 8, 'category': 'Historical'},
                {'id': 2, 'name': 'Park', 'score': 6, 'category': 'Nature'},
                {'id': 3, 'name': 'End', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 3, 5, 10],
                [3, 0, 4, 7],
                [5, 4, 0, 5],
                [10, 7, 5, 0]
            ],
            "max_budget": 30.0,
            "threshold": 2,
            "expected_score": 12.445,
            "expected_route": "Start -> Museum -> Park -> End"
        },
        "Public Test Case 2 (Easy)": {
            "locations": [
                {'id': 0, 'name': 'Home', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Cafe', 'score': 7, 'category': 'Food'},
                {'id': 2, 'name': 'Waterfall', 'score': 9, 'category': 'Nature'},
                {'id': 3, 'name': 'Office', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 2, 6, 12],
                [2, 0, 5, 10],
                [6, 5, 0, 6],
                [12, 10, 6, 0]
            ],
            "max_budget": 15.0,
            "threshold": 1,
            "expected_score": 14.369,
            "expected_route": "Home -> Cafe -> Waterfall -> Office"
        },
        "Public Test Case 3 (Medium)": {
            "locations": [
                {'id': 0, 'name': 'City Center', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Fort', 'score': 10, 'category': 'Historical'},
                {'id': 2, 'name': 'Temple', 'score': 8, 'category': 'Historical'},
                {'id': 3, 'name': 'Garden', 'score': 7, 'category': 'Nature'},
                {'id': 4, 'name': 'Beach', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 4, 6, 8, 15],
                [4, 0, 3, 6, 12],
                [6, 3, 0, 4, 10],
                [8, 6, 4, 0, 7],
                [15, 12, 10, 7, 0]
            ],
            "max_budget": 25.0,
            "threshold": 1,
            "expected_score": 18.302,
            "expected_route": "City Center -> Fort -> Temple -> Garden -> Beach"
        },
        "Public Test Case 4 (Medium)": {
            "locations": [
                {'id': 0, 'name': 'Station', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Zoo', 'score': 9, 'category': 'Nature'},
                {'id': 2, 'name': 'Restaurant', 'score': 6, 'category': 'Food'},
                {'id': 3, 'name': 'Mosque', 'score': 8, 'category': 'Historical'},
                {'id': 4, 'name': 'Mall', 'score': 5, 'category': 'Food'},
                {'id': 5, 'name': 'Airport', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 5, 3, 7, 4, 18],
                [5, 0, 6, 5, 8, 14],
                [3, 6, 0, 4, 2, 15],
                [7, 5, 4, 0, 6, 12],
                [4, 8, 2, 6, 0, 14],
                [18, 14, 15, 12, 14, 0]
            ],
            "max_budget": 20.0,
            "threshold": 1,
            "expected_score": 17.182,
            "expected_route": "Station -> Restaurant -> Mall -> Mosque -> Zoo -> Airport"
        },
        "Public Test Case 5 (Hard)": {
            "locations": [
                {'id': 0, 'name': 'Hotel', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Castle', 'score': 10, 'category': 'Historical'},
                {'id': 2, 'name': 'Ruins', 'score': 8, 'category': 'Historical'},
                {'id': 3, 'name': 'Lake', 'score': 9, 'category': 'Nature'},
                {'id': 4, 'name': 'Forest', 'score': 7, 'category': 'Nature'},
                {'id': 5, 'name': 'Bistro', 'score': 6, 'category': 'Food'},
                {'id': 6, 'name': 'Hilltop', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 5, 8, 6, 10, 3, 22],
                [5, 0, 4, 7, 9, 6, 18],
                [8, 4, 0, 5, 7, 8, 16],
                [6, 7, 5, 0, 4, 7, 14],
                [10, 9, 7, 4, 0, 9, 12],
                [3, 6, 8, 7, 9, 0, 20],
                [22, 18, 16, 14, 12, 20, 0]
            ],
            "max_budget": 35.0,
            "threshold": 1,
            "expected_score": 19.830,
            "expected_route": "Hotel -> Bistro -> Castle -> Ruins -> Lake -> Forest -> Hilltop"
        },
        "Public Test Case 6 (Hard)": {
            "locations": [
                {'id': 0, 'name': 'Depot', 'score': 0, 'category': 'START'},
                {'id': 1, 'name': 'Shrine', 'score': 9, 'category': 'Historical'},
                {'id': 2, 'name': 'Monastery', 'score': 8, 'category': 'Historical'},
                {'id': 3, 'name': 'Vineyard', 'score': 7, 'category': 'Nature'},
                {'id': 4, 'name': 'Cliff', 'score': 9, 'category': 'Nature'},
                {'id': 5, 'name': 'Bakery', 'score': 5, 'category': 'Food'},
                {'id': 6, 'name': 'Diner', 'score': 6, 'category': 'Food'},
                {'id': 7, 'name': 'Lighthouse', 'score': 0, 'category': 'END'}
            ],
            "distance_matrix": [
                [0, 6, 9, 7, 11, 4, 5, 25],
                [6, 0, 4, 8, 10, 7, 8, 20],
                [9, 4, 0, 6, 8, 9, 10, 18],
                [7, 8, 6, 0, 5, 8, 9, 16],
                [11, 10, 8, 5, 0, 10, 11, 14],
                [4, 7, 9, 8, 10, 0, 3, 22],
                [5, 8, 10, 9, 11, 3, 0, 21],
                [25, 20, 18, 16, 14, 22, 21, 0]
            ],
            "max_budget": 40.0,
            "threshold": 1,
            "expected_score": 17.303,
            "expected_route": "Depot -> Shrine -> Monastery -> Cliff -> Vineyard -> Lighthouse"
        }
    }

    print("\n" + "=" * 80)
    print("             IIT DHARWAD PUBLIC TEST SUITE VERIFICATION LOG             ")
    print("=" * 80)

    for case_name, data in public_test_suite.items():
        solver = IITDharwadSightseeingSolver(
            locations=data["locations"],
            distance_matrix=data["distance_matrix"],
            max_budget=data["max_budget"],
            category_threshold=data["threshold"]
        )
        
        path, score = solver.optimize()
        _, distance = solver.calculate_route_score_and_distance(path)
        
        named_path = " -> ".join([data["locations"][idx]["name"] for idx in path])
        
        print(f"\n[ RUNNING ] {case_name}")
        print(f"  ├─ Expected Route : {data['expected_route']}")
        print(f"  ├─ Computed Route : {named_path}")
        print(f"  ├─ Target Score   : {data['expected_score']:.3f}")
        print(f"  ├─ Computed Score : \033[1;32m{score:.3f}\033[0m")
        print(f"  └─ Budget Profile : {distance:.1f} km / {data['max_budget']:.1f} km")
        
        # Validation test check
        if abs(score - data['expected_score']) < 0.01:
            print("  \033[1;32m[STATUS] VERIFIED: 100% MATCH WITH OFFICIAL ANSWER KEY!\033[0m")
        else:
            print("  \033[1;31m[STATUS] MISMATCH: Check edge criteria calculation.\033[0m")
        print("-" * 80)
