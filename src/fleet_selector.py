import random

class FleetSelector:
    def __init__(self, total_demand: int = 50, budget: int = 10000):
        self.total_demand = total_demand
        self.budget = budget
        
        # Drone definitions
        # Format: {'type': (cost, capacity)}
        self.drones = {
            'Light': (1000, 5),   # Cheaper but low capacity
            'Heavy': (2500, 15)   # Expensive but high capacity
        }
        
    def fitness(self, genome) -> float:
        """
        Evaluates how good a fleet configuration is.
        genome is a list: [num_light_drones, num_heavy_drones]
        """
        num_light, num_heavy = genome
        cost = num_light * self.drones['Light'][0] + num_heavy * self.drones['Heavy'][0]
        capacity = num_light * self.drones['Light'][1] + num_heavy * self.drones['Heavy'][1]
        
        # Penalize solutions that exceed the budget
        if cost > self.budget:
            return -1.0  
            
        coverage_percentage = min(1.0, capacity / self.total_demand)
        budget_used_percentage = cost / self.budget
        
        # Formula provided in the project requirements
        score = (0.75 * coverage_percentage) - (0.25 * budget_used_percentage)
        return score

    def generate_population(self, pop_size=20):
        """Creates an initial random population of fleets."""
        pop = [[0, 0]] # Always guarantee at least one perfectly valid base solution
        for _ in range(pop_size - 1):
            # Randomly guess number of drones based on max possible within budget
            max_l = max(1, self.budget // self.drones['Light'][0])
            max_h = max(1, self.budget // self.drones['Heavy'][0])
            l = random.randint(0, max_l)
            h = random.randint(0, max_h)
            pop.append([l, h])
        return pop

    def crossover(self, parent1, parent2):
        """Mixes traits of two parents to create a child fleet."""
        # E.g., Take Light drones count from parent1, Heavy drones count from parent2
        if random.random() > 0.5:
            return [parent1[0], parent2[1]]
        else:
            return [parent2[0], parent1[1]]

    def mutate(self, genome):
        """Randomly alters the drone count slightly to explore new combinations."""
        idx = random.randint(0, 1) # Pick either Light or Heavy
        change = random.choice([-1, 1]) # Add or remove one
        genome[idx] += change
        
        # Ensure we don't have negative drones
        if genome[idx] < 0:
            genome[idx] = 0
        return genome

    def run_genetic_algorithm(self, generations=50, pop_size=20):
        """Executes the Genetic Algorithm to find the best fleet."""
        population = self.generate_population(pop_size)
        best_overall = None
        best_score = -999
        
        for _ in range(generations):
            # Evaluate fitness of current population
            scored_pop = [(self.fitness(g), g) for g in population]
            scored_pop.sort(key=lambda x: x[0], reverse=True) # Sort highest score first
            
            # Track the best solution found so far
            if scored_pop[0][0] > best_score:
                best_score = scored_pop[0][0]
                best_overall = list(scored_pop[0][1])
                
            # Selection: Keep the top 50% (Elitism)
            survivors = [g for score, g in scored_pop[:pop_size//2]]
            
            # Crossover & Mutation to create the next generation
            next_gen = list(survivors)
            while len(next_gen) < pop_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                child = self.crossover(p1, p2)
                
                # 20% chance to mutate
                if random.random() < 0.2:  
                    child = self.mutate(child)
                next_gen.append(child)
                
            population = next_gen
            
        return best_overall, best_score
        
if __name__ == "__main__":
    selector = FleetSelector(total_demand=50, budget=10000)
    best_fleet, score = selector.run_genetic_algorithm()
    
    num_light, num_heavy = best_fleet
    cost = num_light * 1000 + num_heavy * 2500
    capacity = num_light * 5 + num_heavy * 15
    
    print("\n" + "="*50)
    print("   AeroNet Lite: Fleet Selection Report (GA)")
    print("="*50)
    print(f"Target Demand: 50 units")
    print(f"Total Budget:  $10,000")
    print("-" * 50)
    print(f"Selected Fleet: {num_light} Light Drones, {num_heavy} Heavy Drones")
    print(f"Total Cost:     ${cost}")
    print(f"Total Capacity: {capacity} units")
    print(f"Fitness Score:  {score:.4f}")
    print("="*50 + "\n")
