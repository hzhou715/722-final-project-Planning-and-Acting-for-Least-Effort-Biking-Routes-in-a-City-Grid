# baseline_comparison.py
# Compare different navigation strategies with baselines and ablations

from simple_grid_wrapper import SimpleGridWrapper
import numpy as np
import time
import subprocess
from collections import defaultdict

def cell_name_to_coord(name):
    """Convert cell name to coordinates"""
    x, y = int(name[1:3]), int(name[3:5])
    return (x, y)

def coord_to_cell_name(coord):
    """Convert coordinates to cell name"""
    return f"c{coord[0]:02d}{coord[1]:02d}"

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_neighbors(pos, size):
    """Get valid neighbor positions"""
    x, y = pos
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

def load_plan(path="sas_plan"):
    """Load a plan from file"""
    actions = []
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(";") or not line.strip():
                    continue
                tokens = line.strip("()\n").split()
                if tokens[0] == "move":
                    from_cell, to_cell = tokens[1], tokens[2]
                    actions.append((from_cell, to_cell))
        
        # Debug: Print first few actions
        if actions:
            print(f"    📋 Loaded plan with {len(actions)} actions")
            print(f"    📋 First action: {actions[0][0]} → {actions[0][1]}")
            if len(actions) > 1:
                print(f"    📋 Second action: {actions[1][0]} → {actions[1][1]}")
    except FileNotFoundError:
        print(f"⚠️  Plan file {path} not found")
        return []
    return actions

def replan(current_pos, goal_pos, blocked_cells):
    """Generate new plan from current position"""
    # Generate new PDDL problem with blocked cells information
    cmd = ["python", "pddl_translator_fixed.py", str(current_pos[0]), str(current_pos[1])]
    
    # Add blocked cells to command line
    for bx, by in blocked_cells:
        cmd.extend([str(bx), str(by)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Run Fast Downward planner
        planner_cmd = [
            "./downward/fast-downward.py",
            "grid_costs_domain.pddl",
            "grid_problem.pddl",
            "--search",
            "astar(lmcut())"
        ]
        result = subprocess.run(planner_cmd, check=True, capture_output=True, text=True)
        
        return load_plan("sas_plan")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Command failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"    stderr: {e.stderr[:500]}")
        return []
    except Exception as e:
        print(f"⚠️  Replanning error: {e}")
        return []


class Strategy:
    """Base class for navigation strategies"""
    
    def __init__(self, name):
        self.name = name
        self.reset_metrics()
    
    def reset_metrics(self):
        self.total_cost = 0
        self.steps = 0
        self.replans = 0
        self.success = False
        self.failure_reason = None
    
    def execute(self, env, initial_plan, blocked_cells):
        """Execute strategy - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_metrics(self):
        """Return performance metrics"""
        return {
            'name': self.name,
            'success': self.success,
            'steps': self.steps,
            'total_cost': self.total_cost,
            'replans': self.replans,
            'failure_reason': self.failure_reason
        }


class NoReplanningBaseline(Strategy):
    """Baseline: Execute initial plan without replanning"""
    
    def __init__(self):
        super().__init__("No-Replanning-Baseline")
    
    def execute(self, env, initial_plan, blocked_cells):
        self.reset_metrics()
        print(f"\n{'='*50}")
        print(f"🔵 Strategy: {self.name}")
        print(f"{'='*50}")
        
        current_pos = env.agent_pos
        
        for _, to_cell in initial_plan:
            next_pos = cell_name_to_coord(to_cell)
            
            # Check if blocked
            if env.blocked[next_pos]:
                self.failure_reason = f"Blocked at {next_pos}"
                print(f"❌ {self.failure_reason} - FAILED")
                return
            
            # Check if wall
            if env.walls[next_pos]:
                self.failure_reason = f"Hit wall at {next_pos}"
                print(f"❌ {self.failure_reason} - FAILED")
                return
            
            # Execute move
            current_pos = next_pos
            cost = env.terrain[next_pos]
            self.total_cost += cost
            self.steps += 1
            
            print(f"  Step {self.steps}: {current_pos}, cost={cost}")
        
        if current_pos == env.goal_pos:
            self.success = True
            print(f"✅ Goal reached! Total cost: {self.total_cost}")
        else:
            self.failure_reason = f"Plan ended at {current_pos}, not at goal"
            print(f"❌ {self.failure_reason}")


class RandomWalkBaseline(Strategy):
    """Baseline: Random walk (lower bound performance)"""
    
    def __init__(self, max_steps=100):
        super().__init__("Random-Walk-Baseline")
        self.max_steps = max_steps
    
    def execute(self, env, initial_plan, blocked_cells):
        self.reset_metrics()
        print(f"\n{'='*50}")
        print(f"🔵 Strategy: {self.name}")
        print(f"{'='*50}")
        
        current_pos = env.agent_pos
        visited = set()
        
        while self.steps < self.max_steps:
            # Check if reached goal
            if current_pos == env.goal_pos:
                self.success = True
                print(f"✅ Goal reached! Total cost: {self.total_cost}")
                return
            
            visited.add(current_pos)
            
            # Get valid neighbors
            neighbors = get_neighbors(current_pos, env.size)
            valid_neighbors = [
                n for n in neighbors 
                if not env.blocked[n] and not env.walls[n]
            ]
            
            if not valid_neighbors:
                self.failure_reason = "No valid moves (trapped)"
                print(f"❌ {self.failure_reason}")
                return
            
            # Choose random neighbor
            next_pos = valid_neighbors[np.random.randint(len(valid_neighbors))]
            cost = env.terrain[next_pos]
            self.total_cost += cost
            self.steps += 1
            current_pos = next_pos
            
            if self.steps % 10 == 0:
                print(f"  Step {self.steps}: {current_pos}, cost={cost}")
        
        self.failure_reason = f"Max steps ({self.max_steps}) reached"
        print(f"❌ {self.failure_reason}")


class GreedyHeuristicBaseline(Strategy):
    """Baseline: Greedy approach - always move toward goal"""
    
    def __init__(self, max_steps=100):
        super().__init__("Greedy-Heuristic-Baseline")
        self.max_steps = max_steps
    
    def execute(self, env, initial_plan, blocked_cells):
        self.reset_metrics()
        print(f"\n{'='*50}")
        print(f"🔵 Strategy: {self.name}")
        print(f"{'='*50}")
        
        current_pos = env.agent_pos
        visited_states = set()
        
        while self.steps < self.max_steps:
            # Check if reached goal
            if current_pos == env.goal_pos:
                self.success = True
                print(f"✅ Goal reached! Total cost: {self.total_cost}")
                return
            
            # Get valid neighbors
            neighbors = get_neighbors(current_pos, env.size)
            valid_neighbors = [
                n for n in neighbors 
                if not env.blocked[n] and not env.walls[n]
            ]
            
            if not valid_neighbors:
                self.failure_reason = "No valid moves (trapped)"
                print(f"❌ {self.failure_reason}")
                return
            
            # Choose neighbor closest to goal
            best_neighbor = min(
                valid_neighbors, 
                key=lambda n: manhattan_distance(n, env.goal_pos)
            )
            
            # Check for loops
            state = (best_neighbor, tuple(sorted(visited_states)))
            if state in visited_states and len(visited_states) > 10:
                # Try second-best option if stuck in loop
                valid_neighbors.remove(best_neighbor)
                if valid_neighbors:
                    best_neighbor = min(
                        valid_neighbors,
                        key=lambda n: manhattan_distance(n, env.goal_pos)
                    )
            
            visited_states.add(current_pos)
            
            # Execute move
            cost = env.terrain[best_neighbor]
            self.total_cost += cost
            self.steps += 1
            current_pos = best_neighbor
            
            print(f"  Step {self.steps}: {current_pos}, cost={cost}, dist={manhattan_distance(current_pos, env.goal_pos)}")
        
        self.failure_reason = f"Max steps ({self.max_steps}) reached"
        print(f"❌ {self.failure_reason}")


class LazyLookaheadStrategy(Strategy):
    """Run-Lazy-Lookahead: Replan only when blocked"""
    
    def __init__(self):
        super().__init__("Run-Lazy-Lookahead")
    
    def execute(self, env, initial_plan, blocked_cells):
        self.reset_metrics()
        print(f"\n{'='*50}")
        print(f"🔵 Strategy: {self.name}")
        print(f"{'='*50}")
        
        plan = initial_plan
        idx = 0
        current_pos = env.agent_pos
        
        while current_pos != env.goal_pos:
            if idx >= len(plan):
                self.failure_reason = "Plan exhausted before reaching goal"
                print(f"❌ {self.failure_reason}")
                return
            
            from_cell, to_cell = plan[idx]
            from_pos = cell_name_to_coord(from_cell)
            next_pos = cell_name_to_coord(to_cell)
            
            # CRITICAL FIX: Verify the plan matches our current position
            if from_pos != current_pos:
                print(f"  ⚠️  Plan mismatch: at {current_pos} but plan says move from {from_pos}")
                print(f"  🔄 Replanning from current position...")
                self.replans += 1
                plan = replan(current_pos, env.goal_pos, blocked_cells)
                if not plan:
                    self.failure_reason = "Replanning failed"
                    print(f"❌ {self.failure_reason}")
                    return
                idx = 0
                continue
            
            # Check if blocked
            if env.blocked[next_pos]:
                print(f"  🔄 Blocked at {next_pos}, replanning...")
                self.replans += 1
                plan = replan(current_pos, env.goal_pos, blocked_cells)
                if not plan:
                    self.failure_reason = "Replanning failed"
                    print(f"❌ {self.failure_reason}")
                    return
                idx = 0
                continue
            
            # Execute move
            cost = env.terrain[next_pos]
            self.total_cost += cost
            self.steps += 1
            current_pos = next_pos
            idx += 1
            
            print(f"  Step {self.steps}: {current_pos}, cost={cost}")
        
        self.success = True
        print(f"✅ Goal reached! Total cost: {self.total_cost}, Replans: {self.replans}")


class FullLookaheadStrategy(Strategy):
    """Run-Lookahead: Replan before every step"""
    
    def __init__(self):
        super().__init__("Run-Full-Lookahead")
    
    def execute(self, env, initial_plan, blocked_cells):
        self.reset_metrics()
        print(f"\n{'='*50}")
        print(f"🔵 Strategy: {self.name}")
        print(f"{'='*50}")
        
        current_pos = env.agent_pos
        
        while current_pos != env.goal_pos:
            # Replan before each step
            print(f"  🔄 Replanning from {current_pos}...")
            self.replans += 1
            plan = replan(current_pos, env.goal_pos, blocked_cells)
            
            if not plan:
                self.failure_reason = "Replanning failed"
                print(f"❌ {self.failure_reason}")
                return
            
            # Execute first action
            _, to_cell = plan[0]
            next_pos = cell_name_to_coord(to_cell)
            
            cost = env.terrain[next_pos]
            self.total_cost += cost
            self.steps += 1
            current_pos = next_pos
            
            print(f"  Step {self.steps}: {current_pos}, cost={cost}")
        
        self.success = True
        print(f"✅ Goal reached! Total cost: {self.total_cost}, Replans: {self.replans}")


def print_comparison_table(results):
    """Print a comparison table of all strategies"""
    print(f"\n{'='*80}")
    print("📊 STRATEGY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'Success':<10} {'Steps':<10} {'Cost':<10} {'Replans':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        success = "✅" if r['success'] else "❌"
        steps = r['steps'] if r['success'] else f"{r['steps']}*"
        cost = r['total_cost'] if r['success'] else f"{r['total_cost']}*"
        replans = r['replans']
        
        print(f"{r['name']:<30} {success:<10} {steps:<10} {cost:<10} {replans:<10}")
        
        if not r['success'] and r['failure_reason']:
            print(f"  └─ Failure: {r['failure_reason']}")
    
    print(f"{'='*80}")
    print("* = Failed (partial metrics)")
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    if successful:
        print(f"\n🏆 Best performing strategy (among successful):")
        best = min(successful, key=lambda x: x['total_cost'])
        print(f"   {best['name']}: Cost={best['total_cost']}, Steps={best['steps']}, Replans={best['replans']}")


def run_experiment(seed=42, blocked_cells=[(4, 3)], start=(0, 0), goal=(7, 7)):
    """Run all strategies and compare results"""
    
    # Initialize environment
    env = SimpleGridWrapper(seed=seed, start=start, goal=goal, blocked_cells=blocked_cells)
    
    print("="*80)
    print("🧪 EXPERIMENT SETUP")
    print("="*80)
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Blocked cells: {blocked_cells}")
    print(f"Seed: {seed}")
    print("="*80)
    
    # Load initial plan (assumes you've already generated it)
    initial_plan = load_plan("sas_plan")
    if not initial_plan:
        print("⚠️  Warning: No initial plan found. Generate one first with pddl_translator.py")
        print("Some strategies may not work properly.")
    
    # Define strategies to test
    strategies = [
        NoReplanningBaseline(),
        RandomWalkBaseline(max_steps=100),
        GreedyHeuristicBaseline(max_steps=100),
        LazyLookaheadStrategy(),
        FullLookaheadStrategy(),  # Now included!
    ]
    
    # Run each strategy
    results = []
    for strategy in strategies:
        # Reset environment for each strategy
        env_copy = SimpleGridWrapper(seed=seed, start=start, goal=goal, blocked_cells=blocked_cells)
        strategy.execute(env_copy, initial_plan, blocked_cells)
        results.append(strategy.get_metrics())
    
    # Print comparison
    print_comparison_table(results)
    
    return results


if __name__ == "__main__":
    # Run experiment with default settings
    results = run_experiment(
        seed=42,
        blocked_cells=[(4, 3)],  # Block a cell that's likely on the optimal path
        start=(0, 0),
        goal=(7, 7)
    )