# pddl_translator_fixed.py
# Converts SimpleGridWrapper output to a PDDL problem file with action costs
# Now properly handles blocked cells during replanning

from simple_grid_wrapper import SimpleGridWrapper
import numpy as np
import os
import sys

# Parse command line arguments
# Format: python pddl_translator.py start_x start_y [blocked_x1 blocked_y1 blocked_x2 blocked_y2 ...]
start = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 0)

# Parse blocked cells from remaining arguments (pairs of coordinates)
blocked_cells = []
if len(sys.argv) > 3:
    blocked_args = sys.argv[3:]
    for i in range(0, len(blocked_args), 2):
        if i + 1 < len(blocked_args):
            blocked_cells.append((int(blocked_args[i]), int(blocked_args[i+1])))

print(f"📍 Generating PDDL from position: {start}")
print(f"🚫 Blocked cells: {blocked_cells}")

# Use SimpleGridWrapper ONLY to get walls and terrain
# But don't rely on it for the start position since reset() might fail
try:
    # Try to create wrapper with default positions to get terrain/walls
    env = SimpleGridWrapper(seed=42, start=(0, 0), goal=(7, 7), blocked_cells=[])
    state = env.get_state()
    terrain = state['terrain']
    walls = state['walls']
    goal = state['goal']
    print(f"✅ Using SimpleGridWrapper for terrain and walls")
except Exception as e:
    print(f"⚠️  SimpleGridWrapper failed: {e}")
    print(f"⚠️  Generating terrain manually")
    # Fallback: generate terrain manually
    SEED = 42
    GRID_SIZE = 8
    rng = np.random.default_rng(SEED)
    terrain = rng.integers(1, 6, size=(GRID_SIZE, GRID_SIZE))
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    goal = (7, 7)

def pos_to_cellname(r, c):
    """Convert position (row, col) to cell name"""
    return f"c{r:02d}{c:02d}"

def generate_problem_file(start_pos, goal_pos, blocked_cells, terrain, walls, 
                          domain_name="grid-costs", problem_name="grid5",
                          output_path="grid_problem.pddl"):
    """Generate a PDDL problem file from the grid state"""
    
    blocked_set = set(blocked_cells)
    size = terrain.shape[0]
    
    with open(output_path, "w") as f:
        # Problem definition
        f.write(f"(define (problem {problem_name})\n")
        f.write(f"  (:domain {domain_name})\n")
        
        # Objects (cells)
        f.write("  (:objects\n")
        for r in range(size):
            for c in range(size):
                f.write(f"    {pos_to_cellname(r,c)} - cell\n")
        f.write("  )\n\n")

        # Initial state
        f.write("  (:init\n")
        
        # Agent at specified start position (NOT from SimpleGridWrapper)
        agent_cell = pos_to_cellname(start_pos[0], start_pos[1])
        f.write(f"    (at {agent_cell})\n")
        
        # Set move costs and adjacencies for each cell
        for r in range(size):
            for c in range(size):
                name = pos_to_cellname(r, c)
                is_blocked = (r, c) in blocked_set
                
                # Only set reasonable move cost if the cell is not a wall or blocked
                if not walls[r, c] and not is_blocked:
                    f.write(f"    (= (move-cost {name}) {terrain[r,c]})\n")
                else:
                    # Walls and blocked cells have very high cost (effectively impassable)
                    f.write(f"    (= (move-cost {name}) 9999)\n")
                
                # Define adjacencies (only between valid cells)
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbor_blocked = (nr, nc) in blocked_set
                        # Only add adjacency if neither cell is a wall or blocked
                        if not walls[r, c] and not walls[nr, nc] and not is_blocked and not neighbor_blocked:
                            neighbor = pos_to_cellname(nr, nc)
                            f.write(f"    (adj {name} {neighbor})\n")
        
        # Set total cost to 0 initially (if using metric)
        f.write("    (= (total-cost) 0)\n")
        f.write("  )\n\n")
        
        # Goal
        goal_cell = pos_to_cellname(goal_pos[0], goal_pos[1])
        f.write(f"  (:goal (at {goal_cell}))\n")
        
        # Metric (minimize total cost)
        f.write("  (:metric minimize (total-cost))\n")
        f.write(")\n")
    
    print(f"✅ PDDL problem written to {output_path}")
    return output_path

def parse_plan(plan_file):
    """Parse a plan file and extract the sequence of moves"""
    if not os.path.exists(plan_file):
        print(f"❌ Plan file {plan_file} not found")
        return []
    
    moves = []
    with open(plan_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith(';') or not line:
                continue
            # Parse move actions (format: (move from to))
            if line.startswith('(move'):
                parts = line.replace('(', '').replace(')', '').split()
                if len(parts) == 3:
                    _, from_cell, to_cell = parts
                    moves.append((from_cell, to_cell))
    
    return moves

def cellname_to_pos(cellname):
    """Convert cell name back to position (row, col)"""
    # Assuming format cRRCC where RR is row and CC is column
    if cellname.startswith('c') and len(cellname) == 5:
        row = int(cellname[1:3])
        col = int(cellname[3:5])
        return (row, col)
    else:
        raise ValueError(f"Invalid cell name format: {cellname}")


if __name__ == "__main__":
    # Generate PDDL problem file
    # Use the START position from command line, not from SimpleGridWrapper
    problem_file = generate_problem_file(
        start_pos=start,  # Use command-line argument
        goal_pos=goal,     # From SimpleGridWrapper (or fallback)
        blocked_cells=blocked_cells,
        terrain=terrain,   # From SimpleGridWrapper (or fallback)
        walls=walls,       # From SimpleGridWrapper (or fallback)
        output_path="grid_problem.pddl"
    )
    
    print(f"\nGrid Information:")
    print(f"  Agent starts at: {start}")
    print(f"  Goal is at: {goal}")
    print(f"  Grid size: {terrain.shape[0]}x{terrain.shape[1]}")
    print(f"  Blocked cells: {blocked_cells}")
    
    # Show a sample of terrain costs
    print(f"\nSample terrain costs (top-left 4x4):")
    for r in range(min(4, terrain.shape[0])):
        row_str = ""
        for c in range(min(4, terrain.shape[1])):
            if walls[r, c]:
                row_str += " W "
            elif (r, c) in blocked_cells:
                row_str += " X "
            else:
                row_str += f" {terrain[r, c]} "
        print(row_str)
    
    print(f"\nGrid Information:")
    print(f"  Agent starts at: {state['agent']}")
    print(f"  Goal is at: {state['goal']}")
    print(f"  Grid size: {state['size']}x{state['size']}")
    print(f"  Blocked cells: {blocked_cells}")
    
    # Show a sample of terrain costs
    print(f"\nSample terrain costs (top-left 4x4):")
    for r in range(min(4, state['size'])):
        row_str = ""
        for c in range(min(4, state['size'])):
            if state['walls'][r, c]:
                row_str += " W "
            elif (r, c) in blocked_cells:
                row_str += " X "
            else:
                row_str += f" {state['terrain'][r, c]} "
        print(row_str)