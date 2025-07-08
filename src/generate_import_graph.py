import grimp

# Build a graph for the main pynomaly package
package_name = "pynomaly"
graph = grimp.build_graph(package_name)

print("Available methods on graph:")
print([method for method in dir(graph) if not method.startswith('_')])

print("\nExamining graph structure...")
print(f"Number of modules: {len(graph.modules)}")
print(f"First few modules: {list(graph.modules)[:10]}")

# Generate DOT representation manually
def generate_dot(graph):
    """Generate DOT representation of the import graph."""
    dot_lines = ["digraph ImportGraph {"]
    dot_lines.append("  rankdir=TB;")
    dot_lines.append("  node [shape=box];")
    
    # Add nodes
    for module in graph.modules:
        dot_lines.append(f'  "{module}" [label="{module}"];')
    
    # Add edges by finding imports for each module
    for module in graph.modules:
        # Find modules that this module imports
        for imported_module in graph.find_modules_directly_imported_by(module):
            dot_lines.append(f'  "{module}" -> "{imported_module}";')
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_representation = generate_dot(graph)

# Export to a DOT file
with open("../pynomaly_imports.dot", "w") as dot_file:
    dot_file.write(dot_representation)

print("\nGraph generated successfully!")
print(f"DOT file saved to: ../pynomaly_imports.dot")

# Check for cycles
try:
    from grimp.domain.analysis import find_cycles
    cycles = find_cycles(graph)
except ImportError:
    # Try alternative import or use a different approach
    try:
        from grimp.algorithms.cycles import find_cycles
        cycles = find_cycles(graph)
    except ImportError:
        print("Cycle detection not available in this version of grimp")
        cycles = []
if cycles:
    print(f"\nFound {len(cycles)} cycles:")
    for i, cycle in enumerate(cycles, 1):
        print(f"  Cycle {i}: {' -> '.join(cycle)}")
        
    # Generate a separate DOT file highlighting cycles
    def generate_cycle_dot(graph, cycles):
        """Generate DOT representation highlighting cycles."""
        dot_lines = ["digraph CyclicDependencies {"]
        dot_lines.append("  rankdir=TB;")
        dot_lines.append("  node [shape=box];")
        
        # Mark all modules in cycles
        modules_in_cycles = set()
        for cycle in cycles:
            modules_in_cycles.update(cycle)
        
        # Add all nodes
        for module in graph.modules:
            if module in modules_in_cycles:
                dot_lines.append(f'  "{module}" [label="{module}", color=red, style=filled, fillcolor=lightcoral];')
            else:
                dot_lines.append(f'  "{module}" [label="{module}"];')
        
        # Add edges, highlighting cyclic ones
        cyclic_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                current = cycle[i]
                next_module = cycle[(i + 1) % len(cycle)]
                cyclic_edges.add((current, next_module))
        
        for module in graph.modules:
            for imported_module in graph.find_modules_directly_imported_by(module):
                if (module, imported_module) in cyclic_edges:
                    dot_lines.append(f'  "{module}" -> "{imported_module}" [color=red, penwidth=3];')
                else:
                    dot_lines.append(f'  "{module}" -> "{imported_module}";')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    cycle_dot = generate_cycle_dot(graph, cycles)
    with open("../pynomaly_cycles.dot", "w") as dot_file:
        dot_file.write(cycle_dot)
    
    print(f"Cycle visualization saved to: ../pynomaly_cycles.dot")
else:
    print("\nNo cycles found.")

