# Graph Analyzer

This is a Python-based program for **graph analysis and visualization**.
It allows the user to input a graph (either manually or via CSV file) and perform various analyses such as checking connectivity, detecting cycles, finding shortest paths, performing topological sorting, and more.

It can also **visualize the graph** and save it as an image.

---

## Features

* Input graph manually or from a CSV file
* Support for **directed** and **undirected** graphs
* Support for **weighted** and **unweighted** graphs
* Check **graph simplicity** (simple graph / multigraph / self-loops)
* Calculate **degree** (in-degree, out-degree, total degree)
* Identify **isolated nodes**
* Detect **cycles**
* Check **graph connectivity** and count possible simple paths between two nodes
* Find **shortest path** between two nodes (BFS for unweighted, Dijkstra for weighted)
* Perform **topological sorting** (for directed acyclic graphs)
* Find **connected components** (strongly, weakly, or undirected)
* **Visualize and save** the graph as a PNG file

---

## Implementation Note

All **BFS**, **DFS**, and **topological sort** algorithms are **implemented from scratch** by the author.
The only libraries used are:

* `networkx` → for **visualization only**
* `matplotlib` → for **saving and displaying graphs**

---

## Input

You can choose one of the following input methods:

1. **Manual input**: Enter number of nodes, edges, and the edge connections directly in the terminal.
2. **CSV file**: Provide a file path containing the edge list:

   * For weighted graphs:

     ```
     1,2,5
     2,3,2
     1,3,4
     ```
   * For unweighted graphs:

     ```
     1,2
     2,3
     3,4
     ```

You will also be prompted to specify:

* If the graph is **directed** (`y` or `n`)
* If the graph is **weighted** (`y` or `n`)

---

## Output

The program can output:

* Basic graph type and properties (point, null, line, complete, simple, multigraph, cycle, tree)
* Degree information for specific nodes
* Whether the graph is connected
* Number of possible simple paths between two nodes
* Shortest path between two nodes
* Topological order (if applicable)
* Connected components
* Visualization image saved as `graph_visualization.png`

Example output:

```
Graph type: Simple, Complete
 - Simple: Graph has no loops or multiple edges between the same nodes.
 - Complete: Every pair of distinct nodes is connected by an edge.

The graph is connected.
There are 3 possible simple paths from 1 to 4.
Shortest path: 1 -> 2 -> 4
```

---

## Example Graph Visualization

**Input CSV:**

```
1,2
2,3
3,1
```

**Generated Image:**
(graph\_visualization.png)

---

## Requirements

* Python 3.x
* Libraries: `networkx`, `matplotlib`

Install requirements:

```bash
pip install networkx matplotlib
```

---

## Run the Code

```bash
python "Graph Analyzer.py"
```

---

**Author:** Tina Tavakolifar
GitHub: [https://github.com/tinatavakolifar](https://github.com/tinatavakolifar)
