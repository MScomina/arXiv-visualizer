from python_tsp.exact import solve_tsp_branch_and_bound as solve_tsp
import random
import math
import networkx as nx
import matplotlib.pyplot as plt

class TSPGenerator:
    """
    Factory class for generating various types of TSP graphs.
    Supported generators:

    - "complete"          : complete graph with random edge weights
    - "watts_strogatz"    : WS small‑world graph (k neighbours, rewiring p)
    - "random_geometric"  : points uniformly at random in the unit square,
                            edges inside a radius, weights = Euclidean distance
    - "grid"              : 2‑D grid graph, weights = 1.0

    Parameters
    ----------
    seed : int or None
        Seed used for reproducibility.
    """
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------- #
    # Private helpers
    # ------------------------------------------------------------------- #
    def _build_graph(self, n_nodes: int, generator: str, params: dict):
        """Create the underlying graph before weights are applied."""
        if generator == "complete":
            G = nx.complete_graph(n_nodes)

        elif generator == "watts_strogatz":
            k = params.get("k", 4)
            p = params.get("p", 0.1)
            G = nx.watts_strogatz_graph(n_nodes, k, p,
                                        seed=self.rng.randint(0, 2**32))
            # assign coordinates using spring layout
            pos = nx.spring_layout(G, seed=self.rng.randint(0, 2**32))
            for node in G.nodes:
                G.nodes[node]["pos"] = pos[node]

        elif generator == "random_geometric":
            radius = params.get("radius", 0.2)
            dim = params.get("dim", 2)
            G = nx.random_geometric_graph(n_nodes, radius=radius,
                                          dim=dim, seed=self.rng.randint(0, 2**32))
            # Positions already stored in 'pos'
            for node in G.nodes:
                G.nodes[node]["pos"] = G.nodes[node]["pos"]

        elif generator == "grid":
            side = math.ceil(math.sqrt(n_nodes))
            G = nx.grid_2d_graph(side, side)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        else:
            raise ValueError(f"Unsupported generator: {generator}")

        # Add weights
        weight_type = params.get("weight", "random")
        if weight_type == "random":
            for u, v in G.edges():
                G[u][v]["weight"] = self.rng.uniform(1.0, 10.0)
        elif weight_type == "coordinate":
            for u, v in G.edges():
                pos_u = G.nodes[u].get("pos")
                pos_v = G.nodes[v].get("pos")
                if pos_u is None or pos_v is None:
                    raise RuntimeError(
                        "Both nodes must have 'pos' attributes for coordinate weights."
                    )
                G[u][v]["weight"] = math.dist(pos_u, pos_v)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

        return G

    # ------------------------------------------------------------------- #
    # Public API – single instance
    # ------------------------------------------------------------------- #
    def generate_one(
        self,
        n_nodes: int,
        generator: str = "complete",
        params: dict | None = None,
        weight_range: tuple = (1.0, 10.0),
    ):
        """
        Return a single weighted TSP graph.

        Parameters
        ----------
        n_nodes : int
            Number of vertices.
        generator : str
            Type of network (see the class docstring).
        params : dict, optional
            Generator specific parameters.
        weight_range : tuple
            Only used for "complete" with "random" weights.
        """
        if params is None:
            params = {}

        # Pass the weight range via the params dict if requested
        if generator == "complete" and params.get("weight", "random") == "random":
            params["weight_range"] = weight_range

        G = self._build_graph(n_nodes, generator, params)
        return G

    # ------------------------------------------------------------------- #
    # Public API – multiple instances
    # ------------------------------------------------------------------- #
    def generate_multiple(
        self,
        count: int,
        n_nodes_range: tuple = (5, 20),
        generator: str = "complete",
        params: dict | None = None,
        weight_range: tuple = (1.0, 10.0),
    ):
        """
        Generate a list of weighted TSP graphs.

        Parameters
        ----------
        count : int
            Number of instances.
        n_nodes_range : tuple
            (min_nodes, max_nodes) for random graph size.
        generator : str
            Generator type for all instances.
        params : dict, optional
            Same generator parameters used for every instance.
        weight_range : tuple
            For “complete” with “random” weights only.
        """
        if params is None:
            params = {}

        return [
            self.generate_one(
                self.rng.randint(*n_nodes_range),
                generator=generator,
                params=params,
                weight_range=weight_range,
            )
            for _ in range(count)
        ]

    # ------------------------------------------------------------------- #
    # Solve a given TSP instance
    # ------------------------------------------------------------------- #
    def solve(self, graph: nx.Graph):
        """
        Solve the optimal Hamiltonian tour on `graph`.

        Returns a tuple `(tour, total_distance)`.
        """
        n = graph.number_of_nodes()
        node_list = list(graph.nodes())
        matrix = [[graph[u].get(v, {"weight" : float("inf")})["weight"] for v in node_list] for u in node_list]
        permutation, distance = solve_tsp(matrix)
        tour = [node_list[idx] for idx in permutation]
        return tour, distance


generator = TSPGenerator()
G = generator.generate_one(
    n_nodes=25, 
    generator="watts_strogatz",
    params={
        "k" : 4,
        "p" : 0.2,
        "weight" : "coordinate",
    },
    weight_range=(1.0, 5.0)
)
tour, tour_length = generator.solve(G)
pos = nx.spring_layout(G, seed=0)

# Draw the base graph (gray edges)
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

# Add node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Highlight the TSP tour (blue arrows):
#   * `tour + [tour[0]]` closes the loop.
tour_edges = list(zip(tour, tour[1:] + [tour[0]]))
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=tour_edges,
    edge_color="steelblue",
    width=2.5,
    arrows=False   # no arrows, just a thick line
)

# Optionally annotate edge weights on the tour
for (u, v) in tour_edges:
    if G.has_edge(u, v):
        weight = G[u][v]["weight"]
        mid_pos = (pos[u] + pos[v]) / 2
        plt.text(
            mid_pos[0], mid_pos[1],
            f"{weight:.2f}",
            fontsize=8,
            color="red",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6),
        )

# Final plot tweaks
plt.title("Watts–Strogatz TSP instance\n(Tour highlighted in steel blue)")
plt.axis("off")
plt.tight_layout()
plt.show()