"""The subtopic co-occurrence map.

One node per subtopic, one edge per pair that appears on the same paper. It is
the one picture that shows *structure* rather than counts: which parts of the
corpus sit beside which, and where the bridges are.

Two things make it readable at this size. Components are laid out separately and
packed, because a real co-occurrence graph is a big core plus a scatter of pairs
and one global spring layout gives the pairs the whole rim and the core one
pixel. And the paper count is written *inside* each node, so the label beside it
is just a name and can be moved wherever it fits.

Colour follows the same magnitude ramp as every bar chart, so the map does not
introduce a second colour language.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import viz



def packed_layout(graph: nx.Graph, *, seed: int = 42, spacing: float = 1.06,
                  node_scale: float = 0.16, k_scale: float = 0.9) -> dict:
    """Lay out each connected component separately, then pack them on a spiral.

    ``spring_layout`` on a fragmented graph pushes every small component onto a
    ring and leaves the middle empty. Laying components out independently and
    packing them from the centre outwards, largest first, uses the whole canvas
    and keeps each group's internal shape legible.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    positions: dict = {}
    placed: list[tuple[float, float, float]] = []  # (x, y, radius)
    golden_angle = np.pi * (3 - np.sqrt(5))

    for index, nodes in enumerate(components):
        sub = graph.subgraph(nodes)
        if len(nodes) == 1:
            local = {next(iter(nodes)): np.zeros(2)}
            radius = 0.05
        else:
            local = nx.spring_layout(sub, seed=seed, k=k_scale / np.sqrt(len(nodes)),
                                     iterations=400, weight="weight")
            coords = np.array(list(local.values()))
            extent = np.abs(coords).max() or 1.0
            # Normalise each component, then scale by sqrt(size) so a group of 40
            # occupies more canvas than a group of 3 without dwarfing it.
            scale = node_scale * np.sqrt(len(nodes))
            local = {n: (xy / extent) * scale for n, xy in local.items()}
            radius = scale

        # Vogel spiral: even angular spread, radius growing as sqrt(index).
        step = 0.42
        target_r = step * np.sqrt(index)
        angle = index * golden_angle
        cx, cy = target_r * np.cos(angle), target_r * np.sin(angle)

        # Push outwards until this component clears everything already placed.
        for _ in range(400):
            if all(
                np.hypot(cx - px, cy - py) >= (radius + pr) * spacing
                for px, py, pr in placed
            ):
                break
            target_r += 0.05
            cx, cy = target_r * np.cos(angle), target_r * np.sin(angle)

        placed.append((cx, cy, radius))
        for node, xy in local.items():
            positions[node] = np.array([cx + xy[0], cy + xy[1]])
    return positions



def draw_topic_map(
    graph: nx.Graph,
    *,
    figsize: tuple[float, float] = (14, 9.5),
    seed: int = 42,
    title: str | None = None,
    subtitle: str | None = None,
):
    """Subtopics that share papers, laid out as a map. Returns the figure.

    If the result is a lattice of crossing leader lines, the fix is not a better
    label solver: it is plotting less. Raise ``min_papers`` on the graph until
    the map holds only subtopics a reader would name, at a size they can read
    from the back of a room -- three papers is a good starting point for a
    corpus of a hundred.
    """
    import matplotlib.patheffects as pe
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_facecolor(viz.SURFACE)
    if not graph.number_of_edges():
        return fig

    # A subtopic with no co-occurrence has nothing to say on a co-occurrence
    # map, and leaving the isolates in is what wrecks the layout: spring_layout
    # flings them to the rim and collapses everything that *is* connected into a
    # knot in the middle.
    graph = graph.subgraph([n for n in graph.nodes() if graph.degree(n)])
    # A much larger `k` than a dense graph wants: there are few nodes and each
    # carries a long label, so they need room around them.
    positions = packed_layout(graph, seed=seed, spacing=1.15, node_scale=0.30, k_scale=3.0)
    # Separation has to be relative to the layout's own extent: the spiral packs
    # the small components several units out, so a fixed distance that looks
    # generous in layout coordinates is a few pixels once the whole thing is
    # scaled into the axes.
    span = np.ptp(np.array(list(positions.values())), axis=0).max()
    positions = _spread(positions, min_distance=0.075 * span)

    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    counts = np.array([graph.nodes[n]["papers"] for n in graph.nodes()], dtype=float)
    # Scaled to the maximum rather than linear in the count: one subtopic on 26
    # papers against a floor of 3 gave a node wide enough to swallow the
    # neighbours it is meant to be connected to.
    sizes = 210 + 820 * (counts / counts.max())
    nx.draw_networkx_edges(graph, positions, ax=ax, width=0.8 + 0.8 * (weights - 1),
                           edge_color=viz.INK_MUTED, alpha=0.55)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=sizes,
                           node_color=viz.shade_by_value(counts),
                           edgecolors=viz.SURFACE, linewidths=1.8)

    # The count goes inside the node; the label beside it is then just a name.
    for node, count, marker in zip(graph.nodes(), counts, sizes):
        ax.text(*positions[node], f"{int(count)}", ha="center", va="center",
                fontsize=viz.size("label") * 0.85, fontweight="bold",
                color="white" if count / counts.max() > 0.45 else viz.INK, zorder=5)

    texts = [
        ax.text(*positions[node], node, fontsize=viz.size("label"), fontweight="bold",
                color=viz.INK, ha="center", va="bottom", zorder=4,
                path_effects=[pe.withStroke(linewidth=3.4, foreground=viz.SURFACE)])
        for node in graph.nodes()
    ]
    from matplotlib.transforms import Bbox

    ax.figure.canvas.draw()
    centres = ax.transData.transform(np.array([positions[n] for n in graph.nodes()]))
    radii = np.sqrt(sizes / np.pi) * fig.dpi / 72.0
    boxes = [Bbox([[x - r, y - r], [x + r, y + r]]) for (x, y), r in zip(centres, radii)]
    adjust_text(texts, ax=ax, objects=boxes, expand=(1.15, 1.6),
                force_text=(0.4, 0.7), force_static=(0.3, 0.6),
                force_explode=(0.05, 0.15), explode_radius=6,
                min_arrow_len=10, max_move=26,
                arrowprops={"arrowstyle": "-", "color": viz.INK_MUTED, "lw": 0.9,
                            "shrinkA": 2, "shrinkB": 4})

    if title:
        viz.figure_title(fig, title, subtitle)
    return fig



def _spread(positions: dict, *, min_distance: float, rounds: int = 220) -> dict:
    """Push overlapping nodes apart, without letting the layout drift.

    A force-directed layout minimises edge length, not node overlap, so a hub
    with several strong edges ends up sitting on its own neighbours. A few
    rounds of pairwise separation afterwards fixes that and leaves the shape of
    the layout -- which is the part carrying the information -- intact.
    """
    keys = list(positions)
    points = np.array([positions[k] for k in keys], dtype=float)
    for _ in range(rounds):
        moved = False
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.hypot(deltas[..., 0], deltas[..., 1])
        np.fill_diagonal(distances, np.inf)
        too_close = np.argwhere(distances < min_distance)
        for i, j in too_close:
            if i >= j:
                continue
            offset = points[i] - points[j]
            length = np.hypot(*offset) or 1e-6
            push = (min_distance - length) / 2 * (offset / length)
            points[i] += push
            points[j] -= push
            moved = True
        if not moved:
            break
    return {k: p for k, p in zip(keys, points)}
