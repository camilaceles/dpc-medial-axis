import numpy as np
from pygel3d import graph
from commons.point import PointSet, Point


def __get_curve_indices(inner_points: PointSet) -> list[list[int]]:
    # Use graph as auxiliary data structure to find connections between curve points
    g = graph.Graph()
    for q in inner_points:
        g.add_node(q.pos)

    for q in inner_points:
        if not q.is_fixed:
            g.connect_nodes(q.index, q.front_point)
            g.connect_nodes(q.index, q.back_point)

    visited = set()

    def __expand_from_endpoint(endpoint: int) -> list[int]:
        path = [endpoint]
        while True:
            visited.add(endpoint)
            next_nodes = [n for n in g.neighbors(endpoint) if n not in visited]
            if not next_nodes:
                break
            endpoint = next_nodes[0]
            path.append(endpoint)
        return path

    curves = []
    for node in g.nodes():
        if node not in visited and len(g.neighbors(node)) == 1:
            # Start points are unvisited nodes with 1 neighbors (curve endpoint)
            curve_points = __expand_from_endpoint(node)
            curves.append(curve_points)
    return curves


def __kep_n_longest(curves: list[list[int]], inner_points: PointSet, keep_n_curves: int = None) -> list[list[int]]:
    curves.sort(key=lambda s: -len(s))  # Sort curves by length (descending)
    if keep_n_curves is not None:
        for curve in curves[keep_n_curves:]:
            for point in curve:
                inner_points[point].is_fixed = True
        curves = curves[:keep_n_curves]
    return curves


def __group_every_n(positions: np.ndarray, indices: list[int], n: int) -> tuple[np.ndarray, list[list[int]]]:
    num_points = positions.shape[0]
    full_length = num_points - num_points % n

    main_part = positions[:full_length]
    remainder = positions[full_length:]

    indices = np.array(indices, dtype=int)
    main_part_indices = indices[:full_length]
    remainder_indices = indices[full_length:]

    grouped_indices = []
    if full_length > 0:
        reshaped_main = main_part.reshape(-1, n, 3)
        main_averages = reshaped_main.mean(axis=1)

        grouped_indices = list(main_part_indices.reshape(-1, n))
    else:
        main_averages = np.array([]).reshape(0, 3)

    if remainder.size > 0:
        remainder_average = np.array([remainder.mean(axis=0)])
        combined_averages = np.vstack((main_averages, remainder_average))
        grouped_indices.append(remainder_indices)
    else:
        combined_averages = main_averages

    return combined_averages, list(grouped_indices)


def to_medial_curves(
        inner_points: PointSet,
        keep_n_curves: int = None,
        group_every_n: int = 5
) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    curves = __get_curve_indices(inner_points)
    curves = __kep_n_longest(curves, inner_points, keep_n_curves)

    grouped_curve_pos = []
    grouped_correspondences = []
    for curve in curves:
        curve_pos = np.array([inner_points.positions[i] for i in curve])
        new_pos, new_corr = __group_every_n(curve_pos, curve, group_every_n)

        grouped_curve_pos.append(new_pos)
        grouped_correspondences.append(new_corr)

    return grouped_curve_pos, grouped_correspondences
