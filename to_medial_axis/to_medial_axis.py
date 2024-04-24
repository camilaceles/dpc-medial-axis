import numpy as np
from scipy.spatial import KDTree
from pygel3d import hmesh


def __find_connection_points(medial_sheet: hmesh.Manifold, curves_positions: list[np.ndarray]) -> list[np.ndarray]:
    kd = KDTree(medial_sheet.positions())

    connections = []
    # for each curve, find point in medial sheet to which is connected (closest)
    for i, curve in enumerate(curves_positions):
        start, end = curve[0], curve[-1]
        dist_start, closest_start = kd.query(start, k=1)
        dist_end, closest_end = kd.query(end, k=1)

        # take endpoint closest to sheet as connection
        if dist_start < dist_end:
            connections.append(closest_start)
        else:
            curves_positions.reverse()  # reverse curve so first index is where it connects to sheet
            connections.append(closest_end)

    return connections


def to_medial_axis(
        medial_sheet: hmesh.Manifold,
        sheet_correspondences: list[list[int]],
        curve_positions: list[np.ndarray],
        curve_correspondences: list[list[list[int]]]
) -> dict:
    vertices = np.copy(medial_sheet.positions())
    faces = np.array([medial_sheet.circulate_face(fid) for fid in medial_sheet.faces()])

    connections = __find_connection_points(medial_sheet, curve_positions)
    curves_with_connection = []
    for i, curve in enumerate(curve_positions):
        n = len(vertices)
        idx = [connections[i]] + list(range(n, n + len(curve)))
        vertices = np.concatenate([vertices, curve])
        curves_with_connection.append(idx)

    correspondences = sheet_correspondences
    for curve_corr in curve_correspondences:
        correspondences += curve_corr

    correspondences = np.array(correspondences, dtype=object)
    return {
        "vertices": vertices,
        "medial_sheet_vertices": medial_sheet.positions(),
        "medial_sheet_faces": faces,
        "medial_curves": curves_with_connection,
        "correspondences": correspondences
    }
