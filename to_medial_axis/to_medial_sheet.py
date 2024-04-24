import trimesh
import numpy as np
import random
from pygel3d import hmesh
from commons.point import PointSet
from collections import deque
from scipy.spatial import KDTree
from commons.utils import trimesh_to_manifold, manifold_to_trimesh


def __precompute_face_adjacencies(mesh):
    adjacency_dict = {}
    for adj_index, faces in enumerate(mesh.face_adjacency):
        for face in faces:
            if face not in adjacency_dict:
                adjacency_dict[face] = []
            adjacency_dict[face].append((adj_index, set(faces) - {face}))
    return adjacency_dict


def __expand_from_triangle(mesh: trimesh.Trimesh, start_face: int, angle_threshold_degrees: float):
    angle_threshold_radians = np.radians(angle_threshold_degrees)

    sheet_faces = {start_face}
    to_visit = deque([start_face])

    adjacency_dict = __precompute_face_adjacencies(mesh)

    while to_visit:
        current_face = to_visit.popleft()

        if current_face in adjacency_dict:
            for adj_index, other_faces in adjacency_dict[current_face]:
                adj_face = next(iter(other_faces))

                if adj_face not in sheet_faces:
                    angle = mesh.face_adjacency_angles[adj_index]

                    if angle < angle_threshold_radians:
                        sheet_faces.add(adj_face)
                        to_visit.append(adj_face)

    return sheet_faces


def __single_sheet(m: hmesh.Manifold, dihedral_angle_threshold: float):
    """Extract a single sheet from double-sheet"""
    faces = np.array([m.circulate_face(fid) for fid in m.faces()])
    trimesh_mesh = trimesh.Trimesh(vertices=m.positions(), faces=faces)

    sheet_faces = __single_sheet_faces(trimesh_mesh, dihedral_angle_threshold)
    sheet_mesh = trimesh_mesh.submesh([np.array(sheet_faces)], append=True)

    sheet = trimesh_to_manifold(sheet_mesh)
    return sheet


def __single_sheet_embedded(m: hmesh.Manifold, dihedral_angle_threshold: float):
    """Extract a single sheet from double-sheet"""
    trim = manifold_to_trimesh(m)

    sheet_faces = __single_sheet_faces(trim, dihedral_angle_threshold)
    sheet_mesh = trim.submesh([np.array(sheet_faces)], append=True)

    on_sheet = list(set(trim.faces[sheet_faces].flatten()))
    on_sheet_mask = np.zeros(len(trim.vertices), dtype=bool)
    on_sheet_mask[on_sheet] = True
    leftovers = trim.vertices[~on_sheet_mask]

    prox_query = trimesh.proximity.ProximityQuery(sheet_mesh)
    _, _, face_ids = prox_query.on_surface(leftovers)

    sheet = trimesh_to_manifold(sheet_mesh)

    for point, fid in zip(leftovers, face_ids):
        vid = sheet.split_face_by_vertex(fid)
        sheet.positions()[vid] = point
    sheet.cleanup()

    return sheet


def __single_sheet_faces(trim: trimesh.Trimesh, dihedral_angle_threshold: float):
    n_org_faces = len(trim.faces)

    sheet_faces = []
    max_faces_length = 0
    # if a bad start face is chosen, the resulting mesh is only a few triangles,
    # so we try until it results in at least 30% of the original face count
    for i in range(10):
        start_face = random.choice(range(len(trim.faces)))
        potential_sheet_faces = __expand_from_triangle(trim, start_face, dihedral_angle_threshold)

        # if good enough, return immediately
        if len(potential_sheet_faces) > 0.2 * n_org_faces:
            return list(potential_sheet_faces)

        # otherwise return the largest sheet found
        if len(potential_sheet_faces) > max_faces_length:
            sheet_faces = list(potential_sheet_faces)
            max_faces_length = len(sheet_faces)

    return sheet_faces


def __remove_small_faces(m: hmesh.Manifold):
    trim = manifold_to_trimesh(m, process=True)
    face_areas = trim.area_faces
    faces_to_keep = face_areas > (0.15 * np.mean(face_areas))
    trim.update_faces(faces_to_keep)
    trim.remove_unreferenced_vertices()

    m = trimesh_to_manifold(trim)
    m.cleanup()
    return m


def __smooth_sheet(m: hmesh.Manifold, smooth_iterations: int):
    for _ in range(smooth_iterations):
        for vid in m.vertices():
            if m.is_vertex_at_boundary(vid):
                continue

            neighbors = m.circulate_vertex(vid)
            new_pos = np.mean(m.positions()[neighbors], axis=0)
            m.positions()[vid] = new_pos


def map_correspondences(medial_sheet: hmesh.Manifold, inner_points: PointSet) -> list[list[int]]:
    single_sheet_pos = medial_sheet.positions()
    sheet_indices = inner_points.is_fixed
    kd_tree = KDTree(single_sheet_pos)

    inner_indices = np.arange(inner_points.N)
    sheet_correspondences = [[] for _ in range(len(medial_sheet.vertices()))]

    _, sheet_inner_indices = kd_tree.query(inner_points.positions[sheet_indices])
    inner_indices[sheet_indices] = sheet_inner_indices

    for i in range(inner_points.N):
        if not inner_points.is_fixed[i]:
            continue
        sheet_correspondences[inner_indices[i]].append(i)

    return sheet_correspondences


def to_medial_sheet(
        input_mesh: hmesh.Manifold,
        inner_points: PointSet,
        dihedral_angle_threshold: float,
        smooth_iterations: int = 5
) -> hmesh.Manifold:
    # make mesh out of original connectivity
    inner_mesh = hmesh.Manifold(input_mesh)
    sheet_indices = inner_points.is_fixed

    pos = inner_mesh.positions()
    pos[sheet_indices] = inner_points.positions[sheet_indices]

    for q in inner_points:
        if not q.is_fixed:
            inner_mesh.remove_vertex(q.index)
    inner_mesh.cleanup()

    # extract single sheet & postprocess
    single_sheet = __single_sheet(inner_mesh, dihedral_angle_threshold)
    single_sheet = __remove_small_faces(single_sheet)
    __smooth_sheet(single_sheet, smooth_iterations)

    return single_sheet
