"""Visual-only meshes just to make it look nicer and closer to reality. Physics never touches these; each object collides with hidden primitives."""

import numpy as np


def crumpled_paper_mesh(radius=0.029, seed=7):
    """Visual-only crumpled ball: a subdivided icosahedron with creases. Fixed seed.

    Collision stays the hidden box and ellipsoid, so grasp physics does not change.
    """
    t = (1 + 5**0.5) / 2
    vertices = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    for _ in range(2):
        midpoints, split = {}, []
        for a, b, c in faces:
            m = []
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key not in midpoints:
                    midpoints[key] = len(vertices)
                    vertices.append(tuple((np.array(vertices[u]) + vertices[v]) / 2))
                m.append(midpoints[key])
            split += [(a, m[0], m[2]), (b, m[1], m[0]), (c, m[2], m[1]), tuple(m)]
        faces = split
    points = np.array(vertices, dtype=float)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    # Creases: folds along random planes give flat facets and sharp ridges, unlike smooth noise.
    scale = np.ones(len(points))
    for direction in rng.normal(size=(9, 3)):
        direction /= np.linalg.norm(direction)
        scale -= rng.uniform(0.08, 0.20) * np.abs(
            points @ direction + rng.uniform(-0.4, 0.4)
        )
    scale += rng.uniform(-0.10, 0.10, len(points))
    # Mean radius matches the collision ellipsoid; the clip stops spikes poking far outside it.
    points *= (radius * np.clip(scale / scale.mean(), 0.72, 1.18))[:, None]
    points[:, 2] = np.maximum(
        points[:, 2] - points[:, 2].min() - 0.004, 0.0
    )  # Flat where it rests.
    return (
        " ".join(f"{x:.5f}" for x in points.ravel()),
        " ".join(str(k) for f in faces for k in f),
    )


def can_mesh(radius, height, segments=48):
    """Visual-only can: a profile turned about z, with a tapered neck, rim bead and recessed lid.

    Collision stays a hidden cylinder, so grasp physics does not change.
    """
    r, h = radius, height
    # (radius, z) from base centre, up the outside, over the rim, down to the lid centre.
    profile = [
        (0.0, 0.003),
        (r - 0.004, 0.0),
        (r, 0.005),
        (r, h - 0.012),
        (r - 0.003, h - 0.005),
        (r - 0.0035, h - 0.002),
        (r - 0.003, h - 0.0005),
        (r - 0.004, h),
        (r - 0.005, h - 0.0005),
        (r - 0.0055, h - 0.003),
        (0.0, h - 0.003),
    ]
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = profile[1:-1]
    vertices = [(0.0, 0.0, profile[0][1])]
    for radius_k, z in rings:
        vertices += [(radius_k * np.cos(a), radius_k * np.sin(a), z) for a in angles]
    top = len(vertices)
    vertices.append((0.0, 0.0, profile[-1][1]))
    ring = lambda k, j: 1 + k * segments + j % segments
    # Counter-clockwise from outside, so normals point outwards.
    faces = [(0, ring(0, j + 1), ring(0, j)) for j in range(segments)]
    for k in range(len(rings) - 1):
        for j in range(segments):
            a, b = ring(k, j), ring(k, j + 1)
            c, d = ring(k + 1, j + 1), ring(k + 1, j)
            faces += [(a, b, c), (a, c, d)]
    last = len(rings) - 1
    faces += [(ring(last, j), ring(last, j + 1), top) for j in range(segments)]
    return (
        " ".join(f"{x:.5f}" for x in np.ravel(vertices)),
        " ".join(str(k) for f in faces for k in f),
    )
