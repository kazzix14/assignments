import numpy as np

PROPORTION_MUON = 0.8
PROPORTION_ELECTRON = 0.134
PROPORTION_GAMMA_RAY = 0.066

assert np.math.isclose(
    PROPORTION_MUON + PROPORTION_ELECTRON + PROPORTION_GAMMA_RAY, 1.0)


class Box:
    def __init__(self, x: float, y: float, z: float, w: float, d: float, h: float):
        self.rect_x0 = Rect(y, y + d, z, z + h, x, 2)
        self.rect_x1 = Rect(y, y + d, z, z + h, x + w, 2)
        self.rect_y0 = Rect(x, x + w, z, z + h, y, 1)
        self.rect_y1 = Rect(x, x + w, z, z + h, y + d, 1)
        self.rect_z0 = Rect(x, x + w, y, y + d, z, 0)
        self.rect_z1 = Rect(x, x + w, y, y + d, z + h, 0)

    def decays(self, origins, directions, alivenesses, kinds, lengths_org):

        # hits per ray
        # (n, 6, 3)
        hit_points = np.stack((
            self.rect_x0.inspect_intersection(origins, directions),
            self.rect_x1.inspect_intersection(origins, directions),
            self.rect_y0.inspect_intersection(origins, directions),
            self.rect_y1.inspect_intersection(origins, directions),
            self.rect_z0.inspect_intersection(origins, directions),
            self.rect_z1.inspect_intersection(origins, directions)
        ), axis=1)

        indices = np.arange(hit_points.shape[1])

        indices_x, indices_y = np.meshgrid(indices, indices)
        indices_x = np.tile(indices_x, hit_points.shape[0]).reshape(
            (hit_points.shape[0], indices.shape[0], indices.shape[0]))
        indices_y = np.tile(indices_y, (1, hit_points.shape[0], 1)).reshape(
            (hit_points.shape[0], indices.shape[0], indices.shape[0]))

        indices_ray = np.arange(hit_points.shape[0])
        indices_ray = np.stack(
            [indices_ray] * hit_points.shape[1]**2, axis=1).reshape(indices_x.shape)

        lengths = np.linalg.norm(
            hit_points[indices_ray, indices_x] - hit_points[indices_ray, indices_y], axis=3)

        lengths = lengths.reshape(
            (hit_points.shape[0], lengths.shape[1] * lengths.shape[2]))

        lengths = np.nan_to_num(lengths)
        lengths = np.nanmax(lengths, axis=1)

        lengths_org = lengths_org + lengths

        return alivenesses, lengths_org

    def inspect_intersection(self, origins, directions, alivenesses):

        # hits per ray
        hit_points = np.stack((
            self.rect_x0.inspect_intersection(origins, directions),
            self.rect_x1.inspect_intersection(origins, directions),
            self.rect_y0.inspect_intersection(origins, directions),
            self.rect_y1.inspect_intersection(origins, directions),
            self.rect_z0.inspect_intersection(origins, directions),
            self.rect_z1.inspect_intersection(origins, directions)
        ), axis=1)

        hit_points = hit_points.reshape(
            hit_points.shape[0], hit_points.shape[1] * hit_points.shape[2])

        hits = np.count_nonzero(~np.isnan(hit_points), axis=1)
        hits = 6 <= hits
        hits = hits & alivenesses

        return hits


class Rect:
    # axis
    # 0 : xy
    # 1 : xz
    # 2 : yz
    def __init__(self, x0: float, x1: float, y0: float, y1: float, z: float, axis: int):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z = z
        self.axis = axis

    def inspect_intersection(self, origins, directions):
        if self.axis == 0:
            xi = 0
            yi = 1
            zi = 2
        elif self.axis == 1:
            xi = 0
            yi = 2
            zi = 1
        elif self.axis == 2:
            xi = 1
            yi = 2
            zi = 0

        ts = (self.z - origins[:, zi]) / directions[:, zi]

        hit_point = np.empty(origins.shape)
        hit_point[:, xi] = origins[:, xi] + directions[:, xi] * ts
        hit_point[:, yi] = origins[:, yi] + directions[:, yi] * ts
        hit_point[:, zi] = np.full(origins.shape[0], self.z)

        hit_point[~((self.x0 < hit_point[:, xi]) & (hit_point[:, xi] < self.x1)
                    &
                    (self.y0 < hit_point[:, yi]) & (hit_point[:, yi] < self.y1))] = np.full(3, np.nan)

        return hit_point


def decays_all(lengths, alivenesses, kinds):

    selector = (kinds < PROPORTION_MUON)
    r = np.exp(-(lengths[selector] * 100)**2 / (2 * 111**2))
    alivenesses[selector] = np.random.uniform(
        size=np.count_nonzero(selector)) < r

    selector = (PROPORTION_MUON <= kinds) & (
        kinds < PROPORTION_MUON + PROPORTION_ELECTRON)
    r = np.exp(-((2300000 * lengths[selector]) /
                 ((0.01/1.5*10**6)**1.2-115))**3.3)
    alivenesses[selector] = np.random.uniform(
        size=np.count_nonzero(selector)) < r

    selector = (PROPORTION_MUON + PROPORTION_ELECTRON <= kinds) & (
        kinds < PROPORTION_MUON + PROPORTION_ELECTRON + PROPORTION_GAMMA_RAY)
    r = 10 ** (- 100 * lengths[selector] / 190)
    alivenesses[selector] = np.random.uniform(
        size=np.count_nonzero(selector)) < r

    return alivenesses


def build_origins(num_points, base_point, creation_distance=1000000, rand_max=1, angle_max=None):
    if angle_max is not None:
        rand_max = np.math.acos(
            np.sqrt(1 - (2 / np.pi * angle_max))) / (np.pi / 2)

    angles = np.pi/2 * \
        (1 - np.cos(np.pi/2 * np.random.uniform(0, rand_max, num_points))**2)
    thetas = np.random.rand(num_points) * 2 * np.pi

    return base_point + np.stack([
        np.cos(thetas) * np.sin(angles) * creation_distance,
        np.sin(thetas) * np.sin(angles) * creation_distance,
        np.cos(angles) * creation_distance,
    ], axis=1)


def build_destinations(num_points, target_point, radius):
    thetas = np.random.rand(num_points) * 2 * np.pi
    rs = np.random.rand(num_points) * radius
    return np.array(target_point) + np.stack([
        rs * np.cos(thetas),
        rs * np.sin(thetas),
        np.zeros(num_points),
    ], axis=1)
