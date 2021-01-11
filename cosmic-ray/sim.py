import datetime
import time
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
import numpy as np
import sympy as sym

from sim_prelude import *
from sim_shields import build_shields

floors = [0, 3, 5, 6]


def simulate(thickness, num_muons, distribution_radius, angle_max, creation_height, lower_detector_size, upper_detector_size, detector_spacing, lower_detector_offset, upper_detector_offset, do_plot=False):

    counts_triggered = []
    for idx, floor in enumerate(floors):
        lower_detector_origin = np.array(
            lower_detector_offset[idx]) + np.array((0, 0, 0.6+thickness)) + np.array((0, 0, floor * 3.6))

        upper_detector_origin = lower_detector_origin + \
            np.array(upper_detector_offset[idx])

        kinds = np.random.uniform(size=num_muons)

        # declare origin
        origins = build_origins(
            num_muons, (0, 0, lower_detector_origin[2]), angle_max=angle_max)
        destinations = build_destinations(
            num_muons, lower_detector_origin + (0.125 / 2, 0.125 / 2, detector_spacing/2), distribution_radius)
        directions = destinations - origins

        alivenesses = np.full(num_muons, True)
        lengths = np.zeros(num_muons, dtype=np.float32)

        shields = build_shields(thickness)

        for shield in shields:
            if shield.rect_z1.z <= upper_detector_origin[2]:
                continue
            alivenesses, lengths = shield.decays(
                origins, directions, alivenesses, kinds, lengths)

        alivenesses = decays_all(lengths, alivenesses, kinds)

        lower_detector = Box(*lower_detector_origin, *lower_detector_size)
        upper_detector = Box(*upper_detector_origin, *upper_detector_size)

        hits_lower_detector = lower_detector.inspect_intersection(
            origins, directions, alivenesses)
        hits_upper_detector = upper_detector.inspect_intersection(
            origins, directions, alivenesses)

        hits_triggered = hits_upper_detector & hits_lower_detector
        count_triggered = np.count_nonzero(hits_triggered)
        counts_triggered.append(count_triggered)

        print(f'F{int(floor) + 1} : {thickness:>5.5f}cm')
        #print('  mean length', np.mean(lengths[hits_triggered]))
        #print('  lower detector count: ', np.count_nonzero(hits_lower_detector))
        #print('  upper detector count: ', np.count_nonzero(hits_upper_detector))
        #print('  triggered count: ', count_triggered)

    ### plot ###

    counts_triggered = np.array(counts_triggered)
    counts_triggered_normalized = counts_triggered / counts_triggered[-1]

    se95s = []
    for i, f in enumerate(floors):
        count_outdoor = counts_triggered[-1]
        count = counts_triggered[i]
        se68_i = 1/np.math.sqrt(count)
        se68_o = 1/np.math.sqrt(count_outdoor)
        se68 = np.math.sqrt(se68_i**2 + se68_o**2)
        se95 = 2 * se68
        se95s.append(se95)

        count_normalized = counts_triggered_normalized[i]
        count_plus_se95 = count_normalized * (1 + se95)
        count_minus_se95 = count_normalized * (1 - se95)

    diffs = []
    for i in range(0, len(floors)):
        for j in range(i+1, len(floors)):
            diffs.append(("F{} - F{} : {:>4.2%}".format(i+1, j+1,
                                                        abs(counts_triggered_normalized[i] - counts_triggered_normalized[j])), abs(counts_triggered_normalized[i] - counts_triggered_normalized[j])))

    diffs = sorted(diffs, key=lambda diff: diff[1])
    # for diff in diffs:
    # print(diff[0])

    if do_plot:
        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.scatter(list(map(lambda f: f+1, floors)),
                    counts_triggered_normalized)
        axe.plot([f+1, f+1], [count_minus_se95, count_plus_se95])
        plt.show()

    return (counts_triggered_normalized, se95s)


'''


[thicknesses
    [flux1, flux4, flux6] on 5cm
    [flux1, flux4, flux6] on 6cm
    [flux1, flux4, flux6] on 7cm
    [flux1, flux4, flux6] on 8cm
    ...
]
'''


# 1 4 6 7
vinyl_thickness = 0.005
lower_detector_size = (0.1335 - vinyl_thickness, 0.138 -
                       vinyl_thickness, 0.038 - vinyl_thickness)

upper_detector_size = (0.13 - vinyl_thickness, 0.13 -
                       vinyl_thickness, 0.037 - vinyl_thickness)

detector_spacing = 0.245

lower_offsets = [(0, 0, 0), (0.128392 - lower_detector_size[0], -0.05651639893, 0.158), (0.0674914694 -
                                                                                         lower_detector_size[0], 0.00368017063, 0.158), (9.2 - lower_detector_size[0], 1.545, 0.159)]
upper_offsets = [(0, 0, detector_spacing), (-(upper_detector_size[0] - lower_detector_size[0]), -(upper_detector_size[1] - lower_detector_size[1]), detector_spacing),
                 (-(upper_detector_size[0] - lower_detector_size[0]), -(upper_detector_size[1] - lower_detector_size[1]), detector_spacing), (-(upper_detector_size[0] - lower_detector_size[0]), -(upper_detector_size[1] - lower_detector_size[1]), detector_spacing)]

# 7 6 4 1になってる
lower_offsets.reverse()
upper_offsets.reverse()

thicknesses = [t for t in np.arange(0.10, 0.40, 0.005)]
counts_normalized_per_floors_per_thicknesses = []
se95s_per_floors_per_thicknesses = []
for thickness in thicknesses:
    counts_normalized, se95s = simulate(thickness, 450000, 0.1, 30 * np.pi/180, 100000,
                                        lower_detector_size, upper_detector_size,  0.245, lower_offsets, upper_offsets)

    counts_normalized_per_floors_per_thicknesses.append(counts_normalized)
    se95s_per_floors_per_thicknesses.append(se95s)

counts = np.array(counts_normalized_per_floors_per_thicknesses).T
#plt.scatter(thicknesses, counts[0])
# plt.show()
se95s = np.array(se95s_per_floors_per_thicknesses).T
print(counts)
print(se95s)

while(1):
    filename = f'simulated-counts-{datetime.datetime.now()}.txt'
    if not os.path.exists(filename):
        out_file = open(filename, 'w')
        print(counts, file=out_file)
        print(se95s, file=out_file)
        out_file.close()
        exit()
    else:
        time.sleep(1)
