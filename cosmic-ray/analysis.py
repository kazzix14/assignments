import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
import numpy as np
import scipy.optimize as opt


thicknesses = np.arange(10.0, 40.0 + 0.5, 0.5)

fluxes_simulated_per_thickness_f6 = [
    0.8319175, 0.82661546, 0.82614938, 0.82543729, 0.82530116, 0.82168654,
    0.81924938, 0.82164124, 0.82097687, 0.82042085, 0.81598687, 0.81768784,
    0.81541089, 0.80986429, 0.81328365, 0.81304818, 0.81192241, 0.80658565,
    0.806223,   0.80566588, 0.80282756, 0.803789,   0.8036566,  0.80250813,
    0.80100939, 0.79884532, 0.79760642, 0.79808557, 0.79527412, 0.79511105,
    0.79475021, 0.78901999, 0.78735283, 0.78967789, 0.78779818, 0.78548583,
    0.7828678,  0.78054383, 0.78275355, 0.77943707, 0.77986067, 0.78022116,
    0.77473603, 0.77513013, 0.77161376, 0.77425566, 0.76997688, 0.76765046,
    0.76527844, 0.76574639, 0.76207996, 0.76388601, 0.76146278, 0.76018433,
    0.7580834,  0.75748236, 0.75096254, 0.7498794,  0.74953551, 0.74910264,
    0.74924159
]  # 各階での減衰率を厚さごとに保存

fluxes_simulated_per_thickness_f4 = [
    0.79647896, 0.79175142, 0.78740971, 0.78394444, 0.78254517, 0.7720542,
    0.76905779, 0.765959,   0.7627511,  0.75622537, 0.75193619, 0.746147,
    0.73884032, 0.73351833, 0.73181289, 0.72731078, 0.71832233, 0.71056692,
    0.70683759, 0.70382869, 0.69788828, 0.68791394, 0.68389561, 0.6790747,
    0.67409293, 0.66758488, 0.66260693, 0.65521099, 0.65047494, 0.64301303,
    0.6338961,  0.62804729, 0.62259037, 0.61867578, 0.61422325, 0.60380878,
    0.59712473, 0.58749727, 0.5842843,  0.57705824, 0.56909026, 0.55918248,
    0.55567355, 0.55071093, 0.54491978, 0.53841582, 0.5303803,  0.52360505,
    0.5141922,  0.50849521, 0.50571562, 0.49796096, 0.49060048, 0.48487057,
    0.47688016, 0.47074032, 0.46171779, 0.45274666, 0.44855703, 0.44225588,
    0.4360552
]

fluxes_simulated_per_thickness_f1 = [
    0.58708681, 0.57676192, 0.56410653, 0.55354834, 0.54197054, 0.53129022,
    0.51880206, 0.50702582, 0.49561357, 0.48383231, 0.47131677, 0.46164468,
    0.44755418, 0.43590034, 0.42457873, 0.41466498, 0.40294912, 0.39065164,
    0.37750147, 0.36818466, 0.35653161, 0.34606873, 0.3335593,  0.32466842,
    0.31247158, 0.30358064, 0.29313827, 0.28227503, 0.27357862, 0.26214212,
    0.25220536, 0.24448198, 0.23298571, 0.22595982, 0.21679377, 0.2075119,
    0.2009522,  0.19048622, 0.18548309, 0.17563224, 0.16999016, 0.16245972,
    0.15463592, 0.14757326, 0.14170044, 0.13556059, 0.12894509, 0.12335317,
    0.117643,   0.1116798,  0.10509025, 0.101765,   0.09608895, 0.09205079,
    0.08716089, 0.08285052, 0.07974446, 0.07435514, 0.07116257, 0.06751532,
    0.06465537
]

standard_errors_simulated_per_thickness_f6 = [
    0.00599061, 0.00599653, 0.00599673, 0.00599724, 0.00600027, 0.00600053,
    0.00600463, 0.00600391, 0.00600289, 0.00600454, 0.00601433, 0.00600924,
    0.0060176, 0.0060177, 0.00602345, 0.00602105, 0.00602172, 0.00602346,
    0.00602942, 0.00603374, 0.00604431, 0.00603191, 0.00603665, 0.00603942,
    0.00604293, 0.00604943, 0.0060527, 0.00605487, 0.00605691, 0.00605811,
    0.00605838, 0.00606842, 0.00606539, 0.00607199, 0.00607622, 0.00607877,
    0.00608316, 0.00607992, 0.00608035, 0.0060875, 0.0060915, 0.00609027,
    0.00609428, 0.00609856, 0.00609761, 0.00610849, 0.00611254, 0.00611705,
    0.00612108, 0.00611863, 0.00612844, 0.00612814, 0.0061334, 0.00614036,
    0.00614158, 0.00614703, 0.00615452, 0.00615135, 0.00615434, 0.00616517,
    0.00616664
]

standard_errors_simulated_per_thickness_f4 = [
    0.00606292, 0.00606838, 0.00607697, 0.00608356, 0.00608941, 0.00610549,
    0.00611139, 0.00612254, 0.00612742, 0.00614295, 0.00615377, 0.00616569,
    0.00618698, 0.00618831, 0.00620561, 0.0062137, 0.00623449, 0.00624467,
    0.0062597, 0.00627083, 0.00629133, 0.00630727, 0.00632291, 0.00633661,
    0.00635094, 0.00637147, 0.0063865, 0.00641151, 0.00642145, 0.00644489,
    0.00647251, 0.00648857, 0.00649892, 0.00652404, 0.00653883, 0.00657103,
    0.00659251, 0.0066172, 0.00663437, 0.00666043, 0.00669534, 0.00673256,
    0.00673723, 0.00676244, 0.00677582, 0.00682097, 0.0068483, 0.00687638,
    0.00691606, 0.00694003, 0.00695434, 0.00699455, 0.00702919, 0.00706164,
    0.0070972, 0.00713318, 0.00717147, 0.00721322, 0.00723894, 0.00728605,
    0.00732401
]

standard_errors_simulated_per_thickness_f1 = [
    0.00663754, 0.00666981, 0.00671627, 0.00675608, 0.0068055, 0.00684177,
    0.00689443, 0.00695167, 0.00700183, 0.00705923, 0.0071231, 0.00717167,
    0.00725302, 0.00730604, 0.00738921, 0.00744738, 0.00752149, 0.00759375,
    0.00769492, 0.00776935, 0.00786768, 0.00794119, 0.00805702, 0.00813982,
    0.00825939, 0.00835373, 0.00846803, 0.00859762, 0.00869793, 0.00884689,
    0.00898319, 0.00909249, 0.00926088, 0.00939487, 0.00955579, 0.00972593,
    0.00985441, 0.01006354, 0.0101857, 0.01042371, 0.01057837, 0.01078511,
    0.01100271, 0.01123792, 0.01142262, 0.01167902, 0.01192919, 0.0121649,
    0.01242224, 0.01271262, 0.01306938, 0.01326944, 0.01361994, 0.01389895,
    0.01424312, 0.01458958, 0.01483116, 0.01530667, 0.01562853, 0.01604331,
    0.01637709
]

counts_measured_per_floor = np.array([
    2715,
    2360,
    2381,
])

measurement_time_per_floor = np.array([
    80,
    83,
    107,
])

# cpm
counts_measured_normalized_per_floor = counts_measured_per_floor / \
    measurement_time_per_floor

fluxes_simulated_per_thickness_per_floor = np.array([
    fluxes_simulated_per_thickness_f6,
    fluxes_simulated_per_thickness_f4,
    fluxes_simulated_per_thickness_f1,
])

standard_errors_simulated_per_thickness_per_floor = np.array([
    standard_errors_simulated_per_thickness_f6,
    standard_errors_simulated_per_thickness_f4,
    standard_errors_simulated_per_thickness_f1,
])

assert(fluxes_simulated_per_thickness_per_floor.shape ==
       standard_errors_simulated_per_thickness_per_floor.shape)

functions_thickness_to_flux_simulated_per_floor = []
functions_thickness_to_se68_simulated_per_floor = []
for fluxes_simulated_per_thickness, standard_errors_simulated_per_thickness in zip(fluxes_simulated_per_thickness_per_floor, standard_errors_simulated_per_thickness_per_floor):

    # flux
    coefficients = np.polyfit(thicknesses, fluxes_simulated_per_thickness, 7)
    function_thickness_to_flux_simulated = np.poly1d(coefficients)
    functions_thickness_to_flux_simulated_per_floor.append(
        function_thickness_to_flux_simulated)

    # se68
    coefficients = np.polyfit(
        thicknesses, standard_errors_simulated_per_thickness, 5)
    function_thickness_to_se68_simulated = np.poly1d(coefficients)
    functions_thickness_to_se68_simulated_per_floor.append(
        function_thickness_to_se68_simulated)

    fig = plt.figure()
    axe = fig.add_subplot(111)
    axe.scatter(thicknesses, fluxes_simulated_per_thickness,
                label="Simulated Data")

    xx = np.linspace(min(thicknesses), max(thicknesses), 100)
    axe.plot(xx, function_thickness_to_flux_simulated(xx),
             color='red', label="Approximated Function")

    axe.set_xlabel('Thickness [cm]')
    axe.set_ylabel('Flux')
    axe.grid()
    axe.legend()
    plt.show()


def se68(count):
    return 1 / np.math.sqrt(count)


def se_propagation_on_mul(se_rel1, se_rel2):
    return np.math.sqrt(se_rel1**2 + se_rel2**2)


def count_estimated_f7(thickness):
    return counts_measured_normalized_per_floor[0] / functions_thickness_to_flux_simulated_per_floor[0](thickness)


def se68_estimated_f7(thickness):
    return se68(counts_measured_per_floor[0]) / functions_thickness_to_flux_simulated_per_floor[0](thickness)


def fluxes_measured_on_count_estimated_f7_per_floor(thickness):
    return counts_measured_normalized_per_floor / count_estimated_f7(thickness)


def se68_measured_on_count_estimated_f7_per_floor(thickness):
    return np.vectorize(se_propagation_on_mul)(np.vectorize(se68)(counts_measured_per_floor), se68_estimated_f7(thickness))


def se95_measured_on_count_estimated_f7_per_floor(thickness):
    return 2 * se68_measured_on_count_estimated_f7_per_floor(thickness)


def function_thickness_to_rmse(thickness):
    errors_per_floor = np.array([
        func(thickness) - flux for func, flux in zip(functions_thickness_to_flux_simulated_per_floor, fluxes_measured_on_count_estimated_f7_per_floor(thickness))
    ])

    squared_errors_per_floor = errors_per_floor ** 2
    root_mean_squared_error = np.math.sqrt(np.sum(squared_errors_per_floor))
    return root_mean_squared_error


# assumes -1 <= error_factor <= 1
def function_thickness_to_rmse_with_error(thickness, error_factors):
    errors_per_floor = np.array([
        func(thickness) - flux * (1 + se * err_factor) for func, flux, se, err_factor in zip(functions_thickness_to_flux_simulated_per_floor, fluxes_measured_on_count_estimated_f7_per_floor(thickness), se95_measured_on_count_estimated_f7_per_floor(thickness), error_factors)
    ])

    squared_errors_per_floor = errors_per_floor ** 2
    root_mean_squared_error = np.math.sqrt(np.sum(squared_errors_per_floor))
    return root_mean_squared_error


result_minimize = opt.differential_evolution(
    function_thickness_to_rmse, bounds=[(0, 100)])
[thickness_at_minimum_rmse] = result_minimize.x


def error_factors_to_thickness_plus(error_factors):
    result_minimize = opt.minimize(
        function_thickness_to_rmse_with_error, [15.0], args=error_factors, bounds=[(0, np.inf)])
    return result_minimize.x[0]


def error_factors_to_thickness_minus(error_factors):
    result_minimize = opt.minimize(
        function_thickness_to_rmse_with_error, [15.0], args=error_factors, bounds=[(0, np.inf)])
    return -result_minimize.x[0]


result_minimize = opt.differential_evolution(
    error_factors_to_thickness_plus, bounds=[(-1, 1), (-1, 1), (-1, 1)])
thinnest_95 = result_minimize.fun
error_factors_at_thinnest_95 = result_minimize.x

result_minimize = opt.differential_evolution(
    error_factors_to_thickness_minus, bounds=[(-1, 1), (-1, 1), (-1, 1)])
thickest_95 = -result_minimize.fun
error_factors_at_thickest_95 = result_minimize.x


print('thinnest 95%: ', thinnest_95)
print('thickness at minimum rmse: ', thickness_at_minimum_rmse)
print('thickest 95%: ', thickest_95)
print('minimum rmse: ', function_thickness_to_rmse(thickness_at_minimum_rmse))

xx = np.linspace(min(thicknesses), max(thicknesses), 100)
fig = plt.figure()
axe = fig.add_subplot(111)
axe.plot(xx, np.vectorize(lambda x: function_thickness_to_rmse_with_error(x, error_factors_at_thinnest_95))(xx),
         color='red', label="$+2\sigma_{thickness}$")
axe.plot(xx, np.vectorize(function_thickness_to_rmse)(xx),
         color='green', label="0")
axe.plot(xx, np.vectorize(lambda x: function_thickness_to_rmse_with_error(x, error_factors_at_thickest_95))(xx),
         color='blue', label="$-2\sigma_{thickness}$")

axe.plot(thinnest_95, function_thickness_to_rmse_with_error(
    thinnest_95, error_factors_at_thinnest_95), color='black', marker='+')
axe.plot(thickness_at_minimum_rmse,
         function_thickness_to_rmse(thickness_at_minimum_rmse), color='black', marker='+')
axe.plot(thickest_95, function_thickness_to_rmse_with_error(
    thickest_95, error_factors_at_thickest_95), color='black', marker='+')

axe.set_xlabel("Thickness [cm]")
axe.set_ylabel("RMSE")

plt.grid()
plt.legend()
plt.show()

floors = [6, 4, 1]

fig = plt.figure()
axe = fig.add_subplot(111)
# simulated data
axe.scatter(floors, [func(thickness_at_minimum_rmse)
                     for func in functions_thickness_to_flux_simulated_per_floor], label='Simulated Data', color='red', marker='x')

# error bar
for floor, function_thickness_to_flux_simulated, func_se68 in zip(floors, functions_thickness_to_flux_simulated_per_floor, functions_thickness_to_se68_simulated_per_floor):
    axe.plot([floor, floor], [function_thickness_to_flux_simulated(
        thickest_95) * (1 - 2*func_se68(thickest_95)), function_thickness_to_flux_simulated(thinnest_95) * (1 + 2*func_se68(thinnest_95))], color='red')


# measured data
axe.scatter(floors, fluxes_measured_on_count_estimated_f7_per_floor(
    thickness_at_minimum_rmse), label='Measured Data', color='blue', marker='.')

# error bar
for floor, flux in zip(floors, fluxes_measured_on_count_estimated_f7_per_floor(thickness_at_minimum_rmse)):
    se95 = se95_measured_on_count_estimated_f7_per_floor(
        thickness_at_minimum_rmse)
    axe.plot([floor, floor], [(1 + se95) * flux,
                              (1 - se95) * flux], color='blue')

axe.set_xlabel('Floor')
axe.set_ylabel('Flux')

plt.grid()
plt.legend()
plt.show()
