import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import decimal

np.set_printoptions(
        suppress=True)
# refs.
# https://www.nti-audio.com/ja/%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88/%E6%B8%AC%E5%AE%9A%E3%83%8E%E3%82%A6%E3%83%8F%E3%82%A6/%E3%82%B5%E3%82%A6%E3%83%B3%E3%83%89%E3%83%AC%E3%83%99%E3%83%AB%E6%B8%AC%E5%AE%9A%E3%81%AE%E5%91%A8%E6%B3%A2%E6%95%B0%E9%87%8D%E3%81%BF%E4%BB%98%E3%81%91%E7%89%B9%E6%80%A7%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
freqs, weights_a, weights_c, weights_z = np.array([6.3, -85.4, -21.3, 0.0,
8, -77.8, -17.7, 0.0,
10, -70.4, -14.3, 0.0,
12.5, -63.4, -11.2, 0.0,
16, -56.7, -8.5, 0.0,
20, -50.5, -6.2, 0.0,
25, -44.7, -4.4, 0.0,
31.5, -39.4, -3.0, 0.0,
40, -34.6, -2.0, 0.0,
50, -30.2, -1.3, 0.0,
63, -26.2, -0.8, 0.0,
80, -22.5, -0.5, 0.0,
100, -19.1, -0.3, 0.0,
125, -16.1, -0.2, 0.0,
160, -13.4, -0.1, 0.0,
200, -10.9, 0.0, 0.0,
250, -8.6, 0.0, 0.0,
315, -6.6, 0.0, 0.0,
400, -4.8, 0.0, 0.0,
500, -3.2, 0.0, 0.0,
630, -1.9, 0.0, 0.0,
800, -0.8, 0.0, 0.0,
1000, 0, 0, 0,
1250, 0.6, 0.0, 0.0,
1600, 1.0, -0.1, 0.0,
2000, 1.2, -0.2, 0.0,
2500, 1.3, -0.3, 0.0,
3150, 1.2, -0.5, 0.0,
4000, 1.0, -0.8, 0.0,
5000, 0.5, -1.3, 0.0,
6300, -0.1, -2.0, 0.0,
8000, -1.1, -3.0, 0.0,
10000, -2.5, -4.4, 0.0,
12500, -4.3, -6.2, 0.0,
16000, -6.6, -8.5, 0.0,
20000, -9.3, -11.2, 0.00]).reshape(36, 4).T
print(freqs)

func_weight_a = interpolate.CubicSpline(freqs, weights_a, extrapolate=True)
print(func_weight_a.c)
decimal.getcontext().prec = 10

fig = plt.figure()
axe = fig.add_subplot(111)
axe.scatter(freqs, weights_a)
axe.set_xscale('log')

freqs = np.logspace(0, 5, 1000, base=10)
axe.plot(freqs, func_weight_a(freqs))
plt.show()
