import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# a)
N = 1000
f1 = 21
f2 = 11

time = np.linspace(0, 1, N)
trend = 23 * time * time + 15 * time + 10
sezon = np.sin(2 * np.pi * f1 * time) + np.sin(2 * np.pi * f2 * time)
variatii_mici = np.random.normal(size=N)

serie = trend + sezon + variatii_mici


# fig, axes = plt.subplots(4, 1, figsize=(8, 10))
# axes[0].plot(time, trend)
# axes[0].set_title("Trend")
# axes[1].plot(time, sezon)
# axes[1].set_title("Sezon")
# axes[2].plot(time, variatii_mici)
# axes[2].set_title("Variatii mici")
# axes[3].plot(time, serie)
# axes[3].set_title("Serie")
# plt.tight_layout()
# plt.show()


# b)
def exp_med(series, alpha):
    s = np.zeros(len(series))
    s[0] = series[0]
    for t in range(1, len(series)):
        s[t] = alpha * series[t] + (1 - alpha) * s[t - 1]
    return s


alpha = 0.6
s = exp_med(serie, alpha)
subset = 300

plt.plot(time[:subset], serie[:subset], label="original")
plt.plot(time[:subset], s[:subset], label="mediere exponentiala")
plt.legend()
plt.show()

# print("Valoarea optimă pentru alpha:", optimal_alpha)

# c)
ep = np.random.normal(size=N)


