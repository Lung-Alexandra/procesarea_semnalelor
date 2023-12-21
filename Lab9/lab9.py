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

fig, axes = plt.subplots(4, 1, figsize=(8, 10))
axes[0].plot(time, trend)
axes[0].set_title("Trend")
axes[1].plot(time, sezon)
axes[1].set_title("Sezon")
axes[2].plot(time, variatii_mici)
axes[2].set_title("Variatii mici")
axes[3].plot(time, serie)
axes[3].set_title("Serie")
plt.tight_layout()
plt.savefig("grafice/ex1a.pdf", format="pdf")
plt.savefig("grafice/ex1a.png", format="png")
plt.show()


# b)
def exp_med(series, alpha):
    s = np.zeros(len(series))
    s[0] = series[0]
    for t in range(1, len(series)):
        s[t] = alpha * series[t] + (1 - alpha) * s[t - 1]
    return s


def cost(serie, alpha):
    e = exp_med(serie, alpha)
    mse = 0
    cnt = 0
    for i in range(1, len(serie)):
        mse += ((serie[i] - e[i - 1]) ** 2)
        cnt += 1
    mse /= cnt
    return mse


alphas = np.linspace(0.1, 1, 50)
rez = []
for alp in alphas:
    err = cost(serie, alp)
    rez.append((err, alp))

errors, alphas = zip(*rez)

rez = sorted(rez, key=lambda x: x[0])
plt.title("Valoarea optima a lui alpha")
plt.plot(alphas, errors)
plt.stem(rez[0][1], rez[0][0])
plt.show()

print("Valoarea optima pentru alpha:", rez[0][1])

alpha = 0.6
s = exp_med(serie, alpha)
subset = 300
fig, axs = plt.subplots(2, figsize=(12, 12))
axs[0].plot(time[:subset], serie[:subset], label=f"original")
axs[0].plot(time[:subset], s[:subset], label=f"mediere exponentiala")
axs[0].set_title(f"Mediere exponentiala cu alpha fixat = {alpha}")
axs[0].legend()
alpha = rez[0][1]
s = exp_med(serie, alpha)
subset = 300
axs[1].plot(time[:subset], serie[:subset], label=f"original")
axs[1].plot(time[:subset], s[:subset], label=f"mediere exponentiala")
axs[1].set_title(f"Mediere exponentiala cu alpha = {alpha}")
axs[1].legend()

fig.tight_layout()
plt.savefig("grafice/ex1b.pdf", format="pdf")
plt.savefig("grafice/ex1b.png", format="png")
plt.show()


# c)
def train_ma(train, train_ep, q, med):
    num_lines = len(train) - q
    Y = np.zeros((num_lines, q))
    x = np.zeros(num_lines)

    for line in range(num_lines):
        for column in range(q):
            Y[line, column] = train_ep[q + line - 1 - column] + train_ep[q + line] + med

        x[line] = train[q + line]
    # x=Y*a
    return np.matmul(np.linalg.pinv(Y), x)


def predict_ma(ma, med, test_ep):
    q_ = len(ma)
    res = []
    for x in range(q_, len(test_ep)):
        last = test_ep[x - q_:x]
        value = np.convolve(last, ma, mode="valid") + med + test_ep[x]
        res.append(value)
    return np.array(res)


ep = np.random.normal(size=N)

set_size = 700
train_set = serie[:set_size]
ep_train = ep[:set_size]
ep_test = ep[set_size:]
medie = np.mean(train_set)
q = 100
test_set = serie[set_size + q:]

ma = train_ma(train_set, ep_train, q, medie)
predict = predict_ma(ma, medie, ep_test)

mse = np.mean((serie[set_size:] - predict) ** 2)
plt.title(f"MA predictie MSE = {mse}")
plt.plot(time[set_size + q:], test_set, label="original")
plt.plot(time[set_size + q:], predict, label="predictie")
plt.legend()
plt.tight_layout()
plt.savefig("grafice/ex1c.pdf", format="pdf")
plt.savefig("grafice/ex1c.png", format="png")
plt.show()

# d)
from statsmodels.tsa.arima.model import ARIMA

set_size = 800
p_max = q_max = 20
model = ARIMA(
    serie[:set_size],
    order=([p for p in range(1, p_max + 1)], 0, [q for q in range(1, q_max + 1)]),
    trend="t"
)

result = model.fit(method_kwargs={'maxiter': 400})
pred = result.forecast(steps=len(serie[set_size:]))

mse = np.mean((serie[set_size:] - pred) ** 2)
plt.title(f"Arma MSE = {mse}")
plt.plot(time[set_size:], serie[set_size:], label="original")
plt.plot(time[set_size:], pred, label="prezis")
plt.tight_layout()
plt.legend()
plt.savefig("grafice/ex1d.pdf", format="pdf")
plt.savefig("grafice/ex1d.png", format="png")
plt.show()
