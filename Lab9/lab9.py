import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
np.random.seed(1771)


# a)

def generare_serie(N):
    timp = np.linspace(0, 1, N)
    trend = 23 * timp * timp + 15 * timp + 10
    sezon = np.sin(2 * np.pi * f1 * timp) + np.sin(2 * np.pi * f2 * timp)
    variatii_mici = np.random.normal(size=N)
    serie = trend + sezon + variatii_mici
    return timp, trend, sezon, variatii_mici, serie


N = 3000
f1 = 21
f2 = 11

time, trend, sezon, variatii_mici, serie = generare_serie(N)

fig, axes = plt.subplots(4, 1, figsize=(8, 8))
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
def train_ma(train, train_ep, q_, med):
    num_lines = len(train) - q_
    Y = np.zeros((num_lines, q_))
    x = np.zeros(num_lines)

    for line in range(num_lines):
        for column in range(q_):
            Y[line, column] = train_ep[q_ + line - 1 - column]

        x[line] = train[q_ + line] - train_ep[q_ + line] - med
    # x=Y*a
    return np.matmul(np.linalg.pinv(Y), x)


def predict_ma(ma_, med, test_ep):
    q_ = len(ma_)
    res = []
    for x in range(q_, len(test_ep)):
        last = test_ep[x - q_:x]
        value = np.convolve(last, ma_, mode="valid") + med + test_ep[x]
        res.append(value)
    return np.array(res)


ep = np.random.normal(size=N)

set_size = 1500
q = 100
train_set = serie[:set_size]
ep_train = ep[:set_size]
ep_test = ep[set_size - q:]
medie = np.mean(train_set)
test_set = serie[set_size:]

ma = train_ma(train_set, ep_train, q, medie)
predict = predict_ma(ma, medie, ep_test)

plt.title(f"MA")
plt.plot(time[set_size:], test_set, label="original")
plt.plot(time[set_size:], predict, label="predictie")
plt.legend()
plt.tight_layout()
plt.savefig("grafice/ex1c.pdf", format="pdf")
plt.savefig("grafice/ex1c.png", format="png")
plt.show()


# d)
def train_arma(train, train_ep, p, q):
    num_lines = len(train) - max(p, q)
    Y_ar = np.zeros((num_lines, p))
    Y_ma = np.zeros((num_lines, q))
    x = np.zeros(num_lines)

    for line in range(num_lines):
        # AR part
        for col in range(p):
            Y_ar[line, col] = train[p + line - 1 - col]

        # MA part
        for col in range(q):
            Y_ma[line, col] = train_ep[q + line - 1 - col]

        x[line] = train[p + line] - train_ep[q + line]

    # Concatenate AR and MA matrices
    Y = np.concatenate((Y_ar, Y_ma), axis=1)

    # x = Y * a
    return np.matmul(np.linalg.pinv(Y), x)


def predict_arma(arma, p, q, test_ep, date_trecut):
    res = []
    for x in range(max(p, q), len(test_ep)):
        last_ar = date_trecut[-p:]
        last_ma = test_ep[x - q:x]

        fereastra = np.concatenate((last_ma, last_ar))
        val = np.convolve(fereastra, arma, mode="valid") + test_ep[x]

        res.append(val)
        date_trecut = np.append(date_trecut, val)

    return np.array(res)


p_arma = [1, 2, 5, 11, 40, 50, 100, 150, 200]
q_arma = [1, 2, 5, 11, 40, 50, 100, 150, 200]
set_size_arma = 2000
best_p = best_q = 0
mse_best = np.inf

ep_arma = np.random.normal(size=N)
train_set_arma = serie[:set_size_arma]
ep_train_arma = ep_arma[:set_size_arma]
medie_arma = np.mean(train_set_arma)

test_set_arma = serie[set_size_arma:]
timp = time[set_size_arma:]

for pa in p_arma:
    for qa in q_arma:
        ep_test_arma = ep_arma[set_size_arma - max(pa, qa):]
        arma_params = train_arma(train_set_arma, ep_train_arma, pa, qa)
        arma_predictions = predict_arma(arma_params, pa, qa, ep_test_arma, train_set_arma[-pa:])

        mse = np.mean((test_set_arma - arma_predictions) ** 2)
        # print(f"mse = {mse} p={pa} q={qa}")
        if mse_best > mse:
            mse_best = mse
            best_q = qa
            best_p = pa

print(f"mse = {mse_best} p={best_p} q={best_q}")
ep_test_arma = ep_arma[set_size_arma - max(best_p, best_q):]

arma_params = train_arma(train_set_arma, ep_train_arma, best_p, best_q)
arma_predictions = predict_arma(arma_params, best_p, best_q, ep_test_arma, train_set_arma[-best_p:])

mse = np.mean((test_set_arma - arma_predictions) ** 2)
plt.title(f"ARMA predictie MSE = {mse}")
plt.plot(timp, test_set_arma, label="original")
plt.plot(timp, arma_predictions, label="ARMA prediction")
plt.legend()
plt.savefig("grafice/ex1d_1.pdf", format="pdf")
plt.savefig("grafice/ex1d_1.png", format="png")
plt.show()

# d)
from statsmodels.tsa.arima.model import ARIMA

set_size = 1500
p_max = q_max = 20
model = ARIMA(
    serie[:set_size],
    order=([p for p in range(1, p_max + 1)], 0, [q for q in range(1, q_max + 1)]),
    trend="t"
)

result = model.fit(method_kwargs={'maxiter': 30})
pred = result.forecast(steps=len(serie[set_size:]))

mse = np.mean((serie[set_size:] - pred) ** 2)
plt.title(f"Arma MSE = {mse}")
plt.plot(time[set_size:], serie[set_size:], label="original")
plt.plot(time[set_size:], pred, label="prezis")
plt.tight_layout()
plt.legend()
plt.savefig("grafice/ex1d_2.pdf", format="pdf")
plt.savefig("grafice/ex1d_2.png", format="png")
plt.show()
