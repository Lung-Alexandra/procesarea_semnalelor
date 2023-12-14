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
plt.show()


# b)
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    n = max(result)
    return result[result.size // 2:] / n


autocorelation = autocorr(serie)

plt.plot(autocorelation)
plt.title("Autocorelation")
plt.show()


# c)
def train_model_AR(date_trecut, dimensiune_ar):
    p = dimensiune_ar
    Y = np.zeros((len(date_trecut) - p, p))
    for lin in range(len(date_trecut) - p):
        for col in range(p):
            Y[lin][col] = date_trecut[p + lin - col - 1]
    y = (date_trecut[p:]).T
    # print(y)
    # print(Y)
    cf = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Y.T, Y)), Y.T), y)
    return cf


def predict_AR(cf, date_trecut, dimensiune_ar):
    p = dimensiune_ar
    ultimele_p_val = np.flip(date_trecut[-p:])
    predictie = np.sum(cf * ultimele_p_val)
    return predictie


# c) 
p = 250  # dimensiunea modelului AR
m = int(0.9 * len(serie))
train_date_trecut = serie[:m]

coeficienti = train_model_AR(train_date_trecut, p)
predictii = []
for i in range(len(serie) - m):
    pred = predict_AR(coeficienti, train_date_trecut, p)
    predictii.append(pred)
    train_date_trecut = np.append(train_date_trecut, pred)

plt.plot(time[m:], serie[m:], label="original")
plt.plot(time[m:], predictii, label="predictie")
plt.legend()
plt.show()


# d)
def testmp(series, m, p):
    err = 0
    n = 0
    for i in range(0, len(serie) - m - 1, m):
        train = series[i:i + m]
        test = series[i + m]
        coef = train_model_AR(train, p)
        predict = predict_AR(coef, train, p)
        err += abs(test - predict)
        n += 1
    MAE = err / n
    return MAE


m = [200, 400, 700, 800, 900]
p = [50, 100, 300, 500, 700]
best_error = []
for i in m:
    for j in p:
        if i >= j:
            eroare = testmp(serie, i, j)
            best_error.append((eroare, i, j))

best_error = sorted(best_error, key=lambda x: x[0])
print(f"m = {best_error[0][1]}, p = {best_error[0][2]}")
print(best_error)
