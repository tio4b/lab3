import os
import psutil
import numpy as np, pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

lr_types = {
    'exp': lambda epoch, lr0, decay: lr0 * np.exp(-decay * epoch),
    'step': lambda epoch, lr0, decay: lr0 * 0.5 ** (epoch // 10),
    'const': lambda epoch, lr0, decay: lr0
}


def calc_gradient(X, y, w, reg_type=None, alpha=0.1, l1_ratio=0.5):
    grad = X.T @ (X @ w - y) / X.shape[0]

    w_reg = w.copy()
    w_reg[0] = 0.0

    if reg_type == 'l1':
        grad += alpha * np.sign(w_reg)
    elif reg_type == 'l2':
        grad += 2 * alpha * w_reg
    elif reg_type == 'elasticnet':
        grad += alpha * (l1_ratio * np.sign(w_reg) +
                         (1 - l1_ratio) * 2 * w_reg)
    return grad


def calc_loss(X, y, w, reg_type=None, alpha=0.1, l1_ratio=0.5):
    mse = np.mean((X @ w - y) ** 2)

    w_reg = w.copy()
    w_reg[0] = 0.0

    if reg_type == 'l1':
        reg = alpha * np.sum(np.abs(w_reg))
    elif reg_type == 'l2':
        reg = alpha * np.sum(w_reg ** 2)
    elif reg_type == 'elasticnet':
        reg = alpha * (l1_ratio * np.sum(np.abs(w_reg)) +
                       (1 - l1_ratio) * np.sum(w_reg ** 2))
    else:
        reg = 0.0
    return mse + reg


def sgd(
        X, y,
        lr_fun,
        lr0=0.01,
        n_epochs=800,
        batch=1,
        decay=0.003,
        momentum=0.9,
        reg_type='elasticnet',
        tol=1e-4,
        patience=20,
        log_every=1
):
    m, n = X.shape
    w = np.zeros((n, 1))
    v = np.zeros_like(w)

    loss_hist, iter_hist = [], []
    best_loss, wait, it = np.inf, 0, 0

    for epoch in range(1, n_epochs + 1):
        lr = lr_fun(epoch - 1, lr0, decay)
        perm = np.random.permutation(m)

        for i in range(0, m, batch):
            xi = X[perm[i:i + batch]]
            yi = y[perm[i:i + batch]]

            g = calc_gradient(xi, yi, w, reg_type)
            v = momentum * v - lr * g
            w += v
            it += 1

            if it % log_every == 0:
                cur_loss = calc_loss(X, y, w, reg_type)
                loss_hist.append(cur_loss)
                iter_hist.append(it)

                if best_loss - cur_loss > tol:
                    best_loss = cur_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        return w, iter_hist, loss_hist
    return w, iter_hist, loss_hist


def load_data(fp: str
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(fp, sep=';')
    X = df.drop('quality', axis=1).values
    y = df['quality'].to_numpy().reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_tr = np.c_[np.ones((X_tr.shape[0], 1)), X_tr]
    X_te = np.c_[np.ones((X_te.shape[0], 1)), X_te]
    return X_tr, X_te, y_tr, y_te


def train(lr_name, lr_fun):
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    Xtr, Xte, ytr, yte = load_data('dataset.csv')
    w, its, loss = sgd(
        Xtr, ytr, lr_fun,
        lr0=0.01, n_epochs=800, batch=100,
        decay=0.003, momentum=0.3, reg_type='l2',
        tol=1e-4, patience=20
    )

    mem_after = proc.memory_info().rss
    delta_mb = (mem_after - mem_before) / 1024 ** 2
    total_iters = its[-1]

    y_pred = Xte @ w
    mse = np.mean((yte - y_pred) ** 2)
    mae = np.mean(np.abs(yte - y_pred))
    r2 = 1 - np.sum((yte - y_pred) ** 2) / np.sum((yte - yte.mean()) ** 2)

    print(f"{lr_name.upper():>5}  "
          f"MSE={mse:.3f}  MAE={mae:.3f}  R²={r2:.3f}  "
          f"iters={total_iters:,}  ΔRAM={delta_mb:.1f} MiB")
    print(f"{lr_name.upper():>5}  "
          f"{total_iters:,} & {delta_mb:.1f} & {mse:.3f} & {mae:.3f} & {r2:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.22)
    current = {'view': 'loss'}

    def draw_loss():
        ax.clear()
        ax.set_aspect('auto', adjustable='box')
        ax.plot(its, loss)
        ax.set_xlabel('iter');
        ax.set_ylabel('loss')
        ax.set_title('Loss vs iteration')
        ax.grid(True)
        ax.set_xlim(0, its[-1])
        ax.relim();
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def draw_scatter():
        ax.clear()
        ax.scatter(yte, y_pred, alpha=0.6)
        lo, hi = 3, 8
        ax.plot([lo, hi], [lo, hi], 'r--')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lo - .2, hi + .2)
        ax.set_ylim(lo - .2, hi + .2)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f"True vs Pred  (MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.2f})")
        ax.grid(True)
        fig.canvas.draw_idle()

    btn_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(btn_ax, 'switch')

    def toggle(event):
        if current['view'] == 'loss':
            draw_scatter();
            current['view'] = 'scatter'
        else:
            draw_loss();
            current['view'] = 'loss'

    btn.on_clicked(toggle)
    draw_loss()
    plt.show()


if __name__ == '__main__':
    for name, fun in lr_types.items():
        train(name, fun)
