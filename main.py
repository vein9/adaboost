from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  LogisticRegresstion import LogisticRegresstion as LR
from matplotlib.backends.backend_pdf import PdfPages



np.random.seed(2)
# csv
dataset = pd.read_csv('data_classification.csv', header=None)
row_count = len(dataset.values[0]) - 1
# print(row_count)
X = []
for row in range(row_count):
    # 0,1,2
    a = []
    for value in dataset.values:
        a.append(value[row])
    X.append(a)

X = np.array(X).T
y = [label[row_count] for label in dataset.values]
y = np.array(y)

# bias trick
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hist = logistic_regression(w_init, Xbar, y, lam, lr=0.05, nepoches=500)
print('Solution of Logistic Regression:', w)
print('Final loss:', loss(w, Xbar, y, lam))

filename = 'log_reg_loss.pdf'
with PdfPages(filename) as pdf:
    plt.plot(loss_hist)
    plt.xlabel('number of iterations', fontsize=13)
    plt.ylabel('loss function', fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)
    pdf.savefig(bbox_inches='tight')
    plt.show()

res = predict(w, np.array([6.5, 0.8, 1]))
print('ketqua:', res)