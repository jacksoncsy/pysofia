import numpy as np
from pysofia import sofia_ml
import menpo.io as mio

n_samples, n_features = 500, 500
# X = np.random.randn(n_samples, n_features)
# w = np.random.randn(n_features)
# y = (X.dot(w) + np.random.randn(n_samples)).astype(np.int)
#
# mio.export_pickle([X, y], 'tmp.pkl', overwrite=True)

data = mio.import_pickle('tmp.pkl')
X = data[0]
y = data[1]

coef = sofia_ml.sgd_train(X, y, None, 0.01, model='roc', n_features=n_features)

prediction1 = sofia_ml.sgd_predict(X, coef[0])

prediction2 = sofia_ml.sgd_predict(X, coef[0])

std1 = np.std(prediction1)
mean1 = np.mean(prediction2)


print "done"