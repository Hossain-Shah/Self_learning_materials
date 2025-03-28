from sklearn import datasets,decomposition,svm,metrics
import numpy as np
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
faces = datasets.fetch_olivetti_faces()
print(faces.data.shape)
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone)
X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, random_state=0)
print(X_train.shape, X_test.shape)
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)
plt.imshow(pca.mean_.reshape(faces.images[0].shape),cmap=plt.cm.bone)
plt.show()
print(pca.components_.shape)
fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape),cmap=plt.cm.bone)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
print(X_test_pca.shape)
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),cmap=plt.cm.bone)
    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(faces.target[y_pred],fontsize='small', color=color)
y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
clf = Pipeline([('pca', decomposition.PCA(n_components=150, whiten=True)),('svm', svm.LinearSVC(C=1.0))])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.confusion_matrix(y_pred, y_test))
