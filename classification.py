import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
y, images = digits['target'], digits['images']

n_samples = y.shape[0]

sample = np.random.randint(n_samples)
plt.imshow(images[sample])
plt.title('y = %i' % y[sample])
plt.show()

lista = [1, 0.70, 0.0001]

x = images.reshape((n_samples,-1))
xtrain, xtest,ytrain,ytest = train_test_split(x,y)

for i in lista:
    print("\nResults using Gamma in: ", str(i))

    model = svm.SVC(gamma=i)

    model.fit(xtrain, ytrain)

    print('Train :', model.score(xtrain,ytrain))
    print('Test  :', model.score(xtest, ytest ))

    y_pred = model.predict(xtest)

    print("Classification report: \n", metrics.classification_report(ytest, y_pred))
    print("Confussion Matrix: \n", metrics.confusion_matrix(ytest, y_pred))

    plt.close('all')
    sample = np.random.randint(xtest.shape[0])
    plt.imshow(xtest[sample].reshape((8,8)))
    plt.title('ypred = %i' % y_pred[sample])
    plt.show()
    
    input()

