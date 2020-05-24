# Program created by sunshawn
# date: 2020/5/24


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt


class ArtificialArgumentKNN(KNeighborsClassifier):

    def my_artifi(self, kneighbor, X, y, Xtest=np.array([]), ytest=np.array([])):
		# to ensure whether the test arrays is given
        try:
            print(Xtest[0])
        except:
            Xtest = X
            ytest = y
		
		# use accuracy now
        rightl = []
        for i in range(1, kneighbor):
            self.n_neighbors = i
            self.fit(X, y)
            predict = self.predict(Xtest)
            rights = (predict == ytest).mean()
            rightl.append(rights)
        plt.plot(np.array(range(1, kneighbor)), np.array(rightl))
        return rightl.index(max(rightl)) + 1, max(rightl)


# test the code
if __name__ == '__main__':
    myknn = ArtificialArgumentKNN()
    x = np.array([[1, 3], [2, 8], [3, 5], [4, -1], [5, 5]])
    y = np.array([1, 0, 0, 1, 0])
    xp = np.array([[1, 3]])
    yp = np.array([1])
    k, right = myknn.my_artifi(3, x, y, xp, yp)
    print(k, right)
