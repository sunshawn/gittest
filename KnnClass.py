# Program created by sunshawn
# date: 2020/5/24


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt


class SHArtArgKNNClassifier(KNeighborsClassifier):
    """
    This class extends the KNNClassifier
    This class can use the method 'my_artifi' to change the parameters artificially
    when the accuracy or precision or recall is the most it returns it.
    """

    def my_artifi(self, kneighbor, X, y, Xtest=np.array([]), ytest=np.array([])):
        """
        :param kneighbor: the k
        :param X: the independent var of training set
        :param y: the dependent var of training set
        :param Xtest: the independent var of test set
        :param ytest: the dependent var of test set
        :return: k and it's state when the state is utmost
        """

        # to ensure whether the test arrs is given
        try:
            print(Xtest[0])
        except IndexError:
            Xtest = X
            ytest = y

        # FixMe: use accuracy now
        rightl = []
        for i in range(1, kneighbor):
            self.n_neighbors = i
            self.fit(X, y)
            predict = self.predict(Xtest)
            rights = (predict == ytest).mean()  # the calculation of accuracy,
            # equals rights = (predict == ytest).sum() / len(predict),
            # that's because python can calculate True as 1 and False 0
            rightl.append(rights)
            # TODO: to show the procedure

        plt.plot(np.array(range(1, kneighbor)), np.array(rightl))
        return rightl.index(max(rightl)) + 1, max(rightl)


def main():
    myknn = SHArtArgKNNClassifier()
    x = np.array([[1, 3], [2, 8], [3, 5], [4, -1], [5, 5]])
    y = np.array([1, 0, 0, 1, 0])
    xp = np.array([[1, 3]])
    yp = np.array([1])
    k, right = myknn.my_artifi(3, x, y, xp, yp)
    print(k, right)


# test the code
if __name__ == '__main__':
    main()
