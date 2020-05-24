# Program created by sunshawn
# date: 2020/5/24


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt


class ShArtArgRandomForestClassifier(RandomForestClassifier):

    def my_artifi(self, trees, X, y, Xtest=np.array([]), ytest=np.array([]), criterion='gini', max_features='sqrt'):
        # to ensure whether the test arrs is given
        try:
            print(Xtest[0])
        except IndexError:
            Xtest = X
            ytest = y

        # use accuracy now
        rightl = []
        for i in range(1, trees):
            self.n_estimators = i
            self.criterion = criterion
            self.max_features = max_features
            self.fit(X, y)
            predict = self.predict(Xtest)
            rights = (predict == ytest).mean()
            rightl.append(rights)
        plt.plot(np.array(range(1, trees)), np.array(rightl))
        return rightl.index(max(rightl)) + 1, max(rightl)


def main():
    myforest = ShArtArgRandomForestClassifier()
    x = np.array([[1, 3], [2, 8], [3, 5], [4, -1], [5, 5]])
    y = np.array([1, 0, 0, 1, 0])
    xp = np.array([[1, 3]])
    yp = np.array([1])
    k, right = myforest.my_artifi(3, x, y, xp, yp)
    print(k, right)


# test the code
if __name__ == '__main__':
    main()
