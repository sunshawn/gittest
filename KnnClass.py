# Program created by sunshawn
# date: 2020/6/1


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import some_functions


class SHArtArgKNNClassifier(KNeighborsClassifier):
    """
    This class extends the KNNClassifier
    This class can use the method 'my_artifi' to change the parameters artificially
    when the accuracy or precision or recall is the most it returns it.
    """

    def my_artifi(self, kneighbor, X, y, Xtest=np.array([]), ytest=np.array([]), standard='not_binary'):
        """
        :param kneighbor: the k
        :param X: the independent var of training set
        :param y: the dependent var of training set
        :param Xtest: the independent var of test set
        :param ytest: the dependent var of test set
        :param standard: the evaluation standard of the classifying.
        can use not_binary, accuracy, precision, recall & f1
        :return: k and it's state when the state is utmost
        """

        # to ensure whether the test arrs is given
        try:
            print(Xtest[0])
        except IndexError:
            Xtest = X
            ytest = y

        rightl = []
        for i in range(1, kneighbor):
            self.n_neighbors = i
            self.fit(X, y)
            predict = self.predict(Xtest)

            if standard == 'not_binary':  # the question is not a binary one
                rights = (predict == ytest).mean()  # the calculation of accuracy,
                # equals rights = (predict == ytest).sum() / len(predict),
                # that's because python can calculate True as 1 and False 0
                rightl.append(rights)
            else:  # the question is a binary one
                tn, fp, fn, tp = confusion_matrix(ytest, predict).flatten()  # get the confusion matrix to calculate
                #  the following data

                if standard == 'accuracy':
                    rightl.append(some_functions.accuracy(tp, tn, fp, fn))
                elif standard == 'precision':
                    rightl.append(some_functions.precision(tp, fp))
                elif standard == 'recall':
                    rightl.append(some_functions.recall(tp, fn))
                else:
                    rightl.append(some_functions.f1(tp, fp, tn))
            print(str(i / kneighbor * 100) + '%')

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
