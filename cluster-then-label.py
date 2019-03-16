import os
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import plot_confusion_matrix

np.random.seed(10)


def prepare_data(filename='iris-ssl40/iris-ssl40-10-1trs.dat'):
    '''
    Data preprocessing function.
    :param filename: Filename for data files.
    :return:
        X_data, Y_data - X data and label Y data.
    '''
    # classes = ['unlabeled', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    classes = ['unlabeled', 'g', 'h']
    X = []
    Y = []
    for line in open(filename, 'r'):
        data = line.rstrip()
        if data[0] == '@':      # headers
            continue
        data = data.split(', ')
        data[-1] = classes.index(data[-1])
        X.append(data[:-1])
        Y.append(data[-1])

    X_data = np.array(X).astype(np.float)
    Y_data = np.array(Y).astype(np.int)
    return X_data, Y_data


def plot_scatter(X_data, Y_data, name='test.png'):
    '''
    Helper function for scatter plot.
    :param X_data: X data
    :param Y_data: Y data
    :param name: filename for saving plot
    :return: None, but a scatter plot is saved.
    '''
    X = X_data[:, :2]  # we only take the first two features.
    y = Y_data

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X_data[Y_data == 0, 0], X_data[Y_data == 0, 1],
                s=100, c='grey', edgecolor='k', label='unlabelled')
    plt.scatter(X_data[Y_data == 1, 0], X_data[Y_data == 1, 1],
                s=100, c='red', edgecolor='k', label='Iris-setosa')
    plt.scatter(X_data[Y_data == 2, 0], X_data[Y_data == 2, 1],
                s=100, c='blue', edgecolor='k', label='Iris-versicolour')
    plt.scatter(X_data[Y_data == 3, 0], X_data[Y_data == 3, 1],
                s=100, c='green', edgecolor='k', label='Iris-virginica')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.savefig(name)
    plt.close()


def plot_pca(X_data, Y_data):
    '''
    Helper function for PCA plot.
    :param X_data: X data
    :param Y_data: Y data
    :param name: filename for saving plot
    :return: None, but a PCA plot is saved.
    '''
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X_data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y_data,
               cmap=plt.cm.Accent, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()


def plot_boundaries(kmeans, X_data):
    '''
    To plot cluster boundaries.
    :param kmeans: clusterer, using K-means for now.
    :param X_data: X data.
    :return: None, but a figure is plotted showing cluster boundaries.
    '''
    kmeans.fit(X_data[:, :2])

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(X_data[:, 0], X_data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# TODO: Incorporate other clustering algorithms in this function
def cluster_then_label(X_data, Y_data, filename):
    '''
    Main function for cluster-then-label.
    :param X_data: X data.
    :param Y_data: Y label data.
    :return:
        kmeans - K-means clustering model.
        y_kmeans - Y-predicted value by model.
        label_dict - label dictionary, maps Y-predicted value to label value defined in data.
    '''

    # Experimenting on different number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_data)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.savefig(filename + 'elbow.png')
    plt.close()

    # Creating the kmeans classifier
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=600, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X_data)

    # Assigning labels - majority vote
    label_dict = {}
    n_clusters = len(Counter(y_kmeans))

    for i in range(n_clusters):  # labels in k_means result
        counter_dict = Counter(Y_data[y_kmeans == i])
        del counter_dict[0]     # unlabelled is not useful for us to determine the majority label
        print(counter_dict)
        majority_label = sorted(counter_dict, key=counter_dict.get, reverse=True)[0]
        label_dict[i] = majority_label
    print(label_dict)

    for i in range(len(y_kmeans)):
        if Y_data[i] == 0:
            y_kmeans[i] = label_dict[y_kmeans[i]]   # label the unlabelled
        else:
            y_kmeans[i] = Y_data[i]

    # Visualising the clusters
    plt.figure(3, figsize=(8, 6))
    plt.clf()
    plt.scatter(X_data[y_kmeans == 1, 0], X_data[y_kmeans == 1, 1],
                s=100, c='red', edgecolor='k', label='Iris-setosa')
    plt.scatter(X_data[y_kmeans == 2, 0], X_data[y_kmeans == 2, 1],
                s=100, c='blue', edgecolor='k', label='Iris-versicolour')
    plt.scatter(X_data[y_kmeans == 3, 0], X_data[y_kmeans == 3, 1],
                s=100, c='green', edgecolor='k', label='Iris-virginica')

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=169, linewidths=10,
                color='black', zorder=10, label='Centroids')
    plt.legend()
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.savefig(filename + 'clustered.png')
    plt.close()

    return kmeans, y_kmeans, label_dict


def main():
    # prepare data - unlabelled, labelled, test
    dataset = "magic"
    for j in [10]:
        for k in range(1, 11):
            filename = "temp-{}/".format(k)
            os.makedirs(filename)
            num = k
            percentage = j
            print("Percentage: {} Validation: {}".format(j, k))
            tra_filename = '{}-ssl{}/{}-ssl{}-10-{}tra.dat'.format(dataset, percentage, dataset, percentage, num)
            trs_filename = '{}-ssl{}/{}-ssl{}-10-{}trs.dat'.format(dataset, percentage, dataset, percentage, num)
            tst_filename = '{}-ssl{}/{}-ssl{}-10-{}tst.dat'.format(dataset, percentage, dataset, percentage, num)

            X_unlabelled, Y_unlabelled = prepare_data(tra_filename)
            plot_scatter(X_unlabelled, Y_unlabelled, name=filename + "unlabelled.png")

            X_labelled, Y_labelled = prepare_data(trs_filename)
            plot_scatter(X_labelled, Y_labelled, name=filename + "labelled.png")

            X_test, Y_test = prepare_data(tst_filename)
            print(X_test.shape, Y_test.shape)

            # plot_pca(X_labelled, Y_labelled)
            # break

            # use supervised learning for cluster-then-label
            model, y_predict, label_dict = cluster_then_label(X_labelled, Y_labelled, filename)

            # semi-supervised learning for cluster-then-label
            # model, y_predict, label_dict = cluster_then_label(X_unlabelled, Y_unlabelled, filename)
            # print("Accuracy for labelling: {}".format(accuracy_score(Y_labelled, y_predict)))
            #
            # # confusion matrix for labelling
            cnf_matrix = confusion_matrix(Y_labelled, y_predict)
            np.set_printoptions(precision=2)
            # class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            class_names = ['g', 'h']
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix')
            plt.savefig(filename + 'cm_label.png')
            plt.close()
            #
            # # run on test data
            y_test_predict = model.predict(X_test)
            for i in range(len(y_test_predict)):
                y_test_predict[i] = label_dict[y_test_predict[i]]
            plot_scatter(X_test, Y_test, name=filename + 'test_answer.png')
            plot_scatter(X_test, y_test_predict, name=filename + 'test_predict.png')
            print("Accuracy for testing: {}\n".format(accuracy_score(Y_test, y_test_predict)))
            #
            # # confusion matrix on test data
            cnf_matrix = confusion_matrix(Y_test, y_test_predict)
            np.set_printoptions(precision=2)
            # class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            class_names = ['g', 'h']
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix')
            plt.savefig(filename + 'cm_test.png')
            plt.close()


if __name__ == "__main__":
    main()