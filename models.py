import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC


class ClusterThenLabeller:
    def __init__(self, X_data, Y_data, xlabel="X axis", ylabel="Y axis", fig_filename=""):
        self.X_data = X_data
        self.Y_data = Y_data
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig_filename = fig_filename
        self.clusterer = None
        self.classifier = None
        self.mode = "svm"

    def set_clusterer(self, cluster_num, model="kmeans"):
        if model == "kmeans":
            self.clusterer = KMeans(n_clusters=cluster_num, init='k-means++',
                                    max_iter=300, n_init=10, random_state=0)
        else:
            pass    # extend more cluster algorithm here

    def set_classifier(self, mode="svm"):
        if mode == "linearsvm":
            self.classifier = SVC(gamma="auto")
        else:
            self.classifier = LinearSVC()

    def cluster_then_label(self, mode="majority"):
        '''
        Main function for cluster-then-label.
        :param mode - either "majority" or "svm"
        :return:
            y_kmeans - Y-predicted label by model.
        '''

        # Experimenting on different number of clusters
        wcss = []
        for i in range(1, 11):
            clusterer = self.clusterer
            clusterer.fit(self.X_data)
            wcss.append(clusterer.inertia_)

        # Plotting the results onto a line graph, allowing us to observe 'The elbow'
        plt.plot(range(1, 11), wcss)
        plt.title('The elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')  # within cluster sum of squares
        plt.savefig(self.fig_filename + 'elbow.png')
        plt.close()

        # Creating the kmeans classifier
        y_kmeans = self.clusterer.fit_predict(self.X_data)

        print("y_kmeans.shape: ", y_kmeans.shape)
        print("After clustering: ", Counter(y_kmeans))

        # train a predictor, f, from all labelled data
        X_labelled_data = self.X_data[self.Y_data != 0]
        Y_labelled_data = self.Y_data[self.Y_data != 0]
        print("Labelled data: x - {}, y - {}".format(X_labelled_data.shape, Y_labelled_data.shape))
        labelled_clf = SVC(gamma="auto")
        # labelled_clf = LinearSVC()
        print("Fitting classifier on labelled data...")
        labelled_clf.fit(X_labelled_data, Y_labelled_data)

        # Assigning labels
        cluster_to_label_dict = {}
        n_clusters = len(Counter(y_kmeans))
        list_of_clfs = []  # a classifier for each cluster

        for i in range(n_clusters):  # labels in k_means result
            X_this_cluster = self.X_data[y_kmeans == i]
            Y_this_cluster = self.Y_data[y_kmeans == i]
            counter_dict = Counter(Y_this_cluster)
            print("Cluster " + str(i) + " labels", counter_dict)
            del counter_dict[0]  # unlabelled is not useful for us to determine the majority label

            if mode == "svm":
                # trained on labelled data in the cluster
                X_labelled = X_this_cluster[Y_this_cluster != 0]
                Y_labelled = Y_this_cluster[Y_this_cluster != 0]

                # if there is a significant amount of labelled data / has more than 1 class in the cluster,
                # train the specific cluster clf
                print("Percentage of labelled data in cluster: {}".format(len(X_labelled) / len(X_this_cluster)))
                if len(X_labelled) > 0 and len(counter_dict.keys()) > 1:
                    t1 = time.time()
                    cluster_clf = SVC(gamma="auto")
                    # cluster_clf = LinearSVC()
                    cluster_clf = cluster_clf.fit(X_labelled, Y_labelled)
                    print("Time taken for cluster clf: ", time.time() - t1)
                    list_of_clfs.append(cluster_clf)

                # or else, just use the universal clf trained on all labelled data
                else:
                    list_of_clfs.append(labelled_clf)

            # TODO: supervised learning algorithm here, currently only have majority vote
            if mode == "majority":
                majority_label = sorted(counter_dict, key=counter_dict.get, reverse=True)[0]
                print("Majority label in this cluster: " + str(majority_label))
                cluster_to_label_dict[i] = majority_label
                self.Y_data[(y_kmeans == i) & (self.Y_data == 0)] = majority_label
                print("Cluster to label dict: ", cluster_to_label_dict)

            else:
                y_indices_in_cluster = np.where((y_kmeans == i) & (self.Y_data == 0))[0]
                print(len(y_indices_in_cluster))
                for i in y_indices_in_cluster:
                    predicted_label = list_of_clfs[-1].predict([self.X_data[i]])
                    self.Y_data[i] = predicted_label

        y_kmeans = np.copy(self.Y_data)
        print('Final labelling: ', Counter(y_kmeans))

        # Visualising the clusters
        plt.figure(3, figsize=(8, 6))
        plt.clf()
        plt.scatter(self.X_data[y_kmeans == 1, 0], self.X_data[y_kmeans == 1, 1],
                    s=100, c='red', edgecolor='k', label='Iris-setosa')
        plt.scatter(self.X_data[y_kmeans == 2, 0], self.X_data[y_kmeans == 2, 1],
                    s=100, c='blue', edgecolor='k', label='Iris-versicolour')
        plt.scatter(self.X_data[y_kmeans == 3, 0], self.X_data[y_kmeans == 3, 1],
                    s=100, c='green', edgecolor='k', label='Iris-virginica')

        # Plotting the centroids of the clusters
        plt.scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1],
                    marker='x', s=169, linewidths=10,
                    color='black', zorder=10, label='Centroids')
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.fig_filename + 'clustered.png')
        plt.close()

        return y_kmeans

    def predict_on_test(self, y_predict, X_test):
        '''
        Prediction on test data
        :param y_predict: self-labelled Y-data after cluster_and_label is done.
        :param X_test: X_test data
        :return:
            predicted labels
        '''
        # train on self-labelled data
        t1 = time.time()
        clf = self.classifier
        print("Fitting self-labelled data...")
        clf.fit(self.X_data, y_predict)
        print("Used time: ", time.time() - t1)

        # run test data
        y_test_predict = clf.predict(X_test)
        return y_test_predict