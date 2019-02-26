## Cluster-then label approach on Iris dataset
Updated: 26.2.2019

### What is cluster-then-label?
Clustering is done before-hand in an unsupervised manner.
Unlabelled data points are labelled using the labelled data in the cluster.

For this experiment, K-means clustering is used as our clustering algorithm.
For the labelling part, we simply use first label encountered in the cluster.

### Results
Results are run on 10-validation fold of the dataset.
Below shows the result for iris-40 and iris-30..

| No. | Label acc. (iris-40) | Test accuracy (iris-40) | Label acc. (iris-30) | Test accuracy (iris-30) 
| :---:| :---: | :---: | :---: | :---: |
| 1 | 0.9185   | 0.8667 | 0.9037 | 0.8667 |
| 2 | 0.9481   | 0.8667 | 0.9333 | 0.8667 |
| 3 | 0.9185   | 0.9333 | 0.9111 | 0.9333 |
| 4 | 0.9259   | 0.8 | 0.9259 | 0.8 |
| 5 | 0.9259   | 0.8667 | 0.9158 | 0.8667 |
| 6 | 0.9407   | 0.9333 | 0.9407 | 0.9333 |
| 7 | 0.9111   | 1.0 | 0.9037 | 1.0 |
| 8 | 0.9111   | 0.8667 | 0.9111 | 0.8667 |
| 9 | 0.9111   | 0.8 | 0.9037 | 0.8 |
| 10 | 0.9185   | 1.0 | 0.9037 | 1.0 |
|<b>Average</b>| <b>0.9229</b> | <b>0.8933 </b>|<b>0.9153 </b>|<b>0.8933 </b>|

Below are the results for iris-20 and iris-10.

| No. | Label acc. (iris-20) | Test accuracy (iris-20) | Label acc. (iris-10) | Test accuracy (iris-10) 
| :---:| :---: | :---: | :---: | :---: |
| 1 | 0.9037   | 0.8667 | 0.9037 | 0.8667 |
| 2 | 0.9333   | 0.8667 | 0.9185 | 0.8667 |
| 3 | 0.9037   | 0.9333 | 0.8963 | 0.9333 |
| 4 | 0.9259   | 0.8 | 0.9111 | 0.8 |
| 5 | 0.9037   | 0.8667 | 0.8963 | 0.8667 |
| 6 | 0.9333   | 0.9333 | 0.9185 | 0.9333 |
| 7 | 0.8888   | 1.0 | 0.8815 | 1.0 |
| 8 | 0.9111   | 0.8667 | 0.9037 | 0.8667 |
| 9 | 0.9037   | 0.8 | 0.9037 | 0.8 |
| 10 | 0.8963   | 1.0 | 0.8888 | 1.0 |
|<b>Average</b>| <b>0.9104</b> | <b>0.8933 </b>|<b>0.9022 </b>|<b>0.8933 </b>|

### Discussion
* Only a 2% drop margin is observed in accuracy when labelled examples
drop from 40% to 10%.
* **Test accuracy is not dropped, regardless the number of labelled examples.** 
* Iris dataset is observed to be well clustered - 
    * one of its class is clearly linear separable, and 
    * while the remaining 2 classes are not linearly separable, they can be easily clustered 
    with only a few false positives around the decision boundary.
* Hence, the clusters are almost of the same regardless of the number of labelled data.
* Hypothesis: **The outcome of using cluster-then-label is highly dependent on the data distribution, as if the data itself is well-clustered or not.**
* We can observe result of cluster-then-label using other not-so-well-clustered datasets / other clustering algorithms.

### To-do
[/] Run on iris-10, iris-20 and iris-30<br>
[ ] Use other clustering algorithms (hierachical agglomerative, etc.)<br>
[ ] Any ideas on assigning labels after clustering?