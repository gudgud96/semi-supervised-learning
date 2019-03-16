## Cluster-then label approach on Iris dataset
Updated: 26.2.2019

### What is cluster-then-label?
Clustering is done before-hand in an unsupervised manner.
Unlabelled data points are labelled using the labelled data in the cluster.

For this experiment, K-means clustering is used as our clustering algorithm.
For the labelling part, we use majority vote to determine label of the cluster.

### Results
Results are run on 10-validation fold of the dataset.
Below shows the result for iris-40 and iris-30.

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

#### Magic dataset

| No. | Label acc. (magic-40) | Test accuracy (magic-40) | Label acc. (magic-30) | Test accuracy (magic-30) 
| :---:| :---: | :---: | :---: | :---: |
| 1 | 0.7879   | 0.6530 | 0.7555 | 0.6530 |
| 2 | 0.7886   | 0.6583 | 0.7536 | 0.6583 |
| 3 | 0.7891   | 0.6483 | 0.7555 | 0.6451 |
| 4 | 0.7888   | 0.6525 | 0.7560 | 0.6525 |
| 5 | 0.7918   | 0.6367 | 0.7580 | 0.6367 |
| 6 | 0.7891   | 0.6483 | 0.7559 | 0.6483 |
| 7 | 0.7899   | 0.6409 | 0.7559 | 0.6483 |
| 8 | 0.7891   | 0.6483 | 0.7545 | 0.6540 |
| 9 | 0.7881   | 0.6540 | 0.7542 | 0.6540 |
| 10 | 0.7902  | 0.6514 | 0.7569 | 0.6514 |
|<b>Average</b>| <b>0.7893</b> | <b>0.6492 </b>|<b>0.7556 </b>|<b>0.6502 </b>|

Below are the results for iris-20 and iris-10.

| No. | Label acc. (iris-20) | Test accuracy (iris-20) | Label acc. (iris-10) | Test accuracy (iris-10) 
| :---:| :---: | :---: | :---: | :---: |
| 1 | 0.7210   | 0.6530 | 0.6848 | 0.6530 |
| 2 | 0.7191   | 0.6583 | 0.6842 | 0.6583 |
| 3 | 0.7199   | 0.6451 | 0.6844 | 0.6451 |
| 4 | 0.7216   | 0.6525 | 0.6862 | 0.6525 |
| 5 | 0.7240   | 0.6367 | 0.6880 | 0.6367 |
| 6 | 0.7214   | 0.6483 | 0.6870 | 0.6451 |
| 7 | 0.7227   | 0.6409 | 0.6870 | 0.6409 |
| 8 | 0.7192   | 0.6540 | 0.6839 | 0.6540 |
| 9 | 0.7190   | 0.6540 | 0.6834 | 0.6540 |
| 10 | 0.7225  | 0.6514 | 0.6864 | 0.6514 |
|<b>Average</b>| <b>0.7210</b> | <b>0.6494 </b>|<b>0.6855 </b>|<b>0.6491 </b>|

### Discussion
* For iris, only a 2% drop margin is observed in accuracy when labelled examples
drop from 40% to 10%.
* For magic, there is a 10% drop margin in accuracy for labelling.
* **Test accuracy is not dropped, regardless the number of labelled examples.** 
* Iris dataset is observed to be well clustered - 
    * one of its class is clearly linear separable, and 
    * while the remaining 2 classes are not linearly separable, they can be easily clustered 
    with only a few false positives around the decision boundary.
* Hence, the clusters are almost of the same regardless of the number of labelled data.
* Hypothesis: **The outcome of using cluster-then-label is regardless of 
labelled data due to its unsupervised nature.
Ratherm it is highly dependent on the data distribution, as if the data itself is well-clustered or not.**
* We can observe result of cluster-then-label using other not-so-well-clustered datasets / other clustering algorithms.

### To-do
[ ] Use other clustering algorithms (hierachical agglomerative, etc.)
[ ] Use supervised classifier other than majority vote for classifiying in clusters.