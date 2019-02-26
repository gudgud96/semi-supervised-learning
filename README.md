## Cluster-then label approach on Iris dataset
Updated: 26.2.2019

### What is cluster-then-label?
Clustering is done before-hand in an unsupervised manner.
Unlabelled data points are labelled using the labelled data in the cluster.

### Results
Below shows the result for iris-40 (40% labelled data) using cluster-then-label approach.
Results are run on 10-validation fold of the dataset.

| No. | Label accuracy | Test accuracy |
| :---:| :---: | :---: |
| 1 | 0.9185   | 0.8667 |
| 2 | 0.9481   | 0.8667 |
| 3 | 0.9185   | 0.9333 |
| 4 | 0.9259   | 0.8 |
| 5 | 0.9259   | 0.8667 |
| 6 | 0.9407   | 0.9333 |
| 7 | 0.9111   | 1.0 |
| 8 | 0.9111   | 0.8667 |
| 9 | 0.9111   | 0.8 |
| 10 | 0.9185   | 1.0 |
|<b>Average| 0.9229 | 0.8933 </b>|

### To-do
[ ] Run on iris-10, iris-20 and iris-30<br>
[ ] Use other clustering algorithms (hierachical agglomerative, etc.)<br>
[ ] Any ideas to assign labels after clustering?