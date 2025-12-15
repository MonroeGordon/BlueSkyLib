# Models Module
Module containing various machine learning/AI models that can be created, trained, and deployed.

### Decision Tree
A model that uses a binary tree to make decisions after being trained on the dataset. A decision tree contains nodes that break down the dataset based on its features. When a decision tree is built during its training, nodes are split into child nodes based on a particular threshold value of a feature being used to decide whether to split the node or not. The leaf nodes of the decision tree are the final decision points where the tree predicts the class of the data. Decision trees can be used for classification or regression.

Details:
- **TreeNode**: Class for creating a tree node within a decision tree. A tree node contains feature data, the split threshold, its left and right children, and the value of the node.
- **DecisionTree**: Class for creating, training, and deploying a decision tree model. A decision tree is initially defined with a maximum depth that it can grow to, and the minimum number of data samples required per node split. The decision tree trains on a dataset by building a tree using a random selection of input features and using them to determine the information gain achieved when splitting a node. If the gain is high enough, a node will be split into its left and right child that each contain half of their parent node's input features. This process is continued until splitting a node does not create enough information gain, or the maximum tree depth is reached.

### Gaussian Mixture Model
A model that views the data as a mixture of different Gaussian distributions. This model allows for data clustering, density estimation, generative modeling, and anomaly detection. Gaussian mixture models calculate the probability of input data belonging to each Gaussian distribution, using log likelihoods. 

Details:
- **GMM**: Class for creating a Gaussian mixture model. Gaussian mixture models are initialized with the number of Gaussian components, the maximum number of training iterations, and a tolerance value that is used as stopping criteria for training. The Gaussian mixture model uses the expectation maximization algorithm for training on input data. First, the expected latent variables are calculated, and then they are used to calculate the maximization of log likelihoods of the input features. This is repeated for the specified number of iterations to train the model. When predicting, the expected latent value is calculated for each input feature and the feature with the highest expectation is the predicted output.

### Linear Discriminant Analysis
A model that allows for classification and dimensionality reduction using linear discriminant analysis. Linear discriminant analysis tries to find a linear combination of input features that allow for the data to be linearly separated into two or more classes.

Details:
- **LDA**: Class for creating a linear discriminant analysis model. 
