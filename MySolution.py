import cvxpy as cp
import numpy as np
from collections import Counter
from sklearn.svm import SVC 
from sklearn.metrics import normalized_mutual_info_score, accuracy_score


### TODO: import any other packages you need for your solution


#--- Task 1 ---#

class MyClassifier_OvA:  
    ''' !!! DEPRECATED: This class is not used in the final solution !!! '''
    def __init__(self, num_class: int, num_feature: int):
        self.num_class = num_class  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        self.w = np.ndarray(shape=(num_class, num_feature), dtype=np.float32)
        self.b = np.ndarray(shape=(num_class, 1), dtype=np.float32)

    
    def train(self, trainX, trainY, C=1.0):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        batch_size, feature_dim = trainX.shape
        num_classes = len(np.unique(trainY))
        
        # Variables for each class
        slack = []
        problems = []
        
        for i_class in range(num_classes):
            y_k = np.where(trainY == i_class, 1, -1)
            
            # Variables
            w_k = cp.Variable(feature_dim)
            b_k = cp.Variable()
            eps_k = cp.Variable(batch_size)
            
            # Constraints
            constraints = [y_k[i] * (trainX[i] @ w_k + b_k) >= 1 - eps_k[i] for i in range(batch_size)]
            constraints += [eps_k >= 0]
            
            # Objective
            objective = cp.Minimize(cp.norm(w_k, 1) + C * cp.sum(eps_k))
            
            # Problem
            problem = cp.Problem(objective, constraints)
            problems.append(problem)
            
            # Solve problem
            problem.solve()
            
            # Record weights and biases
            self.w[i_class] = w_k.value
            self.b[i_class] = b_k.value
            slack.append(eps_k)
            
            print(f"Class {i_class}:")
            print("Optimal weights (w):", self.w[i_class])
            print("Optimal bias (b):", self.b[i_class])
            
        return self.w, self.b

    
    def predict(self, testX, testY_range=[0, 1, 2]):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        logits = self.w @ np.transpose(testX) + self.b
        pred_Y = np.argmax(logits, axis=0)
        print(logits)
        return pred_Y
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    
    
class MyClassifier_OvO:
    def __init__(self, num_class: int):
        self.num_class = num_class  # number of classes
        self.my_classifiers = None
    
    def train(self, trainX, trainY):
        ''' Task 1-2 '''
        
        classes = np.unique(trainY)
        classifiers = {}
        
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i, class_j = classes[i], classes[j]
                
                # Filter data for the two classes
                mask = (trainY == class_i) | (trainY == class_j)
                X_pair = trainX[mask]
                y_pair = trainY[mask]
                y_binary = np.where(y_pair == class_i, 1, -1)  # Map to {1, -1}
                
                # Define CVXPY variables
                batch_size, n_features = X_pair.shape
                w = cp.Variable(n_features)
                b = cp.Variable()
                xi = cp.Variable(len(y_binary))  # Slack variables
                C = 1.0  # Regularization strength
                
                # Define the objective function with L1 norm on w
                objective = cp.Minimize(cp.norm(w, 1) + C * cp.sum(xi))
                
                # Define constraints
                constraints = [y_binary[i] * (X_pair[i] @ w + b) >= 1 - xi[i] for i in range(batch_size)]
                constraints += [xi >= 0]
                
                # Solve the problem
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                # Store the trained classifier
                classifiers[(class_i, class_j)] = (w.value, b.value)
                
                print(f"Class {i} and {j}:")
                print("Optimal weights (w):", w.value)
                print("Optimal bias (b):", b.value)
                
        self.my_classifiers = classifiers
        
        return classifiers
    
    def predict(self, testX):
        votes = []
        for (class_i, class_j), (w, b) in self.my_classifiers.items():
            predictions = np.sign(testX @ w + b)  # Predict binary labels
            class_preds = np.where(predictions == 1, class_i, class_j)
            votes.append(class_preds)
        
        # Combine votes
        votes = np.array(votes).T
        final_preds = [Counter(v).most_common(1)[0][0] for v in votes]
        return np.array(final_preds)
            
    
class MyClassification_SVMLinear:
    ''' SKlearn package used as external reference '''
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.my_classifiers = None
        
    def train(self, trainX, trainY):
        classes = np.unique(trainY)
        classifiers = {}
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class_i = classes[i]
                class_j = classes[j]
                
                # Extract samples belonging to the current pair of classes
                indices = np.where((trainY == class_i) | (trainY == class_j))[0]
                X_pair = trainX[indices]
                y_pair = trainY[indices]
                
                # Convert labels to binary: class_i -> 0, class_j -> 1
                y_binary = (y_pair == class_j).astype(int)
                
                # Train a linear SVM
                clf = SVC(kernel='linear', probability=False)  # Use linear kernel
                clf.fit(X_pair, y_binary)
                
                # Store the classifier for this pair
                classifiers[(class_i, class_j)] = clf
                
        self.my_classifiers = classifiers
        return classifiers
    
    def predict(self, testX):
        votes = []
        for (class_i, class_j), clf in self.my_classifiers.items():
            # Get binary predictions
            binary_preds = clf.predict(testX)
            
            # Convert binary predictions back to original class labels
            class_preds = np.where(binary_preds == 0, class_i, class_j)
            votes.append(class_preds)
        
        # Combine votes
        votes = np.array(votes).T
        final_preds = [Counter(v).most_common(1)[0][0] for v in votes]
        return np.array(final_preds)

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes  # number of classes
        self.labels = None
        self.cluster_centers_ = None  # To store the cluster centroids

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''
        batch_size, feature_dim = trainX.shape  # Number of data points and feature dimensions
        num_clusters = self.num_classes  # Number of clusters

        # Variables for LP
        C = cp.Variable((num_clusters, feature_dim))  # Cluster centroids (num_clusters x feature_dim)
        Z = cp.Variable((batch_size, num_clusters), boolean=True)  # Assignment matrix (batch_size x num_clusters)
        U = cp.Variable((batch_size, num_clusters, feature_dim))  # Auxiliary variables for |x_i - c_k|

        # Constraints
        constraints = []

        constraints += [cp.sum(Z, axis=1) == 1]

        for i in range(batch_size):
            for k in range(num_clusters):
                for j in range(feature_dim):
                    constraints += [U[i, k, j] >= trainX[i, j] - C[k, j]]
                    constraints += [U[i, k, j] >= C[k, j] - trainX[i, j]]

        # Objective
        objective = cp.Minimize(cp.sum(cp.multiply(Z, cp.sum(U, axis=2))))

        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Record optimal centroid of each cluster
        self.cluster_centers_ = C.value 
        self.labels = np.argmax(Z.value, axis=1)  # Assign cluster labels

        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Compute distances between each test point and all cluster centers
        distances = np.abs(testX[:, np.newaxis, :] - self.cluster_centers_).sum(axis=2)  # Shape: (num_test_points, num_clusters)

        # Assign each test point to the nearest cluster (smallest distance)
        pred_labels = np.argmin(distances, axis=1)

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 (Option 1) ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    