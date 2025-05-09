import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class ClassificationPipeline:
    """
    Two-stage classification pipeline: SVM followed by FFNN.
    
    The SVM classifier determines the likelihood of one class from the rest.
    The FFNN determines probabilities for the remaining two classes.
    """
    
    def __init__(self, svm_target=0):
        """
        Initialize the classification pipeline.
        
        Parameters:
        -----------
        svm_target : int
            Specifies the target class for the SVM (0, 1, or 2).
        """
        # Store the target class for the SVM
        self.svm_target = svm_target
        
        # TODO: Initialize the SVM classifier
        self.svm = SVC()
        
        # TODO: Initialize attributes for FFNN
        # You may add additional attributes as needed for your implementation
        self.input_size = 4
        self.output_size = 3
        self.h1=4
        self.w1=np.random.rand(self.input_size,self.h1)
        self.b1=np.random.rand(1,self.h1)
        self.w2=np.random.rand(self.h1,self.output_size)
        self.b2=np.random.rand(1,self.output_size)
        self.lr=0.01
        self.epochs=10000
    
        
    def fitSVM(self, X, y):
        """
        Convert the original labels into binary labels and fit the SVM.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Convert labels to binary (1 for target class, 0 for others)
        # and fit the SVM model
        np.where(y==self.svm_target,1,0)
        self.svm.fit(X,y)

        
    def probsSVM(self, X):
        """
        Return the probability estimates from SVM.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Probability that each input belongs to the target class
        """
        # TODO: Return probabilities from SVM
        y_pred=self.svm.decision_function(X)
        index=self.svm_target
        ans=[]
        for temp in y_pred:
            ans.append(float(temp[index]))

        return ans
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(self,x):
        temp=np.max(x)
        new_x=x-temp
        return np.exp(new_x)/(np.sum(np.exp(new_x),axis=0))

    def grad_sigmoid(self,a):
        return a*(1-a)

    # def grad_softmax(self,a):

    def loss(self,y,y_hat):
        return -np.sum(y*np.log(y_hat+1e-9))
        # return np.mean((y-y_hat)**2)

    def one_hot_y(self,y):
        y=y.reshape(-1,1)
        m,n=y.shape
        ans=np.zeros((m,self.output_size))
        for i in range(m):
            temp=y[i]
            ans[i][temp]=1
        return ans

    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities for each class
        """
        # TODO: Implement forward propagation
        # self.X=X
        w1=self.w1
        w2=self.w2
        b1=self.b1
        b2=self.b2

        z1=X @ w1 +b1
        a1=self.sigmoid(z1)
        z2=a1 @ w2 + b2
        a2=self.sigmoid(z2)
        return [z1,a1,z2,a2]

        
    def backward(self, X, y, output):
        """
        Backward propagation to update weights.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
        y : numpy.ndarray
            True labels 
        output : numpy.ndarray
            Output from forward propagation
        """
        # TODO: Implement backpropagation
        z1,a1,z2,a2=output
        temp_y=self.one_hot_y(y)
        
        grad_new=self.grad_sigmoid(a2)
        dz2=(a2-temp_y)*grad_new
        # dz2=(a2-temp_y)
        dw2=np.dot(a1.T,dz2)
        db2=np.sum(dz2,axis=0)

        w2=self.w2
        grad=self.grad_sigmoid(a1)
        dz1=np.dot(dz2,w2.T)*grad
        dw1=np.dot(X.T,dz1)
        db1=np.sum(dz1,axis=0)


        self.w2-=self.lr*dw2
        self.w1-=self.lr*dw1
        self.b2=self.b2-self.lr*db2
        self.b1=self.b1-self.lr*db1
        return None
    


    def fitFFNN(self, X, y):
        """
        Train the FFNN component of the pipeline.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement FFNN training with backpropagation
        for epoch in range(self.epochs):
            output=self.forward(X)
            z1,a1,z2,a2=output
            y_hat=a2
            y_hot=self.one_hot_y(y)
            loss=self.loss(y_hot,y_hat)
            
            self.backward(X,y,output)

            # if epoch%1000==0:
                # print(f"epoch:{epoch}  loss:{loss}")
        
    def probsFFNN(self, X):
        """
        Return the probability estimates from FFNN.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Probability estimates for the two classes other than the target class
        """
        # TODO: Return probabilities from FFNN for the other two classes
        output=self.forward(X)
        z1,a1,z2,a2=output
        index=self.svm_target
        indices=[0,1,2]
        ans=np.zeros((a2.shape[0],2))
        for j in range(a2.shape[0]):
            second=0
            for i in indices:
                if i!=index:
                    if second:
                        ans[j][1]=a2[j][i]
                        second=0
                    else:
                        ans[j][0]=a2[j][i]
                        second=1
                    # ans.append(float(line[i]))
        # ans.to_numpy()
        ans.reshape(-1,2)
        return ans

    def fit(self, X, y):
        """
        Fit both SVM and FFNN components of the pipeline.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement pipeline fitting
        # using the fitSVM and fitFFNN functions
        self.fitSVM(X,y)
        self.fitFFNN(X,y)
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray(one-hot encoded)
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        # TODO: Implement prediction using both SVM and FFNN
        # Combine the probabilities and determine the final class prediction
        t1=self.probsSVM(X)
        t2=self.probsFFNN(X)
        m=len(t1)
        ans=np.zeros((m,3))
        index=self.svm_target
        for i in range(m):
            temp=0
            for j in range(3):
                if(index==0):
                    ans[i][0]=t1[i]
                    ans[i][1]=t2[i][0]
                    ans[i][2]=t2[i][1]
                elif(index==2):
                    ans[i][0]=t2[i][0]
                    ans[i][1]=t2[i][1]
                    ans[i][2]=t1[i]
                elif(index==1):
                    ans[i][0]=t2[i][0]
                    ans[i][1]=t1[i]
                    ans[i][2]=t2[i][1]
                
        # np.hstack()
        return np.argmax(ans,axis=1)

        
class StandardNNClassifier:
    """
    Standard Feed-Forward Neural Network classifier for comparison.
    
    This implements a three-class classifier using a neural network.
    """
    
    def __init__(self, input_size=4, output_size=3):
        """
        Initialize the neural network classifier.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output classes
        """
        # TODO: Initialize network architecture and parameters
        self.input_size = input_size
        self.output_size = output_size
        self.h1=4
        self.w1=np.random.rand(input_size,self.h1)
        self.b1=np.random.rand(1,self.h1)
        self.w2=np.random.rand(self.h1,output_size)
        self.b2=np.random.rand(1,output_size)
        self.lr=0.01
        self.epochs=10000
        # self.X=None
        # self.y=None
        # Add other necessary attributes for your implementation
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(self,x):
        temp=np.max(x)
        new_x=x-temp
        return np.exp(new_x)/(np.sum(np.exp(new_x),axis=0))

    def grad_sigmoid(self,a):
        return a*(1-a)

    # def grad_softmax(self,a):

    def loss(self,y,y_hat):
        return -np.sum(y*np.log(y_hat+1e-9))
        # return np.mean((y-y_hat)**2)

    def one_hot_y(self,y):
        y=y.reshape(-1,1)
        m,n=y.shape
        ans=np.zeros((m,self.output_size))
        for i in range(m):
            temp=y[i]
            ans[i][temp]=1
        return ans

    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Output probabilities for each class
        """
        # TODO: Implement forward propagation
        # self.X=X
        w1=self.w1
        w2=self.w2
        b1=self.b1
        b2=self.b2

        z1=X @ w1 +b1
        a1=self.sigmoid(z1)
        z2=a1 @ w2 + b2
        a2=self.sigmoid(z2)
        return [z1,a1,z2,a2]

        
    def backward(self, X, y, output):
        """
        Backward propagation to update weights.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
        y : numpy.ndarray
            True labels 
        output : numpy.ndarray
            Output from forward propagation
        """
        # TODO: Implement backpropagation
        z1,a1,z2,a2=output
        temp_y=self.one_hot_y(y)
        
        grad_new=self.grad_sigmoid(a2)
        dz2=(a2-temp_y)*grad_new
        # dz2=(a2-temp_y)
        dw2=np.dot(a1.T,dz2)
        db2=np.sum(dz2,axis=0)

        w2=self.w2
        grad=self.grad_sigmoid(a1)
        dz1=np.dot(dz2,w2.T)*grad
        dw1=np.dot(X.T,dz1)
        db1=np.sum(dz1,axis=0)


        self.w2-=self.lr*dw2
        self.w1-=self.lr*dw1
        self.b2=self.b2-self.lr*db2
        self.b1=self.b1-self.lr*db1
        return None


    def fit(self, X, y):
        """
        Train the neural network using backpropagation.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training feature data
        y : numpy.ndarray
            Training labels
        """
        # TODO: Implement network training
        for epoch in range(self.epochs):
            output=self.forward(X)
            z1,a1,z2,a2=output
            y_hat=a2
            y_hot=self.one_hot_y(y)
            loss=self.loss(y_hot,y_hat)
            
            self.backward(X,y,output)

            # if epoch%1000==0:
                # print(f"epoch:{epoch}  loss:{loss}")
        
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input feature data
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        # TODO: Implement prediction
        output=self.forward(X)
        z1,a1,z2,a2=output
        return np.argmax(a2,axis=1)
