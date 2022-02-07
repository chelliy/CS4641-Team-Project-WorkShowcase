import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix



class dlnet:

    def __init__(self, x, y, lr = 0.01):
        '''
        This method initializes the class, it is already implemented for you. 
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            param: dictionary of different layer parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

        '''        
        self.X=x # features
        self.Y=y # ground truth labels

        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [9, 30, 1] # dimensions of different layers

        self.param = { } # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in backprop
        self.loss = [] # list to store loss values 

        self.iter = 0 # iterator to index into data for making a batch 
        self.batch_size = 300 # batch size 
        
        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'regressor'
        self.neural_net_type = "Relu -> Tanh" 


    def nInit(self): 
        '''
        This method initializes the neural network variables, it is already implemented for you. 
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.
        '''   
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))     


    def Relu(self, u):
        '''
        In this method, you are going to implement element wise Relu. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Relu(u) 
        '''
        #TODO: implement this 
        # raise NotImplementedError

        # newU = np.copy(u)
        # compareone = np.zeros((u.shape[0], u.shape[1]))
        # return np.maximum(newU, compareone)
        newU = np.copy(u)
        newU[newU<=0] = 0
        return newU
    

    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Tanh(u) 
        '''
        #TODO: implement this 
        # raise NotImplementedError

        # newUPos = np.copy(u)
        # newUNeg = -np.copy(u)
        # newUPos = np.exp(newUPos)
        # newUNeg = np.exp(newUNeg)
        # return (newUPos-newUNeg)/(newUPos+newUNeg)

        return np.tanh(u)
    
    
    def dRelu(self, u):
        '''
        This method implements element wise differentiation of Relu, it is already implemented for you.  
        Input: u of any dimension
        return: dRelu(u) 
        '''
        u[u<=0] = 0
        u[u>0] = 1
        return u

    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh, it is already implemented for you.
        Input: u of any dimension
        return: dTanh(u) 
        '''
        o = np.tanh(u)
        return 1-o**2
    
    
    def nloss(self,y, yh):
        '''
        In this method you are going to implement mean squared loss. 
        Refer to the description in the notebook and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output after Relu 

        return: MSE 1x1: loss value 
        '''
        
        #TODO: implement this 
        # raise NotImplementedError

        return np.sum(np.power((y-yh), 2)) / (2*y.shape[1])



        

    def forward(self, x):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemented methods.
        Refer to the description in the notebook and implement the appropriate mathematical equations.
        do not change the lines followed by #keep. 
        '''
        
        #TODO: implement this
        # raise NotImplementedError 
            
        if not len(self.param) > 0:
            self.nInit()

        theta1 = self.param['theta1']
        theta2 = self.param['theta2']
        b1 = self.param['b1']
        b2 = self.param['b2']

        self.ch['X'] = x #keep
        
        u1 = np.dot(theta1,x) + b1

        o1 = self.Relu(u1)

        self.ch['u1'],self.ch['o1']=u1,o1 #keep 


        u2 = np.dot(theta2,o1) + b2

        o2 = self.Tanh(u2)
    
        self.ch['u2'],self.ch['o2']=u2,o2 #keep

        self.Yh = o2
        
        return o2 #keep
        
    

    def backward(self, y, yh):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemented methods, and other variables as well
        Refer to the description in the notebook and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.  
        '''
        #TODO: implement this 
        # raise NotImplementedError

        # set dLoss_o2     
          
        dLoss_o2 = (self.ch['o2'] - y)/self.sam

        #Implement equations for getting derivative of loss w.r.t u2, theta2 and b2
        # set dLoss_u2, dLoss_theta2, dLoss_b2 
        
        dLoss_u2 = dLoss_o2 * self.dTanh(self.ch['u2'])

        dLoss_theta2 = np.dot(dLoss_u2, np.transpose(self.ch['o1']))

        dLoss_b2 = np.sum(dLoss_u2)

        # set dLoss_o1

        dLoss_o1 = np.dot(np.transpose(self.param['theta2']), dLoss_u2)
        
        #Implement equations for getting derivative of loss w.r.t u1, theta1 and b1
        # set dLoss_u1, dLoss_theta1, dLoss_b1

        dLoss_u1 = dLoss_o1 * self.dRelu(self.ch['o1'])

        dLoss_theta1 = np.dot(dLoss_u1, np.transpose(self.ch['X']))

        dLoss_b1 = np.dot(dLoss_u1, np.ones((dLoss_u1.shape[1], 1)))         
        
            
        #parameters update, no need to change these lines
        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2 #keep
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2 #keep
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1 #keep
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1 #keep
        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1 #keep
        
        

    def gradient_descent(self, x, y, iter = 80000):
        '''
        This function is an implementation of the gradient descent algorithm.
        Note:
        1. GD considers all examples in the dataset in one go and learns a gradient from them. 
        2. One iteration here is one round of forward and backward propagation on the complete dataset. 
        3. Append loss at multiples of 2000 i.e. at 0th, 2000th, 4000th .... iterations
        ''' 
        #Todo: implement this 
        # raise NotImplementedError

        self.X = x
        self.Y = y
        for i in range(iter):
            self.forward(self.X)
            self.backward(self.Y, self.Yh)
            if i%2000 == 0:
                loss = self.nloss(self.Y, self.Yh)
                self.loss.append(loss)
                print("loss after iteration " + str(i) + ": " + str(loss))



        
    
    #bonus for undergrdauate students 
    def batch_gradient_descent(self, x, y, iter = 100000):
        '''
        This function is an implementation of the batch gradient descent algorithm

        Note: 
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient 
        2. One iteration here is one round of forward and backward propagation on one minibatch. 
           You will use self.iter and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.
        3. Append loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations  
        4. It is fine if you get a noisy plot since learning on a batch adds variance to the 
           gradients learnt
        '''
        
        #Todo: implement this 
        # raise NotImplementedError

        self.X = x
        self.Y = y
        start = 0
        end = self.batch_size
        for i in range(iter):
            if end > self.sam:
                newX = np.concatenate((self.X[:,start:self.sam], self.X[:,0:end - self.sam]), axis = 1)
                newY =np.concatenate((self.Y[:,start:self.sam], self.Y[:,0:(end - self.sam)]), axis = 1)
                self.forward(newX)
                self.backward(newY, self.Yh)
                if i%2000 == 0:
                    loss = self.nloss(newY, self.Yh)
                    self.loss.append(loss)
                    print("loss after iteration " + str(i) + ": " + str(loss))
                start = end -self.sam
                end = start + self.batch_size
            else:
                self.forward(self.X[:,start:end])
                self.backward(self.Y[:,start:end], self.Yh)
                if i%1000 == 0:
                    loss = self.nloss(self.Y[:,start:end], self.Yh)
                    self.loss.append(loss)
                    print("loss after iteration " + str(i) + ": " + str(loss))
                start = end
                end = end + self.batch_size


    def predict(self, x): 
        '''
        This function predicts new data points
        It is already implemented for you
        '''
        Yh = self.forward(x)
        return Yh

#import dataset
datasetNBA = np.genfromtxt('quarterbacks.csv', delimiter=',')
#get labels
y = datasetNBA[:,datasetNBA.shape[1] - 1]
print(y.shape)
print(y)
#Normalization, avoid vanishing gradient
for j in range(len(y)) :
    y[j] = (y[j] - np.mean(y))/np.std(y)
#get features
x = datasetNBA[:,1:]
print(x)
#Normalization
for i in range(len(x[0])): 
        x[:,i] = (x[:, i] - np.mean(x[:, i ])) / np.std(x[:, i])
#trainning set
trainningY = y[0:1500]
trainningX = x[0:1500,:]
#convert trainningY to a 2D array
trainningY = trainningY.reshape(1, trainningY.shape[0])
#test Set
testX = x[1500:,:]
#print(y.shape)

model = dlnet(trainningX.T,trainningY)
model.nInit()
model.gradient_descent(trainningX.T,trainningY)
result = model.predict(x[1500:,:].T)
# label = y[500:].reshape(1, y[500:].shape[0])
# print(label.shape)
error = sklearn.metrics.r2_score(y[1500:],result[0,:])
print("The R2 value is", error)
