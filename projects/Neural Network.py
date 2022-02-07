
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix



class dlnet:

    def __init__(self, x, y, lr = 0.01):

        self.X=x 
        self.Y=y 
        self.Yh=np.zeros((1,self.Y.shape[1]))
        self.dims = [47, 25, 1] 
        self.param = { } 
        self.ch = {} 
        self.loss = [] 
        self.lr=lr
        self.sam = self.Y.shape[1] 
        self._estimator_type = 'regressor'
        self.neural_net_type = "Relu -> Tanh" 


    def nInit(self): 
       
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))     


    def Relu(self, u):
       
        return np.maximum(0, u);
        raise NotImplementedError
    

    def Tanh(self, u):
       
        return np.tanh(u)
        raise NotImplementedError
    
    
    def dRelu(self, u):
        
        u[u<=0] = 0
        u[u>0] = 1
        return u

    def dTanh(self, u):
       
        o = np.tanh(u)
        return 1-o**2
    
    
    def nloss(self,y, yh):
       
        sum = np.sum((y- yh) ** 2)
        coe = 1/(2*y.shape[1])
        return sum * coe
        raise NotImplementedError
        

    def forward(self, x):
       
            
        self.ch['X'] = x 
        u1 = np.dot(self.param['theta1'], self.ch['X']) + self.param['b1']
        o1 = self.Relu(u1)
        self.ch['u1'],self.ch['o1']=u1,o1
        u2 =  np.dot(self.param['theta2'], o1) + self.param['b2']
        #o2 = self.Tanh(u2) 
        #print(o2)
        #the outputs of the second neural work are negative, so we should not use tanh as activation function 
        o2 = u2
  
        self.ch['u2'],self.ch['o2']=u2,o2 
        
        return o2 #keep
        
    
    
    def backward(self, y, yh):
       

        # set dLoss_o2     

        dLoss_o2 = (self.ch['o2'] - y)/ y.shape[1]

       
        dLoss_u2 = np.multiply(dLoss_o2, (1- self.Tanh(self.ch['u2'])**2))
       
     
        dLoss_theta2 = np.matmul(dLoss_u2, self.ch['o1'].T) # 1 x 15
        dLoss_b2 = np.matmul(dLoss_u2, np.ones((dLoss_u2.shape[1], 1)))
    
        dLoss_o1 = np.matmul(self.param['theta2'].T, dLoss_u2)
        self.ch['dLoss_o1'] = dLoss_o1
        self.ch['dLoss_u2'] = dLoss_u2
        self.ch['dLoss_theta2'] = dLoss_theta2
        self.ch['dLoss_b2'] = dLoss_b2

        dLoss_u1 = np.multiply(dLoss_o1, self.dRelu(self.ch['u1']))
        dLoss_theta1 = np.matmul(dLoss_u1, self.ch['X'].T)
        dLoss_b1 = np.matmul(dLoss_u1, np.ones((dLoss_u1.shape[1], 1)))
        
        self.ch['dLoss_u1'] = dLoss_u1
        self.ch['dLoss_theta1'] = dLoss_theta1
        self.ch['dLoss_b1'] = dLoss_b1

        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2 
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1 
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1 
        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1 


    def gradient_descent(self, x, y, iter = 150000):
        self.nInit()
        i = 0
        while i < iter :
            yh = self.forward(x)
            loss = self.nloss(y, yh)
            self.loss.append(loss)   
            if i % 5000 == 0:
                print ("Loss after iteration",i, "is", loss )        
            self.backward(y,yh)
            i = i + 1
        
  

    def predict(self, x): 
        '''
        This function predicts new data points
        It is already implemented for you
        '''
        Yh = self.forward(x)
        return Yh

#import dataset
datasetNBA = np.genfromtxt('finalNBA.csv', delimiter=',')
#get labels
y = datasetNBA[:,0]
print(y.shape)
#Normalization, avoid vanishing gradient
for j in range(len(y)) :
    y[j] = (y[j] - np.mean(y))/np.std(y)
#get features
x = datasetNBA[:,1:]
#Normalization
for i in range(len(x[0])): 
        x[:,i] = (x[:, i] - np.mean(x[:, i ])) / np.std(x[:, i])
#trainning set
trainningY = y[0:500]
trainningX = x[0:500,:]
#convert trainningY to a 2D array
trainningY = trainningY.reshape(1, trainningY.shape[0])
#test Set
testX = x[500:,:]
#print(y.shape)

model = dlnet(trainningX.T,trainningY)
model.nInit()
model.gradient_descent(trainningX.T,trainningY)
result = model.predict(x[500:,:].T)
# label = y[500:].reshape(1, y[500:].shape[0])
# print(label.shape)
error = sklearn.metrics.r2_score(y[500:],result[0,:])
print("The R2 value is", error)
