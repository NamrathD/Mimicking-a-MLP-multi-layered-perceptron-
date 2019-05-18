# In this program, I will mimick the basic function of MLP
import numpy as np
import pandas as pd


class MLPClassifier():
    def __init__(self,hidden_layer_sizes=(100, ),learning_rate_init=0.001,alpha=0.0001,max_iter=200,random_state=None): #alpha here means regularization hyperparameter
        self.hidden_layer_structure = hidden_layer_sizes # (100,) defaults to 1 hidden layer with 100  nodes
        self.lr = learning_rate_init
        self.alpha = alpha # regulaized hyperparameter
        self.max_iter = max_iter # maximum number of iterations
        self.randomseed = random_state

    def _chk_if_x_and_y_are_proper_types(self,x,y):
        if isinstance(x,np.ndarray) == False or isinstance(y,np.ndarray) == False : # to check if the user is inputting a numpy array or not
            raise TypeError ("Make sure to send numpy array for x and y")
        if len(x.shape) <= 1 or  len(y.shape) <= 1:
            exit("Reshape x and y to get proper dimensions")
        if x.shape[0]!=y.shape[0]:
            exit("make sure to send same number of rows (or samples) for x and y")
    def _inti_variables(self,x,y):
        np.random.seed(self.randomseed)
    # Creating helpful variables
        self.Num_hidden_nodes = self.hidden_layer_structure[0]
        if len(self.hidden_layer_structure) == 1:
            self.Num_hidden_layers = 1
        else:
            self.Num_hidden_layers = self.hidden_layer_structure[1]
        self.Num_input_nodes = x.shape[1]
        self.Num_output_nodes = y.shape[1]
        self.total_layers = self.Num_hidden_layers + 2
        self.Num_training_examples = x.shape[0]
    # Creating parameters
        self.paraInput = [[np.random.uniform(-4,4) for i in range(self.Num_input_nodes+1)]for j in range(self.Num_hidden_nodes)] # random value between 0 and 1 and scaled by some episilon constant as suggested by andrew ng
        self.paraOutput = [[np.random.uniform(-4,4) for i in range(self.Num_hidden_nodes+1)]for j in range(self.Num_output_nodes)]
        self.paraHidden = [[[np.random.uniform(-4,4) for i in range(self.Num_hidden_layers-1)]for j in range(self.Num_hidden_nodes+1)]for k in range(self.Num_hidden_nodes)]
        self.paraInput = np.array(self.paraInput,dtype=float) # only for 1st parameter layers between input and 1st hidden layer
        self.paraOutput = np.array(self.paraOutput,dtype=float) # only for last parameter layer values, between output and last hidden layer
        self.paraHidden = np.array(self.paraHidden,dtype=float) # exists only if layers are 2 or more
    # variables that will be computed by Forward Propagation
        self.hidden_layer = np.array([[1 for _ in range(self.Num_hidden_layers)] for j in range(self.Num_hidden_nodes+1)],dtype=float) #hidden activation values will be stored here and change for every example iterationsr
        self.input_layer = np.array([1 for _ in range(self.Num_input_nodes+1)], dtype=float) # includes bias units
        self.output_layer = np.array([1 for _ in range(self.Num_output_nodes)], dtype=float)

        self.input_layer = self.input_layer.reshape(-1, 1)
        self.output_layer = self.output_layer.reshape(-1, 1) # reshaping since input and output layers will always will be 1 col
    # intializing errors that will be computed in Backward propagations
        self.Er_output = np.array([0]*len(self.output_layer),dtype=float) # error only gotten for output
        self.Er_hidden = np.array([[0 for i in range(self.Num_hidden_layers)]for _ in range(self.Num_hidden_nodes+1)],dtype=float) # error for hidden layers
        self.Er_output = self.Er_output.reshape(-1,1) # reshaping since Error for output layer will always will have 1 col
    # intializing Delta values, which will accumulate the error for all training examples for each iterations
    # There are 3 Delta values, similar to parameter variables, because the input, output, hidden parameters can have different dimensions
        self.Delta_input = np.array([[0 for i in range(self.Num_input_nodes+1)]for j in range(self.Num_hidden_nodes)],dtype=float)
        self.Delta_output = np.array([[0 for i in range(self.Num_hidden_nodes+1)]for j in range(self.Num_output_nodes)],dtype=float)
        self.Delta_hidden = np.array([[[0 for i in range(self.Num_hidden_layers-1)]for j in range(self.Num_hidden_nodes+1)]for k in range(self.Num_hidden_nodes)],dtype=float)
    # intializing Partial Derivatives variable, 3 different variables, similar to paramter and delta vlaues
        self.D_input = np.array([[0 for i in range(self.Num_input_nodes+1)]for j in range(self.Num_hidden_nodes)],dtype=float)
        self.D_output = np.array([[0 for i in range(self.Num_hidden_nodes+1)]for j in range(self.Num_output_nodes)],dtype=float)
        self.D_hidden = np.array([[[0 for i in range(self.Num_hidden_layers-1)]for j in range(self.Num_hidden_nodes+1)]for k in range(self.Num_hidden_nodes)],dtype=float)

    def _sigmoid(self, z):
        g = (1 / (1 + np.exp(-z)))
        return g

    def _Forward_prop(self,x,j): # j jere is the training example index
        ##### Below code is first section for doing forward propagation. It is to compute the first hidden layer
        self.input_layer[1:, 0] = x[j, :]  # so that only non bias units will get updated for all x training examples
        self.hidden_layer[1:, 0] = self._sigmoid(np.dot(self.paraInput, self.input_layer)).reshape(1, -1)
        ##### Doing forward propogation for any hidden layers more than 1
        if self.Num_hidden_layers > 1:
            for i_hid in range(1, self.Num_hidden_layers):
                self.hidden_layer[1:, i_hid] = self._sigmoid(np.dot(self.paraHidden[:, :, i_hid - 1], (self.hidden_layer[:,i_hid - 1]).reshape(-1,1))).reshape(1, -1)  # doing i_hid-1, because that way we will use the right parameters to calculate their respective output hidden layers
        ##### computing for output layer in forward propagation using the last hidden layer

        self.output_layer = self._sigmoid(np.dot(self.paraOutput, (self.hidden_layer[:, (self.Num_hidden_layers - 1)]).reshape(-1,1))).reshape(-1, 1)
        #print(self.output_layer)
    def _Back_prop(self,y,j):
    #### Back Propagation for output layer and the last hidden layer
        y_actual = np.array(y[j, :], dtype=float).reshape(-1, 1)
        self.Er_output = self.output_layer - y_actual  # intial error for output layer
        Transpose_paraOutput = np.transpose(self.paraOutput)  # transposing para_output so that I can matrix multiply with Er_output
        second_term = np.array((self.hidden_layer[:, self.Num_hidden_layers - 1] * (1 - self.hidden_layer[:, self.Num_hidden_layers - 1])),dtype=float).reshape(-1, 1)
        self.Er_hidden[:, self.Num_hidden_layers - 1] = (np.dot(Transpose_paraOutput, self.Er_output) * second_term).reshape(1,-1)  # finding out error for last hidden# layer

    ### Back Propagation for other hidden layers
        if self.Num_hidden_layers > 1:
            for i_hid in range(self.Num_hidden_layers - 1, 0, -1):
                Transpose_parahidden = np.transpose(self.paraHidden[:, :, i_hid - 1])
                second_term = np.array((self.hidden_layer[:, i_hid - 1] * (1 - self.hidden_layer[:, i_hid - 1])),dtype=float).reshape(-1,1)  # second term in the equation to find the error a *(1-a)
                ER_ofPrevious = self.Er_hidden[1:, i_hid]
                ER_ofPrevious = ER_ofPrevious.reshape(-1, 1)
                self.Er_hidden[:, i_hid - 1] = (np.dot(Transpose_parahidden, ER_ofPrevious) * second_term).reshape(1,-1)
                test = (np.dot(Transpose_parahidden, ER_ofPrevious) * second_term)
                #print(test.shape)
    def _Accumulate_delta(self):
    ##### Accumulating Delta for input
        Transpose_input = np.transpose(self.input_layer)
        error_firstHiddenLayer = (self.Er_hidden[1:, 0]).reshape(-1, 1)
        self.Delta_input = self.Delta_input + np.dot(error_firstHiddenLayer, Transpose_input)

    #### Accumulating Delta for hidden layers if there are more than 1 hidden layer
        if self.Num_hidden_layers > 1:
            for i_hid in range(self.Num_hidden_layers - 1):
                Transpose_hidden = (self.hidden_layer[:, i_hid]).reshape(-1, 1)
                Transpose_hidden = np.transpose(Transpose_hidden)  # transposing the hidden layer values including the bias unit
                error_nextLayer = (self.Er_hidden[1:, i_hid + 1]).reshape(-1, 1)  # getting and shaping the error of l+1
                self.Delta_hidden[:, :, i_hid] = self.Delta_hidden[:, :, i_hid] + np.dot(error_nextLayer,Transpose_hidden)
    #### Accummulating Delta for output layers
        Transpose_LastHiddenLayer = (self.hidden_layer[:, self.Num_hidden_layers - 1]).reshape(-1, 1)
        Transpose_LastHiddenLayer = np.transpose(Transpose_LastHiddenLayer)
        self.Delta_output = self.Delta_output + np.dot(self.Er_output, Transpose_LastHiddenLayer)
    def _CalculatingPartialDer(self):
        #### computing D input matrix
        self.D_input[:,1:] = (1/self.Num_training_examples)*(self.Delta_input[:,1:]) + self.alpha *self.paraInput[:,1:]
        self.D_input[:,:1] = (1/self.Num_training_examples)*(self.Delta_input[:,:1]) # for bias unit = 0 no regularization
        #### computing for D_hidden if there are more than 1 hidden layers

        if self.Num_hidden_layers > 1:
            for i_hid in range(self.Num_hidden_layers-1):
                self.D_hidden[:,1:,i_hid] = (1/self.Num_training_examples)*(self.Delta_hidden[:,1:,i_hid]) + self.alpha*self.paraHidden[:,1:,i_hid]
                self.D_hidden[:,:1,i_hid] = (1/self.Num_training_examples)*(self.Delta_hidden[:,:1,i_hid])
        #### Computing for D_output
        self.D_output[:,1:] = (1/self.Num_training_examples)*(self.Delta_output[:,1:]) + self.alpha*self.paraOutput[:,1:]
        self.D_output[:,:1] = (1/self.Num_training_examples)*(self.Delta_output[:,:1])
    def _GradientDescent(self,x,y):
        # make sure to updatate para
        for i in range(self.max_iter): # for loop to loop for iterations provided by the user or default
            for j in range(self.Num_training_examples): # nested for loop to loop for all training samples
                self._Forward_prop(x, j) #Do Forward Propagation
                self._Back_prop(y,j)
                #print("Error For Output:",self.Er_output)
                self._Accumulate_delta()
            self._CalculatingPartialDer()
            #### Updating the parameters
            self.paraInput = self.paraInput - self.lr * (self.D_input)
            self.paraHidden = self.paraHidden - self.lr * (self.D_hidden)
            self.paraOutput = self.paraOutput - self.lr * (self.D_output)
            ### Clearing Delta Variable so that it can again start to accumulate the partial deravites for each training example for thenew updated parametrs
            self.Delta_input[:,:] = 0
            self.Delta_hidden[:,:,:] = 0
            self.Delta_output[:,:] = 0



    def fit(self,x,y):
        self._chk_if_x_and_y_are_proper_types(x, y)
        self._inti_variables(x,y)
        self._GradientDescent(x,y)

    def predict(self,x):
        y_pred = []
        for i in range(len(x[:,0])):
            self._Forward_prop(x,i)
            #print(x[i])
            #print("input:",self.input_layer)
            #print("ParaOutput:",self.paraOutput)
            #print("last layer values",(self.hidden_layer[:, (self.Num_hidden_layers - 1)]).reshape(-1,1))

            y_pred.append(float(self.output_layer))
        return y_pred


#x = [[0], [1],[1],[1]]
#y = [[1], [0],[0],[0]]
x = [[0,0],[1,0],[0,1],[1,1]]
y = [[1],[0],[0],[1]]



x = np.array(x)
y = np.array(y)
#mlp = MLPClassifier(hidden_layer_sizes=(3,3), learning_rate_init=0.15,max_iter=1,random_state=5,alpha=0)
mlp = MLPClassifier(hidden_layer_sizes=(5,4), learning_rate_init=0.4,max_iter=1000,random_state=5,alpha=0)
mlp.fit(x,y)

x_test = [[0,0],[1,0],[0,0],[0,0],[1,0],[0,0],[1,1]]
x_test = np.array(x_test)
y_pred = mlp.predict(x)
print("\n\n","Y_predicted values:",y_pred,"\n\n")