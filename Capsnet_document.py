#Capsulenet for document recognition
#need to move relevant files from reference to the folder that this is in


#Imports:
from copy import copy
from __future__ import division, print_function, unicode_literals

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import data
np.random.seed(42) #same output gaurentee
tf.set_random_seed(42)

#Importing data
from "-----" import "input_data"
"var" = input_data.read_data_sets("/path")
#will extract data from specified location

#primary placeholder for the input data, will consist of lines(pre-concatonation) taken from linesegmentation.py
X = tf.placeholder(shape=[None, '', '', 1], dtype=tf.float32, name = "X") #need dimensions of each segment

#First capsule layer:
#each segment will produce 32 maps each of which containing a grid(OF NOT YET DETERMINED DIMESIONS) of 8 dimensional vectors
capsulemaps1 = 32
capsulecaps1 = capsulemaps*'TBD'*'TBD' #This produces all of the first layer capsules
capsuledims1 = 8 #Can tweak, but good for handwriting
#The first layer is computed via the application of 2 convolutional layers
convlayer_parameters1 = { "filters": 256,"kernel_size": 9,"strides": 1,"padding": "valid","activation": tf.nn.relu,

}
convlayer_parameters2 = { "filters": capsulemaps * capsuledims, # 256 convolutional filters
    "kernel_size": 9,"strides": 2,"padding": "valid","activation": tf.nn.relu

}
#Input of first layer is X(placeholder), 2nd layer is output of first.
#second layer produces 'TBD' feature maps. Using reshap to get a grid of vectors.
convlayer1 = tf.layers.conv2d(X,name = "convlayer1", **convlayer_parameters1)
convlayer2 = tf.layers.conv2d(convlayer1, name = "convlayer2", **convlayer_parameters2)

#The output is reshaped into a grid of 8 dimensional vectors which will represent primary capsule outputs. Since this output is fully
#connected to the next layer can flatten the grids
capsuleRs = tf.reshape(convlayer2, [-1,capsulecaps1, casulecapsdims1], name = "capsuleRs") #reshapes to one long list for each instance in batch

#Vectors are "squashed" (normalized along its axis)

def squashing(s, axis = -1, epsilon = 1e-7, name = None):
    with tf.name_scope(name, default_name="squashing"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector #"safe" norm approximation
#Applying the above to the output of each of the primary capsules
capsuleout1 = squashing(capsuleRs, name= "capsuleout1")

#Next capsule will contain 26 capsules (one for each letter) and each vector is 16 dimensional
capsulecaps2 = 26
capsuledims2 = 16 #Each vector is 16 dimensional
#for each capsule in the first layer we want to predict an output for every capsule in the  second layer.
#will do this with transform matrix(W), will map prediction onto 16 dim vector
#using matmul to multiply higher dimensional arrays
#tf.tile used for creating an array composed of a number of base arraysabs

init_sigma = 0.1
Winitalize = tf.random_normal( #outputs random values from a normal distribution
    shape = (1,capsulecaps1, capsulecaps2, capsulecapsdims2, capsulecapsdims1), stddev = init_sigma, dtype = tf.float32, name = "Winitialize")
W = tf.Variable(Winitalize, name = "W")

#make first array by repeating W once per instance
batch_size = tf.shape(x)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name = "W_tiled")
#2nd array...contains the output of the first layer of capsules repeated 26 times
capsuleoutLarger1 = tf.expand_dims(capsuleout, -1, name = "capsuleoutLarger1")
capsuleoutTile1 = tf.expand_dims(capsuleoutLarger1, 2, name="capsuleoutTile1")
capsuleoutTiled1 = tf.tile(capsuleoutTile1, [1, 1, capsulecaps2, 1, 1], name="capsuleoutTiled1")
#Multiplying arrays to get predicted output vectors.

capsule2Predict =  tf.matmul(W_tiled, capsuleoutTiled1,
                            name="capsule2Predict")
#Routing by agreement.
#all routing weights are initialized to 0. One weight per pair of capsules and for each instance
#Then compute the softmax of each primary capsules 26 raw initial routing weights.
#Then compute weighted sum of prediceted output vectors
#Run squash function here to get the outputs of the letter capsules

initRoutingWeights = tf.zeros([batch_size,capsulecaps1, capsulecaps2, 1, 1], dtype = np.float32, name = "InitRoutingWeights")

routingWeights = tf.nn.softmax(initRoutingWeights, dim = 2, name = "routingWeights" )

weightedPredict = tf.multiply(routingWeights, capsule2Predict,name="weightedPredict")

weightedSum = tf.reduce_sum(weightedPredict, axis=1, keep_dims=True, name="weightedSum")

capsuleoutput2First = squash(weightedSum, axis=-2, name="capsuleoutput2First")

#Now there are 26 16 dimensional vectors as output for each instance

#Next must measure prediction loss and update the routing weights.
#Must get scalar product for each predicted vector.

capsuleoutput2FirstTiled = tf.tile(capsuleoutput2First, [1, capsulecaps1, 1, 1, 1],name="capsuleoutput2FirstTiled")

PredicitonAgreement = = tf.matmul(capsule2Predict, capsuleoutput2FirstTiled, transpose_a=True, name="PredicitonAgreement")

#update the initial/ raw weights used for routing
initRoutingWeightsSecond = tf.add(initRoutingWeights, PredicitonAgreement,
                             name="initRoutingWeightsSecond")
#Same routing by agreement procedure as before but uing the raw routing weights obtained from this round

routingWeights2 = tf.nn.softmax(initRoutingWeightsSecond, dim = 2, name = "routingWeights2" )

weightedPredict2 = tf.multiply(routingWeights2, capsule2Predict,name="weightedPredict2")

weightedSum2 = tf.reduce_sum(weightedPredict2, axis=1, keep_dims=True, name="weightedSum2")

capsuleoutput2Second = squash(weightedSum2, axis=-2, name="capsuleoutput2Second")

capsuleoutput2 = capsuleoutput2Second

#Class probability output:
#squash function makes it so that we cant use regular norm function
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
letterProbability = safe_norm(capsuleoutput2, axis =-2, name = "letterProbability")
#Highest probability means most likely letter
letterProbabilityClass = tf.argmax(letterProbability, axis=2, name="y_proba")

predictedClass = tf.squeeze(letterProbabilityClass, axis=[1,2], name="predictedClass")

#Training begins(will calculate margin loss between prediction and labels)
y = tf.placeholder(shape=[None], dtype=tf.int64, name ="y")

#margin loss makes it possible to calculate 2 or more differnt letters per imag
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

capsuleoutput2norm = safe_norm(capsuleoutput2, axis=-2, keep_dims=True,
                              name="capsuleoutput2norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - capsuleoutput2norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., capsuleoutput2norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
#MUST do reconstruction, decoder, reconstruction loss and must train the model
