import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
#Import data from data.mat, which contains training, validating and testing data sets
data = sio.loadmat('data.mat')['data'][0,0]
train_x = data[1].T[:, :3] -1
train_t = data[1].T[:, 3] -1

valid_x = data[0].T[:, :3] -1
valid_t = data[0].T[:, 3] -1

test_x = data[2].T[:, :3] -1
test_t = data[2].T[:, 3] -1

#vocab is a list of 250 used to train the model
vocab = []
for i in range(250):
	vocab.append(data[3][0, i][0])
vocab =np.array(vocab)

#train the model for 10 epoch
epochs = 10
batch_size = 100
learning_rate = 0.1
momentum = 0.9
numhid1 = 50
numhid2 = 200
init_wt = 0.01 #
numwords = 3

show_training_after = 100
show_validation_after = 1000

#split training data in to batches of 100 rows of data
num_batch = int(len(train_x)/batch_size)
train_x= train_x[:num_batch*batch_size, :].reshape(num_batch,batch_size, numwords)
train_t = train_t[:num_batch*batch_size].reshape(num_batch, batch_size)

#initialize the neural weights and biases
word_embedding_weights = init_wt*np.random.random([len(vocab), numhid1])
embed_to_hid_weights = init_wt*np.random.random([numwords*numhid1, numhid2])
hid_to_output_weights = init_wt*np.random.random([numhid2, len(vocab)])

hid_bias = np.zeros(numhid2)
output_bias = np.zeros(len(vocab))

#initiate the change of weights and bias after each training step
word_embedding_weights_delta = np.zeros([len(vocab), numhid1])
word_embedding_weights_gradient =np.zeros([len(vocab),numhid1])

embed_to_hid_weights_delta = np.zeros([numwords*numhid1, numhid2])
hid_to_output_weights_delta = np.zeros([numhid2, len(vocab)])

hid_bias_delta = np.zeros(numhid2)
output_bias_delta = np.zeros(len(vocab))

expansion_matrix = np.identity(len(vocab))

count = 0
tiny = np.exp(-30)

#forward method calculate the output of the neural based on its weights and biases
def forward(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hidden_bias, output_bias):
	embedded_layer_st = []
	for i in range(len(input_batch)):
		t1 = word_embedding_weights[input_batch[i,0]]
		t1 = np.append(t1, word_embedding_weights[input_batch[i,1]])
		t1 = np.append(t1, word_embedding_weights[input_batch[i,2]])
		embedded_layer_st.append(t1)
	embedded_layer_st = np.array(embedded_layer_st)
	inputs_to_hidden_unit = embedded_layer_st.dot(embed_to_hid_weights) +hidden_bias
	hidden_layer_st = 1.0/(1.0+np.exp(-inputs_to_hidden_unit))
#	hidden_layer_state = 1/(1-np.exp(-inputs_to_hidden_unit))
#	hidden_layer_state = 1/(1+ np.exp(-inputs_to_hidden_unit))
#	hidden_layer_state = 1/(1- np.exp(inputs_to_hidden_unit))

	inputs_to_softmax = hidden_layer_st.dot(hid_to_output_weights)+ output_bias
	inputs_to_softmax -= np.amax(inputs_to_softmax, axis = 1).reshape(len(inputs_to_softmax), 1)
	output_layer_st = np.exp(inputs_to_softmax)
	for i in range(len(output_layer_st)):
		output_layer_st[i] = output_layer_st[i]/np.sum(output_layer_st[i])

	return embedded_layer_st, hidden_layer_st, output_layer_st



#--------------------------------
#distance method calculate the distance between two word
def distance(w1, w2):
	r1, r2 = 0, 0
	for i in range(len(vocab)):
		if vocab[i] == w1:
			r1 = i
			break
	for i in range(len(vocab)):
		if vocab[i] == w2:
			r2 = i
			break
	r1 = word_embedding_weights[r1]
	r2 = word_embedding_weights[r2]

	return np.sqrt(np.sum((r1-r2)*(r1-r2)))

#-------------------------------------
#the actual training of the neural network
for epoch in range(epochs):
	this_chunk_CE = 0
	trainset_CE = 0
	for m in range(num_batch):
		input_batch = train_x[m]
		target_batch = expansion_matrix[train_t[m]]

		embedding_layer_state, hidden_layer_state, output_layer_state  =  forward(input_batch,word_embedding_weights,
			embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
		error_deriv = output_layer_state -target_batch

		CE = - np.sum(target_batch*np.log(tiny + output_layer_state))/batch_size
		count +=1
		this_chunk_CE += (CE- this_chunk_CE)/count
		trainset_CE += (CE- trainset_CE)/(m+1)

		if m % show_training_after ==0 :
			count = 0
			if m!= 0:
				print("Batch", m, "Training CE", this_chunk_CE)
			this_chunk_CE = 0

		hid_to_output_weight_gradients = hidden_layer_state.T.dot(error_deriv)
		output_bias_gradients = np.sum(error_deriv, axis = 0)
		back_propagate_1 =  error_deriv.dot(hid_to_output_weights.T)*(hidden_layer_state*(1-hidden_layer_state ))
		embed_to_hid_weight_gradients = embedding_layer_state.T.dot(back_propagate_1)
		hid_bias_gradients   = np.sum(back_propagate_1, axis = 0)

		back_propagate_2 = back_propagate_1.dot(embed_to_hid_weights.T)

		word_embedding_weights_gradient *= 0.0
		for i in range(100):
			for j in range(3):
				word_embedding_weights_gradient[ input_batch[i, j] ]= word_embedding_weights_gradient[input_batch[i, j]]+back_propagate_2[i,numhid1*j: numhid1*(j+1)]

		#update the weights and biases
		word_embedding_weights_delta = momentum*word_embedding_weights_delta+word_embedding_weights_gradient/batch_size
		word_embedding_weights -= word_embedding_weights_delta*learning_rate

		embed_to_hid_weights_delta = momentum*embed_to_hid_weights_delta + embed_to_hid_weight_gradients/batch_size
		embed_to_hid_weights -= embed_to_hid_weights_delta*learning_rate


		hid_to_output_weights_delta = momentum*hid_to_output_weights_delta + hid_to_output_weight_gradients/batch_size
		hid_to_output_weights -= hid_to_output_weights_delta*learning_rate

		output_bias_delta = momentum*output_bias_delta + output_bias_gradients/batch_size
		output_bias -= output_bias_delta*learning_rate

		hid_bias_delta = momentum*hid_bias_delta + hid_bias_gradients/batch_size
		hid_bias -= hid_bias_delta*learning_rate

		if m % show_validation_after ==0 and m != 0:
			embedding_layer_state, hidden_layer_state, output_layer_state = forward(valid_x,
				word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
			expanded_valid_target = expansion_matrix[valid_t]
			CE = - np.sum(expanded_valid_target*np.log(tiny + output_layer_state))/len(test_x)
			print("\t\tvalid CE: ", CE)

	print("\t\tEpoch ", epoch,  trainset_CE)

embedding_layer_state, hidden_layer_state, output_layer_state = forward(valid_x,
word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
expanded_valid_target = expansion_matrix[valid_t]
CE = - np.sum(expanded_valid_target*np.log(tiny + output_layer_state))/len(valid_x)
print("\t\tFinal valid CE: ", CE)

embedding_layer_state, hidden_layer_state, output_layer_state = forward(test_x,
word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
expanded_valid_target = expansion_matrix[test_t]
CE = - np.sum(expanded_valid_target*np.log(tiny + output_layer_state))/len(test_x)
print("\t\tFinal test  CE: ", CE)

#this step shows the distances between pairs of two words
print(distance('could', 'should'))
print(distance('could', 'can'))
print(distance('could', 'some'))
print(distance('could', 'the'))

#this methods print the data out to csv for later usage.
def toFile(file, arr):
	with open(file, 'w') as f:
		for i in range(len(arr)):
			strToWrite = ''
			for j in range(len(arr[i])):
				strToWrite += str(arr[i][j])
				if j!= len(arr[i]) -1:
					strToWrite += ', '
			if i != len(arr) -1:
				strToWrite += '\n'
			f.write(strToWrite)
def toFileSingle(file, arr):
	with open(file, 'w') as f:
		strToWrite = ''
		for i in range(len(arr)):
			strToWrite += str(arr[i])
			if i != len(arr)-1:
				strToWrite += ', '
		f.write(strToWrite)
toFile('word_embedding_weights.csv',word_embedding_weights)
toFile('embed_to_hid_weights.csv', embed_to_hid_weights)
toFile('hid_to_output_weights.csv', hid_to_output_weights)
toFileSingle('hidden_bias.csv', hid_bias)
toFileSingle('output_bias.csv', output_bias)
