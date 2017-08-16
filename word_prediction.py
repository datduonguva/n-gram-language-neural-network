import numpy as np
wew = np.genfromtxt('wew.csv', delimiter=', ')
ethw = np.genfromtxt('ethw.csv', delimiter = ', ')
htow = np.genfromtxt('htow.csv', delimiter = ', ')
hb = np.genfromtxt('hb.csv', delimiter = ', ')
ob = np.genfromtxt('ob.csv', delimiter = ', ')
vocab = []
with open('vocab.csv', 'r') as f:
	for line in f.readlines():
		vocab = line.split(', ')[: -1]
words = []
for i in range(3):
	word = ''
	while word not in vocab:
		word = input('What is word ' + str(i)+ ' ')
		if word not in vocab:
			print('that word is not in dictionary')
	words.append(word)
#get the embedded_layer_state
def predict():
	els = []
	for i in range(3):
		els.append(wew[vocab.index(words[i])])
	els = np.array(els).ravel()

	ith = els.dot(ethw) + hb
	hls = 1.0/(1.0 + np.exp(-ith))

	its = hls.dot(htow) + ob
	its -= np.max(its)
	ols = np.exp(its)
	ols = ols/np.sum(ols)
	maxValue = 0.0
	maxPos = 0
	for i in range(250):
		if ols[i] > maxValue:
			maxValue = ols[i]
			maxPos = i
	predictedWord = vocab[maxPos]
	words.pop(0)
	words.append(predictedWord)
	return predictedWord
strToWrite = ''
for i in range(3):
	strToWrite += words[i] + ' '

for i in range(10):
	n = input()
	strToWrite+= predict() + ' '
	print(strToWrite)

