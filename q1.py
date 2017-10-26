from __future__ import print_function

import sys
import os
import random
import numpy as np

INF = 1000000000
inpTrain,inpTest,inpVal = [],[],[]
labelTrain,labelTest,labelVal = [],[],[]
trainFile = sys.argv[1]
testFile = sys.argv[2]


def singleSample(b,n):
	total = 0
	w = np.zeros(n)
	while(True):
		minDis,maxDis = INF,0
		total +=1
		count = 0
		for x in inpTrain :
			g = np.dot(x,w)
			if g <= b : 
				np.add(w,x, out=w)
				count += 1
			minDis = min(minDis,g)
			maxDis = max(maxDis,g)
		if count == 0:
			break
		if(total > 500):
			break 
	# print("total",total)
	# return total,w,minDis,maxDis	
	return w
	

def batch(b,n):
	total = 0
	w = np.zeros(n)
	while(True):
		minDis,maxDis = INF,0
		total +=1
		count = 0
		toadd = np.zeros(n)
		for x in inpTrain :
			g = np.dot(x,w)
			if g <= b : 
				np.add(toadd,x,out=toadd)
				count += 1
			minDis = min(minDis,g)
			maxDis = max(maxDis,g)
		np.add(w,toadd,out=w)
		if count == 0:
			break
		if(total > 500):
			break 
	# print("total",total)
	return w
	return total,w,minDis,maxDis	

def readFile(arr,feat,lab,flag):
	# print(fd,len(lines))
	for line in arr :
		line = line.strip('\n')
		x = line.split(',')
		x = map(int,x)
		if(flag==0):
			y = -1+2*x[0]
			x[0] = 1
			lab.append(y)
		else: 
			x.insert(0,1)
		feat.append(x)
	feat = np.array(feat)
	lab = np.array(lab)
	return np.shape(feat)[1]

def predict(w):
	for i,x in enumerate(inpTest) :
		g = np.dot(x,w)
		if g > 0 :
			print(1)
		else :	
			print(0)
	return

def validate(w):
	total,count = 0,0
	for i,x in enumerate(inpVal):
		total += 1 
		g = np.dot(x,w) 
		if g > 0 and labelVal[i]==1:
			count+=1
		elif g < 0 and labelVal[i]==-1:
			count+=1
	return float(count*100)/float(total)

fd = open(trainFile,'r')
lines = fd.readlines()
random.shuffle(lines)
trainsize = int(len(lines)*0.8)
n = readFile(lines[0:trainsize],inpTrain,labelTrain,0)
n = readFile(lines[trainsize:],inpVal,labelVal,0)
fd.close()
fd = open(testFile,'r')
lines = fd.readlines()
n = readFile(lines,inpTest,labelTest,1)
fd.close()
inpTrain = np.transpose(inpTrain)
inpTrain = np.multiply(inpTrain,labelTrain)
inpTrain = np.transpose(inpTrain)
# print(np.shape(inpTrain),np.shape(inpTest),np.shape(labelTrain),np.shape(labelTest))
# print(n)


# epochs,param,closest,farthest = singleSample(0,n)
# # print(closest,farthest)
# # acc = predict(param)
# # print(acc)
# epochs,param,closest,farthest = singleSample(1000000,n)
# # print(closest,farthest)
# acc = predict(param)
# print(acc)
# epochs,param,closest,farthest = batch(0,n)
# # print(closest,farthest)
# acc = predict(param)
# print(acc)
# epochs,param,closest,farthest = batch(1000000,n)
# # print(closest,farthest)
# acc = predict(param)
# print(acc)
margin = 1000000
param = singleSample(0,n)
acc = validate(param)
# print(acc)
predict(param)
param = singleSample(margin,n)
acc = validate(param)
# print(acc)
predict(param)
param = batch(0,n)
acc = validate(param)
# print(acc)
predict(param)
param = batch(margin,n)
acc = validate(param)
# print(acc)
predict(param)