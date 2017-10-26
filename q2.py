from __future__ import print_function

import sys
import os
import math
import random
import numpy as np

INF = 1000000000
inpTrain,inpTest,inpVal = [],[],[]
labelTrain,labelTest,labelVal = [],[],[]
trainFile = sys.argv[1]
testFile = sys.argv[2]


def modifiedPer(b,n):
	epoch = 0
	w = np.zeros(n)
	while(True):
		minDis,maxDis = INF,0
		epoch += 1
		wrong,total = 0,0
		for x in inpTrain :
			total+=1
			g = np.dot(x,w)
			if g <= b :
				wrong += 1
				neta = 1 - ((total-wrong)/total)
				toadd = neta*x*0.01
				np.add(w,x,out=w)
			minDis = min(minDis,g)
			maxDis = max(maxDis,g)
		# if epoch < 6 or epoch > 9996 :
		# 	print(epoch,100-float(wrong*100)/total)
		if wrong == 0:
			break
		if(epoch > 10000):
			break 
	# print("total",total)
	# return total,w,minDis,maxDis	
	return w,minDis,maxDis



def predict(w):
	for i,x in enumerate(inpTest) :
		g = np.dot(x,w)
		if g > 0 :
			print(4)
		else :	
			print(2)
	return 
	
def validate(w) :
	total,count = 0,0
	for i,x in enumerate(inpVal) :
		total += 1
		g = np.dot(x,w)
		if g>0 and labelVal[i]==1 : 
			count+=1
		elif g<0 and labelVal[i]==-1 :
			count+=1 	
	return float(count*100)/float(total)


def neta(num):
	return 0.01

def trainFunc(b,n):
	epoch = 0
	w = np.zeros(n)
	while(True):
		minDis,maxDis = INF,0
		epoch +=1
		Jfunc = np.zeros(n)
		wrong,total = 0,0
		for x in inpTrain :
			total += 1
			g = np.dot(x,w)
			if g < b :
				wrong +=1 
				# print(g,b)
				tmp1 = float(b-g)
				tmp2 = np.linalg.norm(x)**2
				tmp = tmp1/tmp2
				# print(tmp1,tmp2,tmp)
				toadd = neta(total)*tmp*x
				# print(toadd)  
				np.add(Jfunc,toadd,out=Jfunc)
				# print(Jfunc)
			minDis = min(minDis,g)
			maxDis = max(maxDis,g)
		# if epoch < 6 or epoch > 9996 :
		# 	print(epoch,100-float(wrong*100)/total) 
		np.add(w,Jfunc,out=w)
		if(epoch > 10000):
			break 
	# print("total",total)
	# return w
	return w,minDis,maxDis	


def readFile(arr,feat,lab,flag):
	for line in arr :
		line = line.strip('\n')
		x = line.split(',')
		for i,tmp in enumerate(x) :
			if tmp=='?' :
				x[i] = 0
		x = map(int,x)
		x[0]=1
		if flag==0 :
			y = x[10]-3
			lab.append(y)
		feat.append(x[0:10])
	feat = np.array(feat)
	lab = np.array(lab)
	return np.shape(feat)[1]

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
# print(n)
margin = 10
inpTrain = np.transpose(inpTrain)
inpTrain = np.multiply(inpTrain,labelTrain)
inpTrain = np.transpose(inpTrain)
# print("Relaxation + Margin")
param,minMar,maxMar = trainFunc(margin,n)
# print(param,minMar,maxMar)
acc = validate(param)
# print(acc)
predict(param)
# print("Modified Perceptron")
param,minMar,maxMar = modifiedPer(0,n)
# print(param,minMar,maxMar)
acc = validate(param)
# print(acc)
predict(param)
# print(np.shape(inpTrain),np.shape(inpTest),np.shape(labelTrain),np.shape(labelTest),np.shape(inpVal),np.shape(labelVal))
# print(inpTrain[0:20])
