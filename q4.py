#!/usr/bin/env python

import sys
import random
import time
import math
import os
import numpy as np
from collections import *

"""Feel free to add any extra classes/functions etc as and when needed.
This code is provided purely as a starting point to give you a fair idea
of how to go about implementing machine learning algorithms in general as
a part of the first assignment. Understand the code well"""

INF = 100000000
modSize = defaultdict(long)
stopWords = {'': True, 'all': True, 'pointing': True, 'four': True, '/': True, 'go': True, 'oldest': True, 'seemed': True, 'whose': True, 'certainly': True, 'young': True, 'presents': True, 'to': True, 'asking': True, 'those': True, 'under': True, '@': True, 'far': True, 'every': True, '`': True, 'presented': True, 'did': True, 'turns': True, 'large': True, 'p': True, 'small': True, 'parted': True, 'smaller': True, 'says': True, 'second': True, 'further': True, 'even': True, 'what': True, '+': True, 'anywhere': True, 'above': True, 'new': True, ';': True, 'ever': True, 'full': True, 'men': True, 'here': True, 'youngest': True, 'let': True, 'groups': True, 'others': True, 'alone': True, 'along': True, 'great': True, 'k': True, '{': True, 'put': True, 'everybody': True, 'use': True, 'from': True, 'working': True, '&': True, 'two': True, 'next': True, 'almost': True, 'therefore': True, 'taken': True, 'until': True, 'today': True, 'more': True, 'knows': True, 'clearly': True, 'becomes': True, 'it': True, 'downing': True, 'everywhere': True, 'known': True, 'cases': True, 'must': True, 'me': True, 'states': True, 'room': True, 'f': True, 'this': True, 'work': True, 'itself': True, 'can': True, 'mr': True, 'making': True, 'my': True, 'numbers': True, 'give': True, 'high': True, 'something': True, 'want': True, '!': True, 'needs': True, 'end': True, 'turn': True, 'rather': True, 'how': True, 'y': True, 'may': True, 'after': True, 'such': True, 'man': True, 'a': True, 'q': True, ')': True, 'so': True, 'keeps': True, 'order': True, 'furthering': True, 'over': True, 'years': True, 'ended': True, 'through': True, 'still': True, 'its': True, 'before': True, 'group': True, 'somewhere': True, 'interesting': True, ',': True, 'better': True, 'differently': True, 'might': True, '<': True, 'then': True, 'non': True, 'good': True, 'somebody': True, 'greater': True, 'downs': True, 'they': True, 'not': True, 'now': True, '\\': True, 'gets': True, 'always': True, 'l': True, 'each': True, 'went': True, 'side': True, 'everyone': True, 'year': True, 'our': True, 'out': True, 'opened': True, 'since': True, 'got': True, 'shows': True, 'turning': True, 'differ': True, 'quite': True, 'members': True, 'ask': True, 'wanted': True, 'g': True, 'could': True, 'needing': True, 'keep': True, 'thing': True, 'place': True, 'w': True, 'think': True, 'first': True, 'already': True, 'seeming': True, '*': True, 'number': True, 'one': True, 'done': True, 'another': True, 'open': True, 'given': True, '"': True, 'needed': True, 'ordering': True, 'least': True, 'anyone': True, 'their': True, 'too': True, 'gives': True, 'interests': True, 'mostly': True, 'behind': True, 'nobody': True, 'took': True, 'part': True, 'herself': True, 'than': True, 'kind': True, 'b': True, 'showed': True, 'older': True, 'likely': True, 'r': True, 'were': True, 'toward': True, 'and': True, 'sees': True, 'turned': True, 'few': True, 'say': True, 'have': True, 'need': True, 'seem': True, 'saw': True, 'orders': True, 'that': True, '-': True, 'also': True, 'take': True, 'which': True, 'wanting': True, '=': True, 'sure': True, 'shall': True, 'knew': True, 'wells': True, 'most': True, 'nothing': True, ']': True, 'why': True, 'parting': True, 'noone': True, 'later': True, 'm': True, 'mrs': True, 'points': True, '}': True, 'fact': True, 'show': True, 'ending': True, 'find': True, '(': True, 'state': True, 'should': True, 'only': True, 'going': True, 'pointed': True, 'do': True, 'his': True, 'get': True, 'cannot': True, 'longest': True, 'during': True, 'him': True, 'areas': True, 'h': True, 'she': True, 'x': True, 'where': True, 'we': True, 'see': True, 'are': True, 'best': True, '#': True, 'said': True, 'ways': True, 'away': True, 'enough': True, 'smallest': True, 'between': True, 'across': True, 'ends': True, 'never': True, 'opening': True, 'however': True, 'come': True, 'both': True, 'c': True, 'last': True, 'many': True, 'against': True, 's': True, 'became': True, 'faces': True, 'whole': True, 'asked': True, 'among': True, 'point': True, 'seems': True, 'furthered': True, '[': True, 'furthers': True, 'puts': True, 'three': True, 'been': True, '.': True, 'much': True, 'interest': True, '>': True, 'wants': True, 'worked': True, 'an': True, 'present': True, '^': True, 'case': True, 'myself': True, 'these': True, 'n': True, 'will': True, 'while': True, '~': True, 'would': True, 'backing': True, 'is': True, 'thus': True, 'them': True, 'someone': True, 'in': True, 'if': True, 'different': True, 'perhaps': True, 'things': True, 'make': True, 'same': True, 'any': True, 'member': True, 'parts': True, 'several': True, 'higher': True, 'used': True, 'upon': True, 'uses': True, 'thoughts': True, 'off': True, 'largely': True, 'i': True, 'well': True, 'anybody': True, 'finds': True, 'thought': True, 'without': True, 'greatest': True, 'very': True, 'the': True, 'yours': True, 'latest': True, 'newest': True, 'just': True, 'less': True, 'being': True, 'when': True, 'rooms': True, 'facts': True, 'yet': True, '$': True, 'had': True, 'lets': True, 'interested': True, 'has': True, 'gave': True, 'around': True, 'big': True, 'showing': True, 'possible': True, 'early': True, 'know': True, 'like': True, 'necessary': True, 'd': True, 't': True, 'fully': True, 'become': True, 'works': True, 'grouping': True, 'because': True, 'old': True, 'often': True, 'some': True, 'back': True, 'thinks': True, 'for': True, 'though': True, 'per': True, 'everything': True, 'does': True, '?': True, 'either': True, 'be': True, 'who': True, 'seconds': True, 'nowhere': True, 'although': True, 'by': True, '_': True, 'on': True, 'about': True, 'goods': True, 'asks': True, 'anything': True, 'of': True, 'o': True, '|': True, 'or': True, 'into': True, 'within': True, 'down': True, 'beings': True, 'right': True, 'your': True, 'her': True, 'area': True, 'downed': True, 'there': True, 'long': True, 'way': True, ':': True, 'was': True, 'opens': True, 'himself': True, 'but': True, 'newer': True, 'highest': True, 'with': True, 'he': True, 'made': True, 'places': True, 'whether': True, 'j': True, 'up': True, 'us': True, 'problem': True, 'z': True, 'clear': True, 'v': True, 'ordered': True, 'certain': True, 'general': True, 'as': True, 'at': True, 'face': True, 'again': True, '%': True, 'no': True, 'generally': True, 'backs': True, 'grouped': True, 'other': True, 'you': True, 'really': True, 'felt': True, 'problems': True, 'important': True, 'sides': True, 'began': True, 'younger': True, 'e': True, 'longer': True, 'came': True, 'backed': True, 'together': True, 'u': True, 'presenting': True, 'evenly': True, 'having': True, 'once': True, '</s>':True, '<s>':True }


def calMetrics(cm,total) :
	prec,recall = 0,0
	for i in xrange(len(classes)):
		rowSum = np.sum(cm[i,:])
		colSum = np.sum(cm[:,i])
		prec+=float(cm[i][i])/rowSum 
		recall+=float(cm[i][i])/colSum
	prec = prec/len(classes) 
	recall = recall/len(classes)
	return prec,recall,float(2*(prec*recall))/(prec+recall)


def findDis(featVec1,featVec2):
	dis,len1,len2 = 0,0,0
	for word in featVec1 :
		# print(word,featVec1[word],featVec2[word])
		dis += float(featVec1[word]*featVec2[word])
		len1 += featVec1[word]**2 
		len2 += featVec2[word]**2
	len1 = math.sqrt(len1)
	len2 = math.sqrt(len2)
	if(dis==0):
		return 0
	return float(dis)/(len1*len2)

def findKNN(featVec,k):
	maxD = 0
	label = "hello"
	fileList = {}
	topK = []
	labelCount=defaultdict(int)
	for filename in featuresTrain :
		# print("finding distance with file ",filename)
		# print(filename)
		dis = findDis(featVec,featuresTrain[filename])
		# print(dis)
		fileList[filename]=dis
	count=0
	for filename in sorted(fileList,key=fileList.get,reverse=True):
		topK.append(filename)
		count+=1
		if(count==k):
			break
	for filename in topK :
		labelCount[labelTrain[filename]]+=1
	for labelName in sorted(labelCount,key=fileList.get,reverse=True):
		return labelName
	return label


def predict():
	k=1
	for filename in featuresTest :
		pred = findKNN(filename,k)
		print(pred[0:-1]) 

def validate(k):
	count = 0
	total = 0
	cm = [[0 for i in xrange(len(classes))] for i in xrange(len(classes))]
	for filename in featuresVal :
		pred = findKNN(featuresVal[filename],k)
		cm[classes[pred]-1][classes[labelVal[filename]]-1]+=1
		if pred==labelVal[filename] :
			count+=1
		total +=1 
	return float(count*100)/total , np.array(cm), total

def readFiles(idir,arr,feat,lab,c):
	for filename in arr :
		fileDict = defaultdict(int)
		fd = open(idir+c+filename,'r')
		lines = fd.readlines()
		for line in lines :
			line.strip()
			words = line.split(" ")
			for word in words :
				if word not in stopWords :
					fileDict[word]+=1
		feat[str(c+filename)] = fileDict
		lab[str(c+filename)] = c
		fd.close()

def readTestFiles(idir,arr,feat):
	for filename in arr :
		fileDict = defaultdict(int)
		fd = open(idir+c+filename,'r')
		lines = fd.readlines()
		for line in lines :
			line.strip()
			words = line.split(" ")
			for word in words :
				if word not in stopWords :
					fileDict[word]+=1
		feat.append(fileDict)
		fd.close()

if __name__ == '__main__':
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	classes = {'galsworthy/':1,'galsworthy_2/':2,'mill/':3,'shelley/':4,'thackerey/':5,'thackerey_2/':6,'wordsmith_prose/':7,'cia/':8,'johnfranklinjameson/':9,'diplomaticcorr/':10}
	featuresTrain,featuresVal,featuresTest = {},{},[]
	labelTrain,labelVal = {},{}
	# print('Making the feature vectors.')

	for c in classes : 
		listing = os.listdir(trainFile+c)
		random.shuffle(listing)
		trainsize = int(len(listing)*0.8)
		readFiles(trainFile,listing[0:trainsize],featuresTrain,labelTrain,c)
		readFiles(trainFile,listing[trainsize:],featuresVal,labelVal,c)
	
	t0=time.time()
	for c in sorted(classes,key=classes.get) :
		listing = os.listdir(testFile+c)
		readTestFiles(testFile,listing,featuresTest)

	# print('Finished making features.')
	# print('Statistics ->')

	k=1
	# for k in xrange(1,0,-2):
	# 	acc,cm,total = validate(k)
	# 	print("Accuracy for k =",k,"is : ",acc)
	# p,r,f1 = calMetrics(cm,total)
	# print("Precision : ",p)
	# print("Recall : ",r)
	# print("F1 score : ",f1)
	# print(cm)
	predict()
	t1 =  time.time()
	# print("time :",t1-t0)

	
