from __future__ import print_function
from sets import Set
import math
import sys
import random

trainFile = sys.argv[1]
testFile = sys.argv[2]
INF = 100
train,val,test = [],[],[]
decision = {}
leaf = {}
depDict = {'management': 0, 'product_mng': 1, 'technical': 2, 'marketing': 3, 'support': 4, 'sales': 5, 'RandD': 6, 'IT': 7, 'hr': 8, 'accounting': 9}
salDict = {'high': 0, 'medium': 1, 'low': 2}
SatLevel = [1,2,3,4,5,6,7,8,9,10]
LastEval = [4,5,6,7,8,9,10]
NProject = [2,3,4,5,6,7]
AvgMoHr = [1,2,3,4,5,6,7,8,9,10,11]
TimeSpend = [2,3,4,5,6,7,8,9,10]
WorkAcc = [0,1]
Prom = [0,1]
LeftCom = [0,1]

def SatRange(satLevel):
	return int(satLevel*10+0.49)

def EvalRange(lastEval):
	return int(lastEval*10+0.49)

def AMHRange(amh):
	return int((amh+9)/20 - 4)

def calLog(num):
	if(num==0):
		return 0
	return -num*math.log(num,2)

def entropy(arr):
	if(arr[0]==0 and arr[1]==0):
		return INF
	a = float(arr[0])/(arr[0]+arr[1])
	b = 1-a
	# print(a,b)
	return calLog(a)+calLog(b)

def qualCalcLeq(dataset,ind,val):
	# print(ind,val)
	left = [0,0]
	right = [0,0]
	for x in dataset :
		if(x[ind]<=val):
			left[x[6]]+=1
		else:
			right[x[6]]+=1
	# print(left,right)
	return entropy(left)+entropy(right)

def qualCalcEq(dataset,ind,val):
	# print(ind,val)
	left = [0,0]
	right = [0,0]
	for x in dataset :
		if(x[ind]==val):
			left[x[6]]+=1
		else:
			right[x[6]]+=1
	# print(left,right)
	return entropy(left)+entropy(right)

def checkQuality(dataset):
	minEnt,categ,value = 100,0,0
	for i,dec in enumerate(attr) :
		for val in dec :
			if(i<5):
				qual = qualCalcLeq(dataset,i,val)
			elif i!=6 :
				qual = qualCalcEq(dataset,i,val)
			# print(i,val,qual)
			if(qual < minEnt):
				minEnt = qual
				categ = i
				value = val
	return categ,value

def seperateEq(dataset,ind,val):
	left = []
	right = []
	for x in dataset :
		# print("ind is ",ind)
		if x[ind]==val :
			left.append(x)
		else:
			right.append(x)
	return left,right


def seperateLeq(dataset,ind,val):
	left = []
	right = []
	for x in dataset :
		if x[ind]<=val:
			left.append(x)
		else:
			right.append(x)
	# print(len(left),len(right))
	return left,right 

def maxLab(dataset):
	cntzero,cntone=0,0
	for x in dataset : 
		if x[6]==0 :
			cntzero +=1
		else :
			cntone +=1
	if(cntone > cntzero):
		return 1
	return 0

def TotEntropy(dataset) :
	pos,neg = 0,0 
	if len(dataset)==0 :
		return 0
	for x in dataset :
		if x[6]==0 :
			neg+=1 
		else :
			pos+=1
	# if len(dataset) > 8800 :
	# 	print("Total values are ",pos,neg)
	return entropy([pos,neg])


def buildTree(dataset,nodeNo,totEntr,depth):
	# print("In node : ",nodeNo)
	if len(dataset)==0 :
		return
	if len(dataset) < 50 or totEntr < 0.1 :
		# print("breaking")
		# print(TotEntropy(dataset))
		leaf[nodeNo] = maxLab(dataset)
		# print(leaf[nodeNo],type(leaf[nodeNo]))
		return
	# if(depth > 5):
	# 	print("too deep")
	# 	return
	query,val = checkQuality(dataset)
	# print("Final : ",query,val)
	decision[nodeNo] = [query,val]
	if(query < 5):
		lchild, rchild = seperateLeq(dataset, query, val)
	elif(query !=6):
		lchild, rchild = seperateEq(dataset, query, val)
	lEntr = TotEntropy(lchild)
	rEntr = TotEntropy(rchild)
	# print("Entropies of two child and total set are ",lEntr,rEntr,totEntr)
	# if totEntr - lEntr - rEntr < 0.05 : 
	buildTree(lchild,2*nodeNo+1,lEntr,depth+1)
	buildTree(rchild,2*nodeNo+2,rEntr,depth+1)

def predict(inp,nodeNo):
	# print("at node : ",nodeNo)
	if nodeNo in leaf :
		# print("returning ",leaf[nodeNo])
		return leaf[nodeNo]
	query,val = decision[nodeNo]
	# print("query,val",query,val)
	if(query < 5):
		# print("its value is ",inp[query])
		if(inp[query] <= val):
			return predict(inp,2*nodeNo+1)
		return predict(inp,2*nodeNo+2)
	if(query!=6):
		# print("its value is ",inp[query])
		if(inp[query] == val):
			return predict(inp,2*nodeNo+1)
		return predict(inp,2*nodeNo+2)

def validate(dataset):
	total = len(dataset)
	count = 0
	for x in dataset : 
		lab = predict(x,0)
		if lab == x[6] :
			count+=1
	return float(count)*100/float(total)

def guess(dataset):
	for x in dataset : 
		lab = predict(x,0)
		print(lab)
	return 

def readFile(arr,feat,flag):
	for line in arr :
		line = line.strip()
		x = line.split(',')
		if(flag==1):
			x.insert(6,0)
		for i in xrange(8):
			x[i] = float(x[i])
		for i in xrange(2,8):
			x[i] = int(x[i])
		x[0] = SatRange(x[0])
		x[1] = EvalRange(x[1])
		x[3] = AMHRange(x[3])
		x[8] = depDict[x[8]]
		x[9] = salDict[x[9]]
		feat.append(x)

fd = open(trainFile,'r')
lines = fd.readlines()
# random.shuffle(lines[1:])
trainsize = int(len(lines)*0.8)
readFile(lines[1:trainsize],train,0)
readFile(lines[trainsize:],val,0)
fd.close()
fd = open(testFile,'r')
lines = fd.readlines()
readFile(lines[1:],test,1)
fd.close()
Dep = [i for i in xrange(10)]
Sal = [i for i in xrange(3)]
attr = [SatLevel,LastEval,NProject,AvgMoHr,TimeSpend,WorkAcc,LeftCom,Prom,Dep,Sal]
# print(attr,depDict,salDict)
# print(train[0:50]
initialEntr = TotEntropy(train)
buildTree(train,0,initialEntr,0)
# for i in sorted(leaf):
# 	print(i,":",leaf[i],end=" ",sep="")
# print(len(leaf),len(decision))
acc = validate(val)
# print(acc)
guess(test)
# print(accuracy)