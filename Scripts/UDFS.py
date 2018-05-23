import argparse
import numpy as np
import scipy as sp
import scipy.optimize
from scipy.sparse import coo_matrix
from numpy import linalg as LA
import pandas as pd
import numpy.matlib
import os
import sys, getopt

if __name__ == '__main__':
    usage = "prog  [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input training file containing features and classes")
    parser.add_argument('-k', '--k', help="input k value ( default: k = 1 )")
    parser.add_argument('-g', '--gamma', help="input gamma value ( default: g = 0.00001 )")
    parser.add_argument('-l', '--lamda', help="input lamda value ( default: l = 0.00001 )")
    parser.add_argument('-n', '--nclasses', help="input number of classes in the training set ")
    parser.add_argument('-o', '--output', help="Output filename containing feature ranking and weights")
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()


df1 = pd.read_csv(args.input)
Xtr1 = np.asarray(df1.iloc[:, :-1])
Ytr = np.asarray(df1.iloc[:,-1])

Xtr = Xtr1.T

#print Xtr

k=int(args.k)
gammaCandi = float(args.gamma)
lamdaCandi = float(args.lamda)
n = int(args.nclasses)

class new_class():
 def __init__(self,number,float):
  self.k=int(number)
  self.gamma=float
  self.lamda=float

a = new_class(k,lamdaCandi)
para1 = a.__dict__
para2 = a.__dict__.keys()

def LocalDisAna(X, a1, a2):
 dn = X.shape
 if a2[0] == 'k':
  k = a1['k']+1
 else:
  k = 16
 if a2[1] == 'lamda':
  lamda = a1['lamda']
 else:
  lamda = 1000
 s=(k,k)
 np.ones(s)
 Lc = np.identity(k) - 0.5*np.ones(s)
 lidxArr1 = []
 aiArr = []
 aiArr1 = []
 row = []
 row1 = []
 col = []
 col1 = []
 for i in range(0,dn[1]):
  lidxArr2 = []
  sort_index = []
  z = np.matlib.repmat(np.vstack(X[:,i]),1,dn[1]) - X
  x1 = np.square(z)
  z1 = x1.sum(axis=0).reshape(1, -1)
  sort_index = np.argsort(z1[0])
  nnidx = sort_index
  #print nnidx
  nn1 = nnidx[0:k]
  Xi = X[:, [nn1[0], nn1[1]]]
  #print nn1[0]
  #print nn1[1]
  a1 = Xi[:,0]*Lc[0][0]
  a2 = Xi[:,1]*Lc[0][1]
  a3 = Xi[:,0]*Lc[1][0]
  a4 = Xi[:,1]*Lc[1][1]
  v1 = np.add(a1,a2)
  v2 = np.add(a3,a4)
  v3 = np.c_[v1,v2]
  if dn[0] > k:
   Ai = np.linalg.inv(np.dot(lamda,np.eye(k)) + np.dot(v3.T,v3))
   Ai = np.dot(Lc,Ai)
  else:
   s1 = np.dot(v3,v3.T)
   s2 = np.dot(lamda,s1)
   e1 = np.eye(dn[0])
   e2 = np.linalg.inv(e1+s2)
   s3 = np.dot(e2,v3)
   s4 = np.dot(v3.T,s3)
   s5 = np.dot(lamda,s4)
   s6 = Lc - s5
  lidxArr2.append(((i+1)-1)*k+1)
  lidxArr2.append(((i+1)-1)*k+k)
  lidxArr1.append(lidxArr2)
  aiArr.append(Ai[0][0])
  aiArr.append(Ai[1][0])
  aiArr.append(Ai[0][1])
  aiArr.append(Ai[1][1])
  row.append(lidxArr2[0]-1)
  row.append(lidxArr2[0])
  row.append(lidxArr2[0]-1)
  row.append(lidxArr2[0])
  col.append(lidxArr2[0]-1)
  col.append(lidxArr2[0]-1)
  col.append(lidxArr2[0])
  col.append(lidxArr2[0])
  row1.append(nn1[0])
  row1.append(nn1[1])
  #col1.append(((i+1)-1)*k+1)
  #col1.append(((i+1)-1)*k+k)
  aiArr1.append(1)
  aiArr1.append(1)  
  x1 = ((i))*k
  x2 = ((i))*k+k-1
  col1.append(x1)
  col1.append(x2)
 co = coo_matrix((aiArr, (row, col)), shape=(dn[1]*k, dn[1]*k)).toarray()
 co1 = coo_matrix((aiArr1, (row1, col1)), shape=(dn[1], dn[1]*k)).toarray()
 x1 = np.dot(co1,co)
 X = np.dot(x1,co1.T)   
 return X;

x2 = LocalDisAna(Xtr,para1,para2)

#print x2

e1 = np.dot(Xtr,x2)
e2 = np.dot(e1,Xtr.T)

#print e2
 
def fs_unsup_udfs(A, k, r):
 NIter = 20;
 an = A.shape
 d1 = np.ones((an[1],1))
 for iter in range(0,NIter):
  d2 = np.diag(d1.T[0])
  m = A + np.dot(r,d2)
  m1 = np.maximum(m,m.T)
  w, v = LA.eigh(m1) #Calculate Eigenvalues and Eigenvectors
  sort_index1 = np.argsort(w)
  x1 = v[:, 0:k]
  eps = 2.2204e-16
  x2 = np.multiply(x1,x1)
  x3 = np.sum(x2,axis=1)
  x4 = np.sqrt(x3 + eps)
  x5 = np.divide(0.5,x4)
  x61 = np.dot(x1.T,e2)
  x62 = np.dot(x61,x1)
  x63 = np.trace(x62)
  #x64 = x63 + np.dot(a1['gamma'],x4)
  x64 = x63 + np.dot(r,x4)
 w1 = [ -x for x in x1[:,0:1].T]
 w2 = x1[:,1:2].T
 A = np.c_[w1[0],w2[0]]
 return A;

w = fs_unsup_udfs(e2,n,gammaCandi)

p1 = np.multiply(w,w)
p2 = np.sum(p1,axis=1)
r1 = np.argsort(p2)
r2 = np.flipud(r1)
rank = [x+1 for x in r2]
weight = sorted(p2,reverse=True)

print rank
print weight
