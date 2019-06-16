#EE18MTECH11002
#Principal Component Analysis using Inbuilt Function

#-----Required Packages-----#
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import csv
#import pandas as pd

#-----Time Requirements-----#
start_time =time.time()

#-----Loading Data from .mat file-----#
mat_contents = sio.loadmat('data_all.mat')
X_data = mat_contents['data_all']

#X_data = pd.read_csv('data.csv')

#-----Standardizing the features-----#
X_data = StandardScaler().fit_transform(X_data)

#-----User giving Dimension-----#
#comp = int(input("Enter no of principle components: "))

#-----PCA-----#

pca = PCA(n_components=2)
X_comp = pca.fit_transform(X_data)

#-----2D Plot-----#
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X_comp[:, 0], X_comp[:, 1],marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2') 
plt.title('PCA 2D Plot')

pca = PCA(n_components=3)
X_comp = pca.fit_transform(X_data) 
      
#-----3D Plot-----# c is for color
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter3D(X_comp[:, 0], X_comp[:, 1],X_comp[:,2], marker='o')
ax.set_xlabel('Prin. Comp 1')
ax.set_ylabel('Prin. Comp 2')
ax.set_zlabel('Prin. Comp 3')
ax.set_title('PCA 3D Plot')

#-----Print Time Required-----#
print("--- %s seconds ---" % (time.time() - start_time))