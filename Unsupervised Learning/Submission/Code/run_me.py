# Import modules

import numpy as np
from scipy import misc,sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math as math

pca_compression_rate = []
def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')

	return (data_x)

def read_faces():
	nFaces = 100
	nDims = 2500
	data_x = np.empty((0, nDims), dtype=float)

	for i in np.arange(nFaces):
		data_x = np.vstack((data_x, np.reshape(misc.imread('../../Data/Faces/face_%s.png' % (i)), (1, nDims))))

	return (data_x)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = 3
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def compute_recon_error(X,recon_X):
    return np.sqrt(np.mean(np.power(X-recon_X,2)))

def compute_comp_rate(X,codebook,k):
    # (#bytes to store all centroids + #Pixels*ceiling(log_2 k) )/(#bytes in original data_x)
    return (codebook.nbytes+400*400*math.ceil(math.log(k,2))/8)/X.nbytes
def sse(X,recon_X):
    return np.sum(np.power(X-recon_X,2))

def PCA(k,X):
    C = np.dot(X.T,X)
    value,vector=sparse.linalg.eigs(C,k=k)
    x_proj = np.dot(X,vector)
    x_recon = np.dot(x_proj,vector.T)
    cr = (x_proj.nbytes+vector.nbytes)/X.nbytes
    pca_compression_rate.append(cr)
    return np.array(x_recon,dtype=float)
if __name__ == '__main__':
	
	################################################
	# PCA

    data_x = read_faces()
    print('X = ', data_x.shape)

    print('Implement PCA here ...')
    image = data_x[1]
    
#    reconstructed_image3=np.reshape(PCA(3,data_x)[1],(50,50))
#    reconstructed_image5=np.reshape(PCA(5,data_x)[1],(50,50))
#    reconstructed_image10=np.reshape(PCA(10,data_x)[1],(50,50))
#    reconstructed_image30=np.reshape(PCA(30,data_x)[1],(50,50))
#    reconstructed_image50=np.reshape(PCA(50,data_x)[1],(50,50))
#    reconstructed_image100=np.reshape(PCA(100,data_x)[1],(50,50))
#    reconstructed_image150=np.reshape(PCA(150,data_x)[1],(50,50))
#    reconstructed_image300=np.reshape(PCA(300,data_x)[1],(50,50))
    reconstructed_image3=PCA(3,data_x)
    reconstructed_image5=PCA(5,data_x)
    reconstructed_image10=PCA(10,data_x)
    reconstructed_image30=PCA(30,data_x)
    reconstructed_image50=PCA(50,data_x)
    reconstructed_image100=PCA(100,data_x)
    reconstructed_image150=PCA(150,data_x)
    reconstructed_image300=PCA(300,data_x)
    
    plt.figure(1,figsize=(10,10))
    p1=plt.subplot(3, 3, 1)
    p1.imshow(image.reshape(50,50),cmap='gray')
    p1.set_title("original")
    
    p2=plt.subplot(3, 3, 2)
    p2.imshow(reconstructed_image3[1].reshape(50,50),cmap='gray')
    p2.set_title("k=3")
    
    p3=plt.subplot(3, 3, 3)
    p3.imshow(reconstructed_image5[1].reshape(50,50),cmap='gray')
    p3.set_title("k=5")
    
    p4=plt.subplot(3, 3, 4)
    p4.imshow(reconstructed_image10[1].reshape(50,50),cmap='gray')
    p4.set_title("k=10")
    
    p5=plt.subplot(3, 3, 5)
    p5.imshow(reconstructed_image30[1].reshape(50,50),cmap='gray')
    p5.set_title("k=30")
    p6=plt.subplot(3, 3, 6)
    p6.imshow(reconstructed_image50[1].reshape(50,50),cmap='gray')
    p6.set_title("k=50")
    p7=plt.subplot(3, 3, 7)
    p7.imshow(reconstructed_image100[1].reshape(50,50),cmap='gray')
    p7.set_title("k=100")
    p8=plt.subplot(3, 3, 8)
    p8.imshow(reconstructed_image150[1].reshape(50,50),cmap='gray')
    p8.set_title("k=150")
    p9=plt.subplot(3, 3, 9)
    p9.imshow(reconstructed_image300[1].reshape(50,50),cmap='gray')
    p9.set_title("k=300")
    plt.savefig("../Figures/pca_3x3.png")
    plt.show()
    
    recon_errors=[]
    recon_errors.append(compute_recon_error(data_x,reconstructed_image3))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image5))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image10))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image30))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image50))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image100))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image150))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image300))
    print("reconstruction errors:")
    print(recon_errors)
    

    print("compression rate:")
    print(pca_compression_rate)
	################################################
	# K-Means

    data_x = read_scene()
    print('X = ', data_x.shape)

    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Flattened image = ', flattened_image.shape)
    print('Implement k-means here ...')
    #2 5 10 25 50 75 100 200

    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(flattened_image)
    clusters2 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels2 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image2 = recreate_image(clusters2,labels2,400,400)
    plt.figure(1,figsize=(20,20))
    p1=plt.subplot(3, 3, 1)
    p1.imshow(data_x)
    p1.set_title("original")
    
    
    p2=plt.subplot(3, 3, 2)
    p2.imshow(reconstructed_image2/255)
    p2.set_title("2 clusters")
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(flattened_image)
    clusters5 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels5 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image5 = recreate_image(clusters5,labels5,400,400)
    
    p3=plt.subplot(3, 3, 3)
    p3.imshow(reconstructed_image5/255)
    p3.set_title("5 clusters")
    
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(flattened_image)
    clusters10 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels10 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image10 = recreate_image(clusters10,labels10,400,400)
    
    p4=plt.subplot(3, 3, 4)
    p4.imshow(reconstructed_image10/255)
    p4.set_title("10 clusters")
    
    kmeans = KMeans(n_clusters=25)
    kmeans.fit(flattened_image)
    clusters25 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels25 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image25 = recreate_image(clusters25,labels25,400,400)
    
    p5=plt.subplot(3, 3, 5)
    p5.imshow(reconstructed_image25/255)
    p5.set_title("25 clusters")
    
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(flattened_image)
    clusters50 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels50 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image50 = recreate_image(clusters50,labels50,400,400)
    
    p6=plt.subplot(3, 3, 6)
    p6.imshow(reconstructed_image50/255)
    p6.set_title("50 clusters")
    
    kmeans = KMeans(n_clusters=75)
    kmeans.fit(flattened_image)
    clusters75 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels75 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image75 = recreate_image(clusters75,labels75,400,400)
    
    p7=plt.subplot(3, 3, 7)
    p7.imshow(reconstructed_image75/255)
    p7.set_title("75 clusters")
    
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(flattened_image)
    clusters100 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels100 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image100 = recreate_image(clusters100,labels100,400,400)
    
    p8=plt.subplot(3, 3, 8)
    p8.imshow(reconstructed_image100/255)
    p8.set_title("100 clusters")
    
    kmeans = KMeans(n_clusters=200)
    kmeans.fit(flattened_image)
    clusters200 = np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
    labels200 = np.asarray(kmeans.labels_,dtype=np.uint8)    
    reconstructed_image200 = recreate_image(clusters200,labels200,400,400)
    
    p9=plt.subplot(3, 3, 9)
    p9.imshow(reconstructed_image200/255)
    p9.set_title("200 clusters")
    plt.savefig("../Figures/kmeans_3x3.png")
    plt.show()
    
    recon_errors=[]
    recon_errors.append(compute_recon_error(data_x,reconstructed_image2))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image5))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image10))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image25))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image50))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image75))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image100))
    recon_errors.append(compute_recon_error(data_x,reconstructed_image200))
    print("reconstruction errors:")
    print(recon_errors)
    
    compression_rate=[]
    compression_rate.append(compute_comp_rate(data_x,clusters2,2))
    compression_rate.append(compute_comp_rate(data_x,clusters5,5))
    compression_rate.append(compute_comp_rate(data_x,clusters10,10))
    compression_rate.append(compute_comp_rate(data_x,clusters25,25))
    compression_rate.append(compute_comp_rate(data_x,clusters50,50))
    compression_rate.append(compute_comp_rate(data_x,clusters75,75))
    compression_rate.append(compute_comp_rate(data_x,clusters100,100))
    compression_rate.append(compute_comp_rate(data_x,clusters200,200))
    print("compression rate:")
    print(compression_rate)
    
    x=[2,5,10,25,50,75,100,200]
    y=[]
    y.append(math.log(sse(data_x,reconstructed_image2)))
    y.append(math.log(sse(data_x,reconstructed_image5)))
    y.append(math.log(sse(data_x,reconstructed_image10)))
    y.append(math.log(sse(data_x,reconstructed_image25)))
    y.append(math.log(sse(data_x,reconstructed_image50)))
    y.append(math.log(sse(data_x,reconstructed_image75)))
    y.append(math.log(sse(data_x,reconstructed_image100)))
    y.append(math.log(sse(data_x,reconstructed_image200)))

    
    plt.figure(2, figsize=(6,4))
    plt.plot(x,y)
    plt.ylabel("Squared Error") #Y-axis label
    plt.xlabel("k") #X-axis label
    plt.xlim(-0.1,201) #set x axis range
    plt.ylim(0,20) #Set yaxis range
    plt.savefig("../Figures/4.png")
        #2 5 10 25 50 75 100 200

#
#    reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
#    print('Reconstructed image = ', reconstructed_image.shape)

