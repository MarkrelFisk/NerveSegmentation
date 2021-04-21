import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import maxflow

#%% Load and visualize

path = 'C:/Users/mark_/Documents/DTU/Kandidat/1.semester/Advanced Image Analysis/Project/'
I = skimage.io.imread(path + 'nerves_part.tiff')

fig, ax = plt.subplots(1,2)
for i in range(2):
    ax[i].imshow(I[i+200,:,:])
    
fig,ax = plt.subplots(1,2)
ax[0].imshow(I[0,:,:]/(2**8 - 1))
ax[1].hist(I[0,:,:]/(2**8 - 1))
#%%

mu = np.array([0.25,0.55]) # Mean value is corresponding to the two 'peaks' in histogram 
beta = 0.001 # should be pretty low for this example

seg = []

for i in range(10):
    im = I[i,:,:]/(2**8 - 1)
    d = im.ravel()
    w_s = (d-mu[0])**2 
    w_t = (d-mu[1])**2 
    N = len(d) 
    
    g = maxflow.Graph[float]()
    
    nodes = g.add_nodes(N)
    
    for ii in range(N-1):
        g.add_edge(nodes[ii], nodes[ii+1], beta, beta)
    
    for ii in range(N):
        g.add_tedge(nodes[ii], (d[ii]-mu[1])**2, (d[ii]-mu[0])**2)
    
    flow = g.maxflow()
    print(f'Maximum flow: {flow}')
    
    labeling = [g.get_segment(nodes[ii]) for ii in range(N)]
    
    # Create segmentation with correct dimension 
    segmentation = np.array(labeling).reshape(im.shape[0],im.shape[1]) 
    seg.append(segmentation)
    
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(im)
    ax[1].imshow(segmentation, cmap = plt.cm.gray)
    
