import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import maxflow

#%% Load and visualize

path = 'C:/Users/mark_/Documents/DTU/Kandidat/1.semester/Advanced Image Analysis/Project/'
slices = skimage.io.imread(path + 'nerves_part.tiff').astype(float)/(2**8-1)

#%% Inspect the image and the histogram
I = slices[0,:,:]
fig, ax = plt.subplots()
ax.imshow(I, cmap=plt.cm.gray)
plt.axis('off')

#%%
edges = np.linspace(0, 1, 349)
fig, ax = plt.subplots()
ax.hist(slices.ravel(),edges)
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('intensity histogram')

#%% Define likelihood
mu = np.array([0.3, 0.55])
U = np.stack([(I-mu[i])**2 for i in range(len(mu))],axis=2)
S0 = np.argmin(U,axis=2)

fig, ax = plt.subplots()
ax.imshow(S0,cmap=plt.cm.gray)
ax.set_title('max likelihood')
plt.axis('off')
#%% Define prior, construct graph, solve
beta  = 0.01
mu = np.array([0.3, 0.55])
S = []
for i in range(len(slices)):
    I = slices[i,:,:]
    U = np.stack([(I-mu[i])**2 for i in range(len(mu))],axis=2)
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(I.shape)
    g.add_grid_edges(nodeids, beta)
    g.add_grid_tedges(nodeids, U[:,:,1], U[:,:,0])
    
    # solving
    g.maxflow()
    S.append(g.get_grid_segments(nodeids))


#%%
idx = 0

fig, ax = plt.subplots()
ax.imshow(S[idx],cmap=plt.cm.magma)
#ax.set_title('max posterior')
plt.axis('off')

#%% Plot voxels

from matplotlib import cm
#%matplotlib inline
%matplotlib qt

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    #cube = normalize(cube)
    
    facecolors = cm.magma(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    z, y, x = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim([-IMG_DIM*2,IMG_DIM*2])
    ax.set_ylim([-IMG_DIM*2,IMG_DIM*2])
    ax.set_zlim([0,IMG_DIM*2])
    
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.axis('off')

    plt.show()

arr = (~np.asarray(S)).astype('float')

IMG_DIM = 100

from skimage.transform import resize
resized = resize(arr,(IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')

plot_cube(resized[:1,::-1,::-1], angle=180) # one voxel slice
# Plot all slices: 
# plot_cube(resized[::-1,::-1,::-1], angle=180)



