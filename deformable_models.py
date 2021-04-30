

#Project
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import maxflow
import simple_snake as sis
#import simple_snake2 as sis
import scipy
import scipy.ndimage

path = 'Data/'

#%% 
I = skimage.io.imread(path + 'nerves_part.tiff')
fig, ax = plt.subplots(1,5, figsize=(12,8))

for i in range(5):
    ax[i].imshow(I[i*100,:,:])


#%% Multiple
im_full = skimage.io.imread(path + 'nerves_part.tiff').astype(np.float)/255
im = im_full[0,:,:]
# sigma = 3
# im_g = scipy.ndimage.gaussian_filter(im, sigma, mode='nearest')
#%%
centers = np.array([[215, 170], [225, 90], [260, 130]])
## Circle outside 
#radius = [0.05*np.mean(I.shape), 0.04*np.mean(I.shape), 0.04*np.mean(I.shape)]
radius = 0.04*np.mean(I.shape)
fig, ax = plt.subplots(1,2)
ax[0].imshow(im)
ax[0].scatter([170, 90, 130], [215, 225, 260], color = 'r')
ax[0].scatter(183, 93, color = 'r')
ax[0].scatter(153, 64, color = 'g')
# ax[1].imshow(im_g)
# ax[1].scatter([170, 90, 130], [215, 225, 260], color = 'r')

#%%  First slice
import simple_snake as sis
nr_points = 100
nr_iter = 25
step_size = 3
alpha = 0.6
beta = 0.5

B = sis.regularization_matrix(nr_points, alpha, beta)

snakes = []
for i in centers:
    snake = sis.make_circular_snake(nr_points, i, radius)
    snakes.append(snake)
    

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.set_title('Initialization')


fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snakes[0,0]],'r-')

#%%
## Plot all
for i in range(nr_iter):
    fig, ax = plt.subplots()
    for j in range(3):
        snakes[j] = sis.evolve_snake(snakes[j], im, B, step_size)

    ax.clear()
    ax.imshow(im, cmap=plt.cm.gray)
    ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
    ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
    ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
    ax.set_title(f'iteration {i}')
    plt.pause(0.001)


#%%
## Plot last
for i in range(nr_iter):
    for j in range(3):
        snakes[j] = sis.evolve_snake(snakes[j], im, B, step_size)

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.set_title(f'iteration {i}')

#%% Tracking through structure
#cent1 = np.array([215, 170])
#cent2 = np.array([225, 90])
#cent3 = np.array([260, 130])
cent4 = np.array([260, 285])



snakes3D = []

## Plot last
snakes = sis.make_circular_snake(nr_points, cent4, radius)

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[1],snakes[1,0]],np.r_[snakes[0],snakes[0,0]],'g-')

#%%
nr_iter = 25
nr_iter2 = 3

#%% First iter
for j in range(nr_iter):
    snakes = sis.evolve_snake(snakes, im_full[0,:,:], B, step_size) 

snakes3D.append(snakes)  

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][1],snakes3D[0][1,0]],np.r_[snakes3D[0][0],snakes3D[0][0,0]],'r-')

#%% After iters
for i in range(2,25):
    for j in range(nr_iter2):
        snakes = sis.evolve_snake(snakes, im_full[i,:,:], B, step_size)      
    snakes3D.append(snakes)

#%%
imgs = np.arange(1,25, 1) 
for j in imgs:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    ax.plot(np.r_[snakes3D[j][1],snakes3D[j][1,0]],np.r_[snakes3D[j][0],snakes3D[j][0,0]],'r-')

    
#%%
ig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(snakes3D)):
    ax.plot3D(np.r_[snakes3D[i][0],snakes3D[i][0][0]], np.r_[snakes3D[i][1],snakes3D[i][1][0]], len(snakes3D)-i,'red');
ax.set_xlim(0, 349); ax.set_ylim(0, 349);


#%% Tracking multiple
centers = np.array([[215, 170], [225, 90], [260, 130], [260, 285]])
radius = 0.04*np.mean(I.shape)


#%%  First slice
nr_points = 100
nr_iter = 25
step_size = 3
alpha = 0.6
beta = 0.5

B = sis.regularization_matrix(nr_points, alpha, beta)

snakes = []
snakes3D = []
snakes_array = np.zeros(10)

for i in centers:
    snake = sis.make_circular_snake(nr_points, i, radius)
    snakes.append(snake)
    
    
fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.plot(np.r_[snakes[3][1],snakes[3][1,0]],np.r_[snakes[3][0],snakes[3][0,0]],'y-')
ax.set_title('Initialization')


#%% First iters
for j in range(nr_iter):
    for k in range(4):
        snakes[k] = sis.evolve_snake(snakes[k], im_full[0,:,:], B, step_size)
    
snakes3D.append(snakes)  
snakes_array[0] = snakes
#%%
fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][0][1],snakes3D[0][0][1,0]],np.r_[snakes3D[0][0][0],snakes3D[0][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][1][1],snakes3D[0][1][1,0]],np.r_[snakes3D[0][1][0],snakes3D[0][1][0,0]],'g-')
ax.plot(np.r_[snakes3D[0][2][1],snakes3D[0][2][1,0]],np.r_[snakes3D[0][2][0],snakes3D[0][2][0,0]],'b-')         
ax.plot(np.r_[snakes3D[0][3][1],snakes3D[0][3][1,0]],np.r_[snakes3D[0][3][0],snakes3D[0][3][0,0]],'y-')


#%% After iters

for i in range(2,10):
    for j in range(nr_iter2):
        for k in range(4):
            snakes[k] = sis.evolve_snake(snakes[k], im_full[i,:,:], B, step_size) 
    snakes3D.append(snakes)
    snakes_array[i] = snakes      


#%% 2D plot of result
imgs = np.arange(1,10, 1) 
for j in imgs:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    for k in range(4):
        ax.plot(np.r_[snakes3D[j][k][1],snakes3D[j][k][1,0]],np.r_[snakes3D[j][k][0],snakes3D[j][k][0,0]],'r-')
    
    
    
ax.plot(np.r_[snakes3D[j][1][1],snakes3D[j][1][1,0]],np.r_[snakes3D[j][1][0],snakes3D[j][1][0,0]],'g-')
ax.plot(np.r_[snakes3D[j][2][1],snakes3D[j][2][1,0]],np.r_[snakes3D[j][2][0],snakes3D[j][2][0,0]],'b-')         
ax.plot(np.r_[snakes3D[j][3][1],snakes3D[j][3][1,0]],np.r_[snakes3D[j][3][0],snakes3D[j][3][0,0]],'y-')


#%%
fig, ax = plt.subplots()
ax.imshow(im_full[0,:,:], cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][0][1],snakes3D[0][0][1,0]],np.r_[snakes3D[0][0][0],snakes3D[0][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][1][1],snakes3D[0][1][1,0]],np.r_[snakes3D[0][1][0],snakes3D[0][1][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][2][1],snakes3D[0][2][1,0]],np.r_[snakes3D[0][2][0],snakes3D[0][2][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][3][1],snakes3D[0][3][1,0]],np.r_[snakes3D[0][3][0],snakes3D[0][3][0,0]],'r-')

#%%
x = 5
fig, ax = plt.subplots()
ax.imshow(im_full[x,:,:], cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[x][0][1],snakes3D[x][0][1,0]],np.r_[snakes3D[x][0][0],snakes3D[x][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][1][1],snakes3D[x][1][1,0]],np.r_[snakes3D[x][1][0],snakes3D[x][1][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][2][1],snakes3D[x][2][1,0]],np.r_[snakes3D[x][2][0],snakes3D[x][2][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][3][1],snakes3D[x][3][1,0]],np.r_[snakes3D[x][3][0],snakes3D[x][3][0,0]],'r-')



#%% Try with different for each snake
nr_points = 100

#Project
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import maxflow
import simple_snake as sis
#import simple_snake2 as sis
import scipy
import scipy.ndimage

path = 'Data/'

#%% 
I = skimage.io.imread(path + 'nerves_part.tiff')
fig, ax = plt.subplots(1,5, figsize=(12,8))

for i in range(5):
    ax[i].imshow(I[i*100,:,:])


#%% Multiple
im_full = skimage.io.imread(path + 'nerves_part.tiff').astype(np.float)/255
im = im_full[0,:,:]
# sigma = 3
# im_g = scipy.ndimage.gaussian_filter(im, sigma, mode='nearest')
#%%
centers = np.array([[215, 170], [225, 90], [260, 130]])
## Circle outside 
#radius = [0.05*np.mean(I.shape), 0.04*np.mean(I.shape), 0.04*np.mean(I.shape)]
radius = 0.04*np.mean(I.shape)
fig, ax = plt.subplots(1,2)
ax[0].imshow(im)
ax[0].scatter([170, 90, 130], [215, 225, 260], color = 'r')
ax[0].scatter(183, 93, color = 'r')
ax[0].scatter(153, 64, color = 'g')
# ax[1].imshow(im_g)
# ax[1].scatter([170, 90, 130], [215, 225, 260], color = 'r')

#%%  First slice
import simple_snake as sis
nr_points = 100
nr_iter = 25
step_size = 3
alpha = 0.6
beta = 0.5

B = sis.regularization_matrix(nr_points, alpha, beta)

snakes = []
for i in centers:
    snake = sis.make_circular_snake(nr_points, i, radius)
    snakes.append(snake)
    

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.set_title('Initialization')


fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snakes[0,0]],'r-')

#%%
## Plot all
for i in range(nr_iter):
    fig, ax = plt.subplots()
    for j in range(3):
        snakes[j] = sis.evolve_snake(snakes[j], im, B, step_size)

    ax.clear()
    ax.imshow(im, cmap=plt.cm.gray)
    ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
    ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
    ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
    ax.set_title(f'iteration {i}')
    plt.pause(0.001)


#%%
## Plot last
for i in range(nr_iter):
    for j in range(3):
        snakes[j] = sis.evolve_snake(snakes[j], im, B, step_size)

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.set_title(f'iteration {i}')

#%% Tracking through structure
#cent1 = np.array([215, 170])
#cent2 = np.array([225, 90])
#cent3 = np.array([260, 130])
cent4 = np.array([260, 285])



snakes3D = []

## Plot last
snakes = sis.make_circular_snake(nr_points, cent4, radius)

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[1],snakes[1,0]],np.r_[snakes[0],snakes[0,0]],'g-')

#%%
nr_iter = 25
nr_iter2 = 3

#%% First iter
for j in range(nr_iter):
    snakes = sis.evolve_snake(snakes, im_full[0,:,:], B, step_size) 

snakes3D.append(snakes)  

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][1],snakes3D[0][1,0]],np.r_[snakes3D[0][0],snakes3D[0][0,0]],'r-')

#%% After iters
for i in range(2,25):
    for j in range(nr_iter2):
        snakes = sis.evolve_snake(snakes, im_full[i,:,:], B, step_size)      
    snakes3D.append(snakes)

#%%
imgs = np.arange(1,25, 1) 
for j in imgs:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    ax.plot(np.r_[snakes3D[j][1],snakes3D[j][1,0]],np.r_[snakes3D[j][0],snakes3D[j][0,0]],'r-')

    
#%%
ig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(snakes3D)):
    ax.plot3D(np.r_[snakes3D[i][0],snakes3D[i][0][0]], np.r_[snakes3D[i][1],snakes3D[i][1][0]], len(snakes3D)-i,'red');
ax.set_xlim(0, 349); ax.set_ylim(0, 349);


#%% Tracking multiple
centers = np.array([[215, 170], [225, 90], [260, 130], [260, 285]])
radius = 0.04*np.mean(I.shape)


#%%  First slice
nr_points = 100
nr_iter = 25
step_size = 3
alpha = 0.6
beta = 0.5

B = sis.regularization_matrix(nr_points, alpha, beta)

snakes = []
snakes3D = []
snakes_array = np.zeros(10)

for i in centers:
    snake = sis.make_circular_snake(nr_points, i, radius)
    snakes.append(snake)
    
    
fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes[0][1],snakes[0][1,0]],np.r_[snakes[0][0],snakes[0][0,0]],'r-')
ax.plot(np.r_[snakes[1][1],snakes[1][1,0]],np.r_[snakes[1][0],snakes[1][0,0]],'g-')
ax.plot(np.r_[snakes[2][1],snakes[2][1,0]],np.r_[snakes[2][0],snakes[2][0,0]],'b-')
ax.plot(np.r_[snakes[3][1],snakes[3][1,0]],np.r_[snakes[3][0],snakes[3][0,0]],'y-')
ax.set_title('Initialization')


#%% First iters
for j in range(nr_iter):
    for k in range(4):
        snakes[k] = sis.evolve_snake(snakes[k], im_full[0,:,:], B, step_size)
    
snakes3D.append(snakes)  
snakes_array[0] = snakes
#%%
fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][0][1],snakes3D[0][0][1,0]],np.r_[snakes3D[0][0][0],snakes3D[0][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][1][1],snakes3D[0][1][1,0]],np.r_[snakes3D[0][1][0],snakes3D[0][1][0,0]],'g-')
ax.plot(np.r_[snakes3D[0][2][1],snakes3D[0][2][1,0]],np.r_[snakes3D[0][2][0],snakes3D[0][2][0,0]],'b-')         
ax.plot(np.r_[snakes3D[0][3][1],snakes3D[0][3][1,0]],np.r_[snakes3D[0][3][0],snakes3D[0][3][0,0]],'y-')


#%% After iters

for i in range(2,10):
    for j in range(nr_iter2):
        for k in range(4):
            snakes[k] = sis.evolve_snake(snakes[k], im_full[i,:,:], B, step_size) 
    snakes3D.append(snakes)
    snakes_array[i] = snakes      


#%% 2D plot of result
imgs = np.arange(1,10, 1) 
for j in imgs:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    for k in range(4):
        ax.plot(np.r_[snakes3D[j][k][1],snakes3D[j][k][1,0]],np.r_[snakes3D[j][k][0],snakes3D[j][k][0,0]],'r-')
    
    
    
ax.plot(np.r_[snakes3D[j][1][1],snakes3D[j][1][1,0]],np.r_[snakes3D[j][1][0],snakes3D[j][1][0,0]],'g-')
ax.plot(np.r_[snakes3D[j][2][1],snakes3D[j][2][1,0]],np.r_[snakes3D[j][2][0],snakes3D[j][2][0,0]],'b-')         
ax.plot(np.r_[snakes3D[j][3][1],snakes3D[j][3][1,0]],np.r_[snakes3D[j][3][0],snakes3D[j][3][0,0]],'y-')


#%%
fig, ax = plt.subplots()
ax.imshow(im_full[0,:,:], cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[0][0][1],snakes3D[0][0][1,0]],np.r_[snakes3D[0][0][0],snakes3D[0][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][1][1],snakes3D[0][1][1,0]],np.r_[snakes3D[0][1][0],snakes3D[0][1][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][2][1],snakes3D[0][2][1,0]],np.r_[snakes3D[0][2][0],snakes3D[0][2][0,0]],'r-')
ax.plot(np.r_[snakes3D[0][3][1],snakes3D[0][3][1,0]],np.r_[snakes3D[0][3][0],snakes3D[0][3][0,0]],'r-')

#%%
x = 5
fig, ax = plt.subplots()
ax.imshow(im_full[x,:,:], cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D[x][0][1],snakes3D[x][0][1,0]],np.r_[snakes3D[x][0][0],snakes3D[x][0][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][1][1],snakes3D[x][1][1,0]],np.r_[snakes3D[x][1][0],snakes3D[x][1][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][2][1],snakes3D[x][2][1,0]],np.r_[snakes3D[x][2][0],snakes3D[x][2][0,0]],'r-')
ax.plot(np.r_[snakes3D[x][3][1],snakes3D[x][3][1,0]],np.r_[snakes3D[x][3][0],snakes3D[x][3][0,0]],'r-')



#%% Try with different for each snake
import simple_snake as sis
nr_points = 100
step_size = 5
alpha = 4
beta = 3.8
# alpha = 3
# beta = 2.5
nr_iter = 80
nr_iter2 = 2
#small = [70, 180]
#long = [50, 240]
#top = [65, 180]
centers = np.array([[215, 170], [225, 90], [260, 130], [265, 285]])
radius1 = 0.07*np.mean(im.shape)
B = sis.regularization_matrix(nr_points, alpha, beta)

snakes3D_1 = []
snakes3D_2 = []
snakes3D_3 = []
snakes3D_4 = []

snakes_1 = sis.make_circular_snake(nr_points, centers[0], radius)
snakes_2 = sis.make_circular_snake(nr_points, centers[1], radius)
snakes_3 = sis.make_circular_snake(nr_points, centers[2], radius)
snakes_4 = sis.make_circular_snake(nr_points, centers[3], radius1)
    
fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes_1[1],snakes_1[1,0]],np.r_[snakes_1[0],snakes_1[0,0]],'r-')
ax.plot(np.r_[snakes_2[1],snakes_2[1,0]],np.r_[snakes_2[0],snakes_2[0,0]],'g-')
ax.plot(np.r_[snakes_3[1],snakes_3[1,0]],np.r_[snakes_3[0],snakes_3[0,0]],'b-')
ax.plot(np.r_[snakes_4[1],snakes_4[1,0]],np.r_[snakes_4[0],snakes_4[0,0]],'y-')
ax.set_title('Initialization')


#%% First iteration (slice 0)
for j in range(nr_iter):
    snakes_1 = sis.evolve_snake(snakes_1, im_full[0,:,:], B, step_size)
    snakes_2 = sis.evolve_snake(snakes_2, im_full[0,:,:], B, step_size) 
    snakes_3 = sis.evolve_snake(snakes_3, im_full[0,:,:], B, step_size) 
    snakes_4 = sis.evolve_snake(snakes_4, im_full[0,:,:], B, step_size) 

snakes3D_1.append(snakes_1)
snakes3D_2.append(snakes_2)  
snakes3D_3.append(snakes_3)  
snakes3D_4.append(snakes_4)  

fig, ax = plt.subplots()
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(np.r_[snakes3D_1[0][1],snakes3D_1[0][1,0]],np.r_[snakes3D_1[0][0],snakes3D_1[0][0,0]],'r-')
ax.plot(np.r_[snakes3D_2[0][1],snakes3D_2[0][1,0]],np.r_[snakes3D_2[0][0],snakes3D_2[0][0,0]],'g-')
ax.plot(np.r_[snakes3D_3[0][1],snakes3D_3[0][1,0]],np.r_[snakes3D_3[0][0],snakes3D_3[0][0,0]],'b-')
ax.plot(np.r_[snakes3D_4[0][1],snakes3D_4[0][1,0]],np.r_[snakes3D_4[0][0],snakes3D_4[0][0,0]],'y-')

#%% Following iterations (slice 0 +  for n in 1,2,...,1024)
for i in range(2,900):
    for j in range(nr_iter2):
        snakes_1 = sis.evolve_snake(snakes_1, im_full[i,:,:], B, step_size)
        snakes_2 = sis.evolve_snake(snakes_2, im_full[i,:,:], B, step_size)
        snakes_3 = sis.evolve_snake(snakes_3, im_full[i,:,:], B, step_size)
        snakes_4 = sis.evolve_snake(snakes_4, im_full[i,:,:], B, step_size)
        
            
    snakes3D_1.append(snakes_1)
    snakes3D_2.append(snakes_2)
    snakes3D_3.append(snakes_3)
    snakes3D_4.append(snakes_4)

#%% Visualizing all the way trhoug 
imgs = np.arange(1,900,30) 
for j in imgs:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    ax.plot(np.r_[snakes3D_1[j][1],snakes3D_1[j][1,0]],np.r_[snakes3D_1[j][0],snakes3D_1[j][0,0]],'r-')
    ax.plot(np.r_[snakes3D_2[j][1],snakes3D_2[j][1,0]],np.r_[snakes3D_2[j][0],snakes3D_2[j][0,0]],'g-')
    ax.plot(np.r_[snakes3D_3[j][1],snakes3D_3[j][1,0]],np.r_[snakes3D_3[j][0],snakes3D_3[j][0,0]],'b-')
    ax.plot(np.r_[snakes3D_4[j][1],snakes3D_4[j][1,0]],np.r_[snakes3D_4[j][0],snakes3D_4[j][0,0]],'y-')

#%% Saving image slices
for j in [0, 250, 500, 750, 898]:
    fig, ax = plt.subplots()
    ax.imshow(im_full[j,:,:], cmap=plt.cm.gray)
    ax.plot(np.r_[snakes3D_1[j][1],snakes3D_1[j][1,0]],np.r_[snakes3D_1[j][0],snakes3D_1[j][0,0]],'r-')
    ax.plot(np.r_[snakes3D_2[j][1],snakes3D_2[j][1,0]],np.r_[snakes3D_2[j][0],snakes3D_2[j][0,0]],'g-')
    ax.plot(np.r_[snakes3D_3[j][1],snakes3D_3[j][1,0]],np.r_[snakes3D_3[j][0],snakes3D_3[j][0,0]],'b-')
    ax.plot(np.r_[snakes3D_4[j][1],snakes3D_4[j][1,0]],np.r_[snakes3D_4[j][0],snakes3D_4[j][0,0]],'y-')
    
