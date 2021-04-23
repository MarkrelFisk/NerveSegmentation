path = 'C:/Users/mark_/Documents/DTU/Kandidat/1.semester/Advanced Image Analysis/Project/'
slices = skimage.io.imread(path + 'nerves_part.tiff').astype(float)/(2**8-1)

#%% Define initial snakes

img = slices[0,:,:]
center = np.array([[120,170],[150,100],[225,90],[260,135],[220,160],[240,130],[290,225]])
#center = np.array([[225,90],[260,135]])
radius = 0.05*np.mean(img.shape)
num_points = 100
iterations = 10
step_size = 100
alpha = 0.001
beta = 0.1

init_S = []

def initialize_snake(img,center,radius,num_points,iterations,step_size,alpha,beta):
    snake_init = sis.make_circular_snake(num_points, center, radius)
    #snakeN = sis.snake_normals(snake_init)
    B = sis.regularization_matrix(num_points, alpha=alpha, beta=beta)
    snake = snake_init.copy()
    for i in range(iterations):
        snake = sis.displace_snake_II(snake,img,step_size,B)
    return snake

for i in range(len(center)):
    
    init_S.append(initialize_snake(img,center[i],radius,num_points,iterations,step_size,alpha,beta))

snake_init = sis.make_circular_snake(num_points, center[0], radius)
fig,ax = plt.subplots()
ax.imshow(img)
ax.plot(np.r_[snake_init[1],snake_init[1,0]],np.r_[snake_init[0],snake_init[0,0]],'r-') #np.r_ to append missing point
#ax.plot(np.r_[snake_out[1],snake_out[1,0]],np.r_[snake_out[0],snake_out[0,0]],'b-')
#ax.plot(np.r_[snake_in[1],snake_in[1,0]],np.r_[snake_in[0],snake_in[0,0]],'b-')
#ax.quiver(snake_init[0,:],snake_init[1,:],snakeN[0,:]+snake_init[0,:],snakeN[1,:]+snake_init[1,:])
#ax.plot(snakeN[0,:],snakeN[1,:]+snake_init[1,:])
ax.set_title('Initialization')

fig,ax = plt.subplots()
ax.imshow(img)
#ax.plot(np.r_[snake_init[1],snake_init[1,0]],np.r_[snake_init[0],snake_init[0,0]],'r-') #np.r_ to append missing point
for i in range(len(init_S)):
    snake = init_S[i]
    ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')

#%% Find snakes through slices
step_size = 50
snake_1 = []
snake_1.append(init_S[0])
B = sis.regularization_matrix(num_points, alpha=0.05, beta=beta)
snake_1.append(sis.evolve_snake_II(snake_1[0], slices[1,:,:], B, step_size))
for i in range(2,500):
    snake_1.append(sis.evolve_snake_II(snake_1[i-1], slices[i,:,:], B, step_size))

snake_2 = []
snake_2.append(init_S[1])
B = sis.regularization_matrix(num_points, alpha=0.05, beta=beta)
snake_2.append(sis.evolve_snake_II(snake_2[0], slices[1,:,:], B, step_size))
for i in range(2,500):
    snake_2.append(sis.evolve_snake_II(snake_2[i-1], slices[i,:,:], B, step_size))

#%%
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(snake_1)):
    ax.plot3D(np.r_[snake_1[i][0],snake_1[i][0][0]], np.r_[snake_1[i][1],snake_1[i][1][0]], len(snake_1)-i,'red');
    ax.plot3D(np.r_[snake_2[i][0],snake_2[i][0][0]], np.r_[snake_2[i][1],snake_2[i][1][0]], len(snake_1)-i,'blue');
ax.set_xlim(0, 349); ax.set_ylim(0, 349);

#%%
idx = 400
s = snake_1[idx]
plt.imshow(slices[idx])
plt.plot(np.r_[s[1],s[1,0]],np.r_[s[0],s[0,0]],'r-')
