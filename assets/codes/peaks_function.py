import matplotlib.pyplot as plt
import numpy as np

############################  -3d plot- ################################
_range=2.5
_x=np.linspace(-_range,_range,100)
_y=np.linspace(-_range,_range,100)
X,Y = np.meshgrid(_x,_y)
# matlab peaks function
Z=3*(1-X)**2*np.exp(-(X**2) - (Y+1)**2)- 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2) - 1/3*np.exp(-(X+1)**2 - Y**2) 
# Plot the 3D surface and contour
fig1=plt.figure(figsize=(6,4.5))
ax1=fig1.add_subplot(111,projection='3d')
ax1.plot_surface(X, Y, Z, cmap='jet', edgecolor='none',linewidth=1,rstride=1, cstride=1,zorder=1)
ax1.contour(X, Y, Z, 30, zdir='z', offset=-9.0, cmap='jet',linewidths=1.0)
ax1.contour(X, Y, Z, 30, cmap='jet',linewidths=1.0)
ax1.set(zlim=(-9, 8))
# removed grid lines on the panes.
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
# brought the labels closer to the axis line with the help of labelpad
ax1.set_xlabel("b",labelpad=-10)
ax1.set_ylabel("w",labelpad=-10)
ax1.set_zlabel("J (w,b)",labelpad=-10)
ax1.set_title("3D and Contour of Cost Function")
# defined roughly the first descent path, p1
path1_b=np.array([1.2, 1.1,1.0,0.9,0.6,0.3,0.25,0.21])
path1_w=np.array([-0.3,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6])
X1,Y1=(path1_b,path1_w)
Z1=3*(1-X1)**2*np.exp(-(X1**2) - (Y1+1)**2)- 10*(X1/5 - X1**3 - Y1**5)*np.exp(-X1**2-Y1**2) - 1/3*np.exp(-(X1+1)**2 - Y1**2) 
ax1.plot(X1, Y1, Z1, marker='*', zorder=5,  color='black')
# defined roughly the second descent path, p2
path2_b=np.array([-0.8, -0.75,-0.8,-0.9,-1,-1.1,-1.2,-1.4])
path2_w=np.array([1.4,1.2,1.1,1.0,0.8,0.7,0.4,0.2])
X2,Y2=(path2_b,path2_w)
Z2=3*(1-X2)**2*np.exp(-(X2**2) - (Y2+1)**2)- 10*(X2/5 - X2**3 - Y2**5)*np.exp(-X2**2-Y2**2) - 1/3*np.exp(-(X2+1)**2 - Y2**2) 
ax1.plot(X2, Y2, Z2, marker='*', color='black',zorder=5)
#annotations
ax1.text(-1, 1, 4, "$p_{1}$", color='black', zorder=5)
ax1.text(1, 0, 4, "$p_{2}$", color='black', zorder=5)
fig1.savefig('assets/images/cost_3d.svg', format='svg', dpi=1200)

plt.show()

############################  -2D CONTOUR PLOT- ################################
# plot only contour
fig2=plt.figure(figsize=(4,3), alpha=0.5)
ax2=fig2.add_subplot(111) # there will be only one plot, row=1,col=1, and 1st figure.
ax2.contour(X, Y, Z, 30, cmap='jet',linewidths=1.0)
# removed tick labels with the help of empty list.
ax2.set_xticks([])
ax2.set_yticks([])
# set the title and labels
ax2.set_xlabel("b")
ax2.set_ylabel("w")
ax2.set_title("Contour of Cost Function")
# defined roughly the first descent path, p1
path1_b=np.array([1.2, 1.1,1.0,0.9,0.6,0.3,0.25,0.21])
path1_w=np.array([-0.3,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6])
plt.plot(path1_b,path1_w, marker='*',markersize=3,color='black', linewidth=0.7)
# defined roughly the second descent path, p2
path2_b=np.array([-0.8, -0.75,-0.8,-0.9,-1,-1.1,-1.2,-1.4])
path2_w=np.array([1.4,1.2,1.1,1.0,0.8,0.7,0.4,0.2])
plt.plot(path2_b,path2_w, marker='*',markersize=3,color='black', linewidth=0.7)
#annotations
ax2.text(-1.2, 1.2, "$p_{1}$", color='black', zorder=5)
ax2.text(0.8, -0.2, "$p_{2}$", color='black', zorder=5)

plt.show()
# saved the figure.
fig2.savefig('assets/images/cost_contour.svg', format='svg', dpi=1200)
