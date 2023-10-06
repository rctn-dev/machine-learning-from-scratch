import matplotlib.pyplot as plt
import numpy as np

_range=2.5
_x=np.linspace(-_range,_range,100)
_y=np.linspace(-_range,_range,100)
X,Y = np.meshgrid(_x,_y)

# matlab peaks function
_Z=3*(1-X)**2*np.exp(-(X**2) - (Y+1)**2)- 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2) - 1/3*np.exp(-(X+1)**2 - Y**2) 
Z=1.5*_Z/np.max(_Z)
# Plot a 3D surface and contour
fig1=plt.figure(figsize=(6,4.5), alpha=0.5)
ax=fig1.add_subplot(111,projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none',linewidth=1, antialiased=True,rstride=1, cstride=1)
ax.contour(X, Y, Z, 30, zdir='z', offset=-1.5, cmap='jet',linewidths=1.0)
ax.set(zlim=(-1.5, 1))
# removed grid lines on the panes.
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# brought the labels closer to the axis line with the help of labelpad
ax.set_xlabel("b",labelpad=-10)
ax.set_ylabel("w",labelpad=-10)
ax.set_zlabel("J (w,b)",labelpad=-10)
ax.set_title("3D and Contour of Cost Function")
plt.show()
fig1.savefig('assets/images/cost_3d.svg', format='svg', dpi=1200)

# plot only contour
fig2=plt.figure(figsize=(4,3), alpha=0.5)
ax=fig2.add_subplot(111) # there will be only one plot, row=1,col=1, and 1st figure.
ax.contour(X, Y, Z, 30, zdir='z', offset=-1.5, cmap='jet',linewidths=1.0)
# removed tick labels with the help of empty list.
ax.set_xticks([])
ax.set_yticks([])
# set the title and labels
ax.set_xlabel("b")
ax.set_ylabel("w")
ax.set_title("Contour of Cost Function")
plt.show()
# saved the figure.
fig2.savefig('assets/images/cost_contour.svg', format='svg', dpi=1200)
