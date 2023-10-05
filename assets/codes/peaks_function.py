from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#if using a Jupyter notebook, include:
import matplotlib as mpl
cmap = mpl.colormaps['viridis']
rng=2.5
x = np.linspace(-rng,rng,100)
y = np.linspace(-rng,rng,100)
X,Y = np.meshgrid(x,y)
Z=3*(1-X)**2*np.exp(-(X**2) - (Y+1)**2)- 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2) - 1/3*np.exp(-(X+1)**2 - Y**2) 
#Z=np.sin(X)*np.cos(Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#fig, ax = plt.subplots()
# Plot a 3D surface
ax.plot_surface(X, Y, Z/np.max(Z), cmap='jet', edgecolor='none',linewidth=1, antialiased=True,rstride=1, cstride=1)


# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False

# # Now set color to white (or whatever is "invisible")
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
#ax.grid(False)
#ax.axis(False)
ax.contour(X, Y, Z, 20, zdir='z', offset=-1.5, cmap='jet',linewidths=1.0)

# match the lower bound of zlim to the offset
ax.set(zlim=(-1.5, 1))
ax.view_init(elev=20, azim=-120, roll=0)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.title('3D and Contour Image of Peaks Function')

plt.show()
fig.savefig('assets/images/peaks_3d.svg', format='svg', dpi=1200)


fig2, ax = plt.subplots()

CS=ax.contour(X,Y,Z,30,cmap='jet',linewidths=0.75)
#ax.axis(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('Contour Image of Peaks Function')
plt.show()
fig2.savefig('assets/images/peaks_contour.svg', format='svg', dpi=1200)

