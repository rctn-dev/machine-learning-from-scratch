import numpy as np
from  matplotlib import pyplot as plt

# dummy training dataset for house price prediction.
prices=np.array([85,120,200,150,250,230,280,260,290,310])
sizes=np.array([480,600,750,900,1000,1200,1500,1600,1800,2200])
sizes_interp=np.arange(np.min(sizes),np.max(sizes),1) # arange(start,stop,step)
predicted_price_linear=0.12*sizes_interp+72
predicted_price_nonlinear=-0.000066*sizes_interp**2 +0.29*(sizes_interp)-17


fig=plt.figure(figsize=(4,3), alpha=0.5)
# ax = plt.axes()
# ax.set_facecolor("whitesmoke")


plt.title('Regression: House Price Prediction')
plt.xlabel('House Sizes (m^2)')
plt.ylabel('House Prices (*1000$)')
plt.scatter(sizes,prices,marker="x", color="blue")
plt.plot(sizes_interp,predicted_price_linear, color="red")
plt.plot(sizes_interp,predicted_price_nonlinear, color="orange")
plt.legend(["training dataset\n (right answers)", "linear fit","nonlinear fit"], loc ="lower right")
plt.xticks([])
plt.yticks([])


plt.show()
fig.savefig('assets/images/house_pricing_prediction.svg', format='svg', dpi=1200)

