import numpy as np
from  matplotlib import pyplot as plt

# dummy training dataset for house price prediction.
tumor_size_benign=np.array([1.2,1.4,0.8,2.1,1.4,2.0,1.3,1.6,1.2,0.9,0.7,1.1])
age_benign=np.array([15,17,20,22,25,28,30,33,35,37,40,42])
tumor_size_malign=np.array([2.0,1.9,1.5,1.2,1.8,2.2,1.6,0.8,1.2,1.0,1.6,1.9,1.7,2.1])
age_malign=np.array([37,40,43,45,48,50,52,55,57,60,63,65,68,70])
boundary_x=np.arange(30,60,1)
boundary_y=0.002*(boundary_x-60)**2+0.6


fig=plt.figure(figsize=(4,3), tight_layout=True, alpha=0.5)
# ax = plt.axes()
# ax.set_facecolor("whitesmoke")


plt.title('Regression: Tumor Prediction')
plt.xlabel('Patient Age')
plt.ylabel('Tumor Size (mm^2)')
plt.scatter(age_benign,tumor_size_benign,marker="o", color="blue")
plt.scatter(age_malign,tumor_size_malign,marker="x",color="red")
plt.plot(boundary_x,boundary_y,color="black")
plt.legend(["benign", "malignent","boundary-curve"], loc ="lower right")

plt.xticks([])
plt.yticks([])

plt.show()
fig.savefig('assets/images/tumor_prediction.svg', format='svg', dpi=1200)