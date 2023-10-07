import numpy as np
from  matplotlib import pyplot as plt

# dummy training dataset for tumor size and patient age.
tumor_size_benign=np.array([1.2,1.4,0.8,1.5,1.1,0.9,1.3,1.6,1.2])
age_benign=np.array([15,17,20,22,25,28,30,33,35])
tumor_size_malign=np.array([1.8,2.2,1.6,1.7,2.1,1.4,1.6,1.9,1.7,2.1])
age_malign=np.array([48,50,52,55,57,60,63,65,68,70])
boundary_x=np.arange(35,60,1)
boundary_y=0.0015*(boundary_x-70)**2+0.6


fig1=plt.figure(figsize=(4,3), alpha=0.5)
plt.scatter(age_benign,tumor_size_benign,marker="o",color="blue")
plt.plot(boundary_x,boundary_y,color="black")
plt.scatter(age_malign,tumor_size_malign,marker="x",color="red")
plt.title("Supervised Learning: Regression")
plt.xlabel('Patient Age')
plt.ylabel('Tumor Size')
plt.xticks([])
plt.yticks([])
plt.annotate("boundary-curve",xy=(38,2.1), xytext=(15, 1.8), arrowprops=dict(facecolor='black', width=1, headwidth=5))
plt.show()
fig1.savefig('assets/images/unsup_vs_sup_learning1.svg', format='svg', dpi=1200)

# creating a new figure for unsupervised learning.
fig2=plt.figure(figsize=(4,3), alpha=0.5)
plt.scatter(age_benign,tumor_size_benign,marker="v",color="orange")
plt.scatter(age_malign,tumor_size_malign,marker="v",color="orange")
plt.title("Unsupervised Learning: Clustering")
plt.xlabel('Patient Age')
plt.ylabel('Tumor Size')
plt.xticks([])
plt.yticks([])
plt.show()
fig2.savefig('assets/images/unsup_vs_sup_learning2.svg', format='svg', dpi=1200)
