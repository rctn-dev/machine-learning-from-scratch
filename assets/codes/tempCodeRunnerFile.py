plt.scatter(x, y, marker='x', c='r', label="target"); 
plt.title("multi-features: x x**2,..., x**13, normalized")
plt.plot(x,x_norm@model_w + model_b, label="predicted");
plt.xlabel("x");
plt.ylabel("y=cos(x/3)"); 

plt.annotate(r"$\alpha={alpha}$".format(alpha=alpha),xy=(0,1), xytext=(0, 1))
plt.legend(loc='lower right'); 
plt.show()
fig.savefig('assets/images/sinx_regression.svg', format='svg', dpi=1200)