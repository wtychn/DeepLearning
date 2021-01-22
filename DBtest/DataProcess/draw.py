import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_x = np.load("data_x.npy")
data_y = np.load("data_c.npy")

# temp = np.array([1])
# vel = np.array([1])
# num = np.array([1])
#
# for i in range(len(data_x)):
#     for j in range(len(data_x[0])):
#         temp = np.append(temp, data_x[i, j, 0])
#         vel = np.append(vel, data_x[i, j, 1])
#         num = np.append(num, j)
#
# temp = np.delete(temp, 0)
# vel = np.delete(vel, 0)
# num = np.delete(num, 0)
#
# df = pd.DataFrame({"temp": temp, "vel": vel, "num": num})
#
# sns.jointplot("temp", "num", data=df, kind="hex")
sns.displot(data_y)
plt.show()
