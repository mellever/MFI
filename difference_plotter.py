import numpy as np 
import matplotlib.pyplot as plt


diff_arr = np.loadtxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/difference.txt")
key_list = np.loadtxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/keylist.txt", dtype='str')
t = np.arange(1,len(diff_arr[0])+1)


plt.figure()
for i in range(len(diff_arr)): plt.plot(t, diff_arr[i], label=str(key_list[i]))
plt.xlabel("t")
plt.ylabel("difference")
plt.title("Difference plot")
plt.grid()
plt.legend()
plt.show()
