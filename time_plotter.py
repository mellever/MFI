import numpy as np
import matplotlib.pyplot as plt

times_ss = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_ss.txt')
times_hs = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_hs.txt')
initials = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/initials.txt')

print("Average runtime smartseed = ",np.round(np.mean(times_ss[~np.isnan(times_ss)]),3),"s")
print("Number of nans smartseed = ", len(times_ss[np.isnan(times_ss)]))
print("Number of not nans smartseed = ", len(times_ss[~np.isnan(times_ss)]))
print("Percentage of nans smartseed = ", np.round(len(times_ss[np.isnan(times_ss)])/(len(times_ss[~np.isnan(times_ss)])+len(times_ss[np.isnan(times_ss)]))*100,3),"%")

print("Average runtime homotopy = ", np.round(np.mean(times_hs[~np.isnan(times_hs)]),3),"s")
print("Number of nans homotopy = ", len(times_hs[np.isnan(times_hs)]))
print("Number of not nans homotopy = ", len(times_hs[~np.isnan(times_hs)]))
print("Percentage of nans homotopy = ", np.round(len(times_hs[np.isnan(times_hs)])/(len(times_hs[~np.isnan(times_hs)])+len(times_hs[np.isnan(times_hs)]))*100,3),"%")

print("Speed improvement", np.round(100*(np.mean((times_hs[~np.isnan(times_hs)])-np.mean(times_ss[~np.isnan(times_ss)]))/(np.mean(times_hs[~np.isnan(times_hs)]))), 3), "%")

smart_seed = np.array([1.095857, 1.04805, 0.593602, 0.551997, 0.087886, 0.078495])

diff = np.sum(np.abs(initials - smart_seed),axis=1)
sorted_indices = np.argsort(diff)
times_ss_sorted = times_ss[sorted_indices]
times_hs_sorted = times_hs[sorted_indices]
indices = np.arange(0, len(times_ss))

plt.scatter(indices, times_ss, marker='.', label="Smart seed")
plt.scatter(indices, times_hs, marker='.', label="Homotopy")
plt.ylabel("Runtime [s]")
plt.xlabel("Initial condition number")
plt.legend()
plt.grid()
plt.show()



