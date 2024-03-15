import numpy as np 
import matplotlib.pyplot as plt

solution_hom = np.loadtxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/solution.txt")
solution_ss = np.loadtxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/solution_ss.txt")
key_list = np.loadtxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/keylist.txt", dtype='str')

data_path = "rtc-tools-examples/cascading_channels/output/timeseries_export.csv"
record = np.recfromcsv(data_path, encoding=None)

c = "UpperChannel"
solution = record[c.lower() + "h1"]

solution_hom = solution_hom[1:]
solution = solution[1:]
t = np.arange(1,len(solution_hom)+1)

plt.figure()
plt.plot(t,solution_hom, label="Homotopy")
plt.plot(t,solution_ss, label="Smart seed")
plt.scatter(t,solution, label="Actual solution", c='r')
plt.xlabel("t [h]")
plt.ylabel("h [m]")
plt.title(key_list)
plt.grid()
plt.legend()
plt.savefig("/home/melle/Pictures/UpperChannelDifferentInitial.pdf")
plt.show()

