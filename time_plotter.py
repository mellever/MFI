import numpy as np


times_ss = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_ss.txt')
times_hs = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_hs.txt')
times_ss_bounded = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_ss_bounded.txt')
times_ss_bounded2 = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_ss_bounded2.txt')
times_ss_bounded3 = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/times_ss_bounded3.txt')
initial = np.loadtxt('/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/initial.csv')


runtime_hs = np.mean(times_hs[~np.isnan(times_hs)])
runtime_ss = np.mean(times_ss[~np.isnan(times_ss)])
runtime_ss_bounded = len(times_ss_bounded3[~np.isnan(times_ss_bounded3)])
print("Average runtime homotopy", runtime_hs)
print("Average runtime smartseed", runtime_ss)
print("Speed improvement", round(100*(runtime_hs-runtime_ss)/runtime_hs, 3), "%")
print("Average runtime smartseed bounded", runtime_ss_bounded)

smart_seed = [1.095857, 1.04805, 0.593602, 0.551997, 0.087886, 0.078495]

#initial conditions that give a nan, i.e. do not converge
nan_initials = initial[np.isnan(times_ss)]
diff_nan_initials = np.abs(nan_initials - smart_seed) #This needs to be the seed for the second hour
#max_diff_nan = np.max(diff_nan_initials, axis=0)
diff_nan_av = np.mean(diff_nan_initials, axis=0)

notnan_initials = initial[~np.isnan(times_ss)]
diff_notnan_initials = np.abs(notnan_initials - smart_seed)
#max_diff_notnan = np.max(diff_notnan_initials, axis=0)
diff_notnan_av = np.mean(diff_notnan_initials, axis=0)

print("Average difference not nan = ", diff_notnan_av)
print("Average difference nan = ", diff_nan_av)

sum_diff = np.sum(np.abs(diff_notnan_av - diff_nan_av))

print("Sum of differences = ", sum_diff)


