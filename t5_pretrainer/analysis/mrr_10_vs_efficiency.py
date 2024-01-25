import matplotlib.pyplot as plt 
import numpy as np 
import ujson 
import os 

fig, ax = plt.subplots(figsize=(6,3.5))

a =  (90 * 60) / 6980 * 8 # (sec/query)
#xs = [10, 50, 100, 200, 500, 1000]
ripor_xs = np.array([a/100, a/20, a/10, a/5, a/2, a]) * 1000 # (ms/query)
ripor_perfs = [0.17036129303679468, 0.24915768408605, 0.2772657706826748, 0.3007011529540169, 0.32091946377404734, 0.332967] 


#gd_ripor_xs = [10, 50, 100, 200, 500, 1000]
gd_ripor_xs = (np.array([308, 967, 1751, 3311, 3311*1.94, 3311*1.94*1.98]) + 177) / 6980 * 1000
gd_ripor_perfs = [.380, .385, .386, .386, .386, .386]

ltrgr_xs = np.array([11384, 18407, 25336, 40890, 85869, 154564]) * 1000 / 6980
ltrgr_perfs = [.253, .219, .212, 0.1939, 0.187, 0.1801]

dsi_qg_xs = np.array([149, 426, 868, 1728, 4451, 9081]) * 1000 / 6980
dsi_qg_perfs = np.array([0.092, .0915, .0908, .0899, .08928, .08877]) * 1.141304347826087


print("ripor: ", ripor_xs)
print("gd_ripor: ", gd_ripor_xs)
ax.plot(dsi_qg_xs, dsi_qg_perfs, label="DSI-QG", marker="s")
ax.plot(ltrgr_xs, ltrgr_perfs, label="LTRGR", marker="o")
ax.plot(ripor_xs, ripor_perfs, marker="^", label="RIPOR")
ax.plot(gd_ripor_xs, gd_ripor_perfs, marker=".", label="PAG")

#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.title("The Performance and Lantency Trade-off on MSMARCO Dev", fontsize=14)

ax.set_xlabel("Query lantecy (ms/query)", fontsize=12)
ax.set_ylabel("MRR@10", fontsize=12)
ax.set_xscale("log")
plt.legend()
plt.tight_layout()

plt.savefig("figures/diff_gr_comp_perf_vs_lantency.jpg")
