import numpy as np 
import matplotlib.pyplot as plt
import os

a =  (90 * 60) / 6980 * 8 # (sec/query)
a_0 = a / 100
xs = [10, 50, 100, 200, 500, 1000]
#ys = [a/100, 1/20, a/10, a/5, a/2, a]
ys = [a_0, a_0 * 4.2, a_0 * 8.9, a_0 * 8.9 * 2, a_0 * 8.9 * 4.8, a_0 * 8.9*9.8]
ys = np.array(ys) * 1000 # (ms / query)
list_mrr_10 = [0.17036129303679468, 0.24915768408605, 0.2772657706826748, 0.3007011529540169, 0.32091946377404734, 0.332967] 

fig, ax1 = plt.subplots(figsize=(6, 3.5))

color = 'tab:red'
ax1.set_xlabel('beam size', fontsize=12)
ax1.set_ylabel('MRR@10', color=color,  fontsize=12)
ax1.set_title("RIPOR Perf. and Latency on MSMARCO Dev",  fontsize=14)

lns1 = ax1.plot(xs, list_mrr_10, color=color, marker="^", label="RIPOR perf.")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('query latency (ms/query)', color=color,  fontsize=12)  # we already handled the x-label with ax1
lns2 = ax2.plot(xs, ys, color=color, marker=".", label="RIPOR latency")
ax2.tick_params(axis='y', labelcolor=color)



# bm25
lns3 = ax1.axhline(y = .185, label = "BM25 perf.", linestyle=":", color='tab:red')
lns4 = ax1.axhline(y = .35969, label = "brute force RIPOR perf.", linestyle="-.", color='tab:red')

plt.xticks(xs)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
lns = lns1 
lns += [lns3, lns4]
lns += lns2
labels = [l.get_label() for l in lns]

plt.legend(lns, labels, loc='center right', bbox_to_anchor=(1.0,0.32))

out_dir = "./figures/"
plt.savefig(os.path.join(out_dir, "perf_and_latency_vs_bz_1gpu.jpg"))
