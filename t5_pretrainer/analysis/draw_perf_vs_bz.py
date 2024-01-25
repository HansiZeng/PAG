import ujson 
import os 
import matplotlib.pyplot as plt 

beam_sizes = [10, 50, 100, 200, 500, 1000]
out_dir = "./t5_pretrainer/analysis/figures/"

os.makedirs(out_dir, exist_ok=True)

list_xs = [10, 50, 100, 200, 500, 1000]                         
list_mrr_10 = [0.17036129303679468, 0.24915768408605, 0.2772657706826748, 0.3007011529540169, 0.32091946377404734, 0.332967] 
plt.figure(figsize=(6,3.5))
# RIPOR
plt.plot(list_xs, list_mrr_10, label="RIPOR", color="cornflowerblue", marker="^")

# bm25
plt.axhline(y = .185, label = "BM25", linestyle=":", color="violet")
plt.axhline(y = .35969, label = "RIPOR rank all docs", linestyle="-.", color="slateblue")

plt.xticks(list_xs)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("Beam size", fontsize=12)
plt.ylabel("MRR@10",fontsize=12)
plt.title("RIPOR Perf. on MSMARCO Dev", fontsize=14)
plt.tight_layout()
plt.legend()

plt.savefig(os.path.join(out_dir, "perf_vs_bz.jpg"))


