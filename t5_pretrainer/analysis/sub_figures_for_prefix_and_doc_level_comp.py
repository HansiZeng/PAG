import ujson 
import os 
import matplotlib.pyplot as plt

def get_doc_level_perfs(ripor_runs_dir, lex_ripor_runs_dir, metric_key):
    beam_sizes = [10, 50, 100, 200]
    pref_lengths = [2, 4, 6, 8]

    list_perfs = []
    for beam_size in beam_sizes:
        perfs = []
        for length in pref_lengths:
            perf_path = os.path.join(ripor_runs_dir, f"ret_{beam_size}_sub_{length}/MSMARCO/perf.json")
            with open(perf_path) as fin:
                perf = ujson.load(fin)
            
            perfs.append(perf[metric_key])
        
        list_perfs.append(perfs)

    other_list_perfs = []
    for beam_size in beam_sizes:
        perfs = []
        for length in pref_lengths:
            perf_path = os.path.join(lex_ripor_runs_dir, f"ltmp_smt_ret_{beam_size}_sub_{length}/MSMARCO/perf.json")
            
            if beam_size == 100 and length == 8:
                if metric_key == "mrr_10":
                    perf = {"mrr_10": 0.3847}
                else:
                    perf = {"recall_10": 0.6704}
                pass
            else:
                #print(perf_path)
                with open(perf_path) as fin:
                    perf = ujson.load(fin)
            
            perfs.append(perf[metric_key])
        
        other_list_perfs.append(perfs)

    return list_perfs, other_list_perfs

def get_prefix_level_perfs(ripor_runs_dir, lex_ripor_runs_dir, metric_key):
    beam_sizes = [10, 50, 100, 200]
    pref_lengths = [2, 4, 6, 8]

    list_perfs = []
    for beam_size in beam_sizes:
        perfs = []
        for length in pref_lengths:
            perf_path = os.path.join(ripor_runs_dir, f"ret_{beam_size}_sub_{length}/MSMARCO/perf.json")
            with open(perf_path) as fin:
                perf = ujson.load(fin)
            
            perfs.append(perf[metric_key])
        
        list_perfs.append(perfs)

    other_list_perfs = []
    for beam_size in beam_sizes:
        perfs = []
        for length in pref_lengths:
            perf_path = os.path.join(lex_ripor_runs_dir, f"ltmp_smt_ret_{beam_size}_sub_{length}/MSMARCO/perf.json")
            
            if beam_size == 10 and length == 2:
                if metric_key == "mrr_10":
                    perf = {"mrr_10": 0.5517}
                else:
                    perf = {"recall_10": 0.8295}
            else:
                with open(perf_path) as fin:
                    perf = ujson.load(fin)
            
            perfs.append(perf[metric_key])
        
        other_list_perfs.append(perfs)

    return list_perfs, other_list_perfs

if __name__ == "__main__":
    out_dir = "./t5_pretrainer/analysis/prefix_perf_comp/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    metric_key = "mrr_10"

    ripor_runs_dir = "./data/experiments-full-lexical-ripor/ripor_direct_lng_knp_seq2seq_1/doc_ret_by_sub_tokens/"
    lex_ripor_runs_dir = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/doc_ret_by_sub_tokens/"
    doc_list_perfs, other_doc_list_perfs = get_doc_level_perfs(ripor_runs_dir, lex_ripor_runs_dir, metric_key)

    ripor_runs_dir = "./data/experiments-full-lexical-ripor/ripor_direct_lng_knp_seq2seq_1/sub_tokens/"
    lex_ripor_runs_dir = "./data/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/lex_ret_1000/"
    prefix_list_perfs, other_prefix_list_perfs = get_prefix_level_perfs(ripor_runs_dir, lex_ripor_runs_dir, metric_key)

    fig, axs = plt.subplots(1,2, figsize=(10,5), sharey=True)

    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', 'H']
    linestyles = ['-', '--', '-.', ':']
    pref_lengths = [2, 4, 6, 8]
    beam_sizes = [10, 50, 100, 200]

    for i, perfs in enumerate(prefix_list_perfs):
        axs[0].plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="RIPOR w/ bs={}".format(beam_sizes[i]))
    
    for i, perfs in enumerate(other_prefix_list_perfs):
        axs[0].plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="PAG w/ bs={}".format(beam_sizes[i]))
        axs[0].set_title("Prefix-level Relevant Labels", fontsize=20)
        axs[0].set_xlabel("prefix_length", fontsize=18)
        axs[0].set_ylabel("MRR@10" if metric_key=="mrr_10" else "Recall@10", fontsize=18)
        axs[0].set_xticks(pref_lengths)

    for i, perfs in enumerate(doc_list_perfs):
        axs[1].plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="RIPOR w/ bs={}".format(beam_sizes[i]))

    for i, perfs in enumerate(other_doc_list_perfs):
        axs[1].set_title("Doc-level Relevant Labels", fontsize=20)
        axs[1].plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="PAG w/ bs={}".format(beam_sizes[i]))
        axs[1].set_xlabel("prefix_length", fontsize=18)
        axs[1].set_xticks(pref_lengths)


    handles, labels = axs[1].get_legend_handles_labels()

    #plt.suptitle(f"MSMARCO Dev set with prefix-level relevant labels")
    #plt.ylabel("MRR@10" if metric_key=="mrr_10" else "Recall@10")
    plt.xticks(pref_lengths)  # Using ks_1 to set xticks, as ks_1 and ks_2 are same
    #plt.figlegend(loc = 'lower center', ncol=4, labelspacing=0.)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='center left', ncol=2, bbox_to_anchor=(0.54, 0.81))
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric_key}_doc_prefix_level_curve_diff_beam_size.jpg"), format='jpg', dpi=300)

