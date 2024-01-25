import ujson 
import os 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ripor_runs_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/ripor_direct_lng_knp_seq2seq_1/sub_tokens/"
    lex_ripor_runs_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-full-lexical-ripor/lexical_ripor_direct_lng_knp_seq2seq_1/lex_ret_1000/"
    out_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/t5_pretrainer/analysis/prefix_perf_comp/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    beam_sizes = [10, 50, 100, 200]
    pref_lengths = [2, 4, 6, 8]

    metric_key = "mrr_10"
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

    plt.figure(figsize=(10,6))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'P', 'H']
    linestyles = ['-', '--', '-.', ':']

    for i, perfs in enumerate(list_perfs):
        plt.plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="ripor with {} beam size".format(beam_sizes[i]))
    
    for i, perfs in enumerate(other_list_perfs):
        plt.plot(pref_lengths, perfs, marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label="bow_ripor with {} beam size".format(beam_sizes[i]))


    plt.title(f"MSMARCO Dev set with prefix-level relevant labels")
    plt.xlabel("prefix length")
    plt.ylabel("MRR@10" if metric_key=="mrr_10" else "Recall@10")
    plt.xticks(pref_lengths)  # Using ks_1 to set xticks, as ks_1 and ks_2 are same
    plt.legend(loc="upper right")
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric_key}_curve_diff_beam_size_prefix_level.jpg"), format='jpg', dpi=300)