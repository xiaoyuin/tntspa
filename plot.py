import pandas as pd
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import plot_ppls, read_fairseq_history

# matplotlib.style.use('seaborn')
# matplotlib.style.available

def find_ppl_csvs(files):
    d = {}
    for f in files:
        if "train_ppl.csv" in f:
            d['train_ppl'] = f
        if "dev_ppl.csv" in f:
            d['dev_ppl'] = f
        if "test_ppl.csv" in f:
            d['test_ppl'] = f
    return d

def find_fairseq_output(files):
    filtered = [f for f in files if (f.startswith('train_fairseq_') or f.startswith('train_ml')) and f.endswith('.out')]
    if len(filtered) == 0:
        return None
    else:
        return filtered

def plot_fairseq_output(event_file_paths, tags, tags_legends_mapping, y_label, csvs_output_directory, plot_output_directory, set_ylim=True, output_name=None):
    train_tags = tags["train"]
    valid_tags = tags["valid"]
    test_tags = tags["test"]

    def read_line_into_dict(line):
        d = {}
        cols = [ c.strip() for c in line.split('|') ]
        for col in cols:
           parts = col.split()
           if len(parts) == 2:
                try:
                   d[parts[0]] = float(parts[1])
                except ValueError:
                   d[parts[0]] = parts[1]
        return d

    train_dicts = []
    valid_dicts = []
    test_dicts = []

    for event_file_path in event_file_paths:
        with open(event_file_path) as f:
            for line in f.readlines():
                if line.startswith('| epoch'):
                    line = line[1:-1]
                    if "|" in line:
                        d = read_line_into_dict(line)
                        if "valid on 'valid' subset" in line:
                            valid_dicts.append(d)
                        elif "valid on 'test' subset" in line:
                            test_dicts.append(d)
                        else:
                            train_dicts.append(d)
        # try:
            
        # except:
        #     print(event_file_path, "Reading Error")
    
    fig = plt.figure()

    save_paths = [ os.path.join(plot_output_directory, p_name) for p_name in [("&".join(train_tags+valid_tags+test_tags) if output_name is None else output_name)+".png", ("&".join(train_tags+valid_tags+test_tags) if output_name is None else output_name)+".svg"] ]

    ax = fig.add_subplot(111)

    ylim = 0

    if len(train_dicts) != 0 and len(train_tags) != 0:
        train_df = pd.DataFrame(train_dicts).sort_values('num_updates')
        train_df.to_csv(os.path.join(csvs_output_directory, 'train.csv'))
        if set_ylim:
            ylim = max(ylim, *[ train_df.iloc[-1][tag] for tag in train_tags])
        for t in train_tags:
            train_df.plot(x="epoch", y=t, ax=ax, label=tags_legends_mapping["train"][t])
    if len(valid_dicts) != 0 and len(valid_tags) != 0:
        valid_df = pd.DataFrame(valid_dicts).sort_values('num_updates')
        valid_df.to_csv(os.path.join(csvs_output_directory, 'valid.csv'))
        if set_ylim:
            ylim = max(ylim, *[valid_df.iloc[-1][tag] for tag in valid_tags])
        for t in valid_tags:
            valid_df.plot(x="epoch", y=t, ax=ax, label=tags_legends_mapping["valid"][t])
    if len(test_dicts) != 0 and len(test_tags) != 0:
        test_df = pd.DataFrame(test_dicts).sort_values('num_updates')
        test_df.to_csv(os.path.join(csvs_output_directory, 'test.csv'))
        if set_ylim:
            ylim = max(ylim, *[test_df.iloc[-1][tag] for tag in test_tags])
        for t in test_tags:
            test_df.plot(x="epoch", y=t, ax=ax, label=tags_legends_mapping["test"][t])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    if set_ylim:
        ylim = math.ceil(ylim)
        ax.set_ylim(0, set_ylim if str.isnumeric(str(set_ylim)) else ylim)
    ax.grid(True)

    for save_path in save_paths:
        fig.savefig(save_path, dpi=150)

    plt.close(fig=fig)

def plot_tfevent(event_file_paths, tags, tags_legends_mapping, y_label, csvs_output_directory, plot_output_directory, set_ylim=True, output_name=None):
    tag_dicts = {}
    for t in tags:
        tag_dicts[t] = []

    for event_file_path in event_file_paths:
        try:
            for e in tf.train.summary_iterator(event_file_path):
                for v in e.summary.value:
                    if v.tag in tag_dicts:
                        tag_dicts[v.tag].append({"Step": e.step, "Wall time": e.wall_time, "Value": v.simple_value})
        except:
            print(event_file_path, "Reading Error")
        
    
    # Check if any of the tags exist in the event files
    for t in tags:
        if len(tag_dicts[t]) == 0:
            del tag_dicts[t]
    if len(tag_dicts) == 0:
        return

    tag_dfs = {}
    for tag in tag_dicts:
        tag_dfs[tag] = pd.DataFrame(tag_dicts[tag]).sort_values('Wall time')
    
    fig = plt.figure()

    save_paths = [ os.path.join(plot_output_directory, p_name) for p_name in [("&".join(tags) if output_name is None else output_name)+".png", ("&".join(tags) if output_name is None else output_name) +".svg"] ]

    ax = fig.add_subplot(111)

    for tag in tag_dfs:
        tag_dfs[tag].to_csv(os.path.join(csvs_output_directory, tag+'.csv'))
        tag_dfs[tag].plot(x="Step", y="Value", ax=ax, label=tags_legends_mapping[tag])

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    if set_ylim:
        ylim = math.ceil(max([ tag_dfs[tag].iloc[-1]["Value"] for tag in tag_dfs]))
        ax.set_ylim(0, set_ylim if str.isnumeric(str(set_ylim)) else ylim)
    ax.grid(True)

    for save_path in save_paths:
        fig.savefig(save_path, dpi=150)

    plt.close(fig=fig)



r_dir = './results'
datasets = ["dbnqa1", "lc-quad1", "monument_600", "monument2_1", "monument2_2"]
models = ["neural_sparql_machine", "neural_sparql_machine_bahdanau_attention", "neural_sparql_machine_luong_attention", "fconv_wmt_en_de", "lstm_luong_wmt_en_de", "transformer_iwslt_de_en", "wmt16_gnmt_4_layer", "wmt16_gnmt_8_layer"]
runs = ["run1", "run2"]

dataset_folders = [ os.path.join(r_dir, d) for d in datasets[2:] if os.path.isdir(os.path.join(r_dir, d))]

for df in dataset_folders:
    run_folders = [ os.path.join(df, d) for d in runs[0:1] if os.path.isdir(os.path.join(df, d))  ]
    for rf in run_folders:
        model_folders = [ os.path.join(rf, d) for d in models if os.path.isdir(os.path.join(rf, d))  ]
        for mf in model_folders:
            files = os.listdir(mf)
            fairseq_output = find_fairseq_output(files)
            step_or_epoch = True
            if fairseq_output is not None:
                plot_fairseq_output([ os.path.join(mf, d) for d in fairseq_output ], {"train": ["ppl"], "valid": ["valid_ppl"], "test":[]}, {"train":{"ppl":"Train"}, "valid":{"valid_ppl":"Valid"}, "test":{}}, "Perplexity", mf, mf, set_ylim=True, output_name="ppls")
                # plot_fairseq_output([ os.path.join(mf, d) for d in fairseq_output ], {"train": ["ppl"], "valid": ["valid_ppl"], "test":[]}, {"train":{"ppl":"Train"}, "valid":{"valid_ppl":"Valid"}, "test":{}}, "Perplexity", mf, mf, set_ylim=True, output_name="ppls")
            else:
                ef = os.path.join(mf, 'train')
                event_files = [ os.path.join(ef, d) for d in os.listdir(ef) if "tfevents" in d ]
                if len(event_files) > 0:
                    # Plot the perplexity graphs
                    plot_tfevent(event_files, ["train_ppl", "dev_ppl"], {"train_ppl":"Train", "dev_ppl":"Valid"}, "Perplexity", mf, mf, set_ylim=True, output_name="ppls")
                    # Plot the BLEU graphs
                    plot_tfevent(event_files, ["dev_bleu", "test_bleu"], {"dev_bleu":"Valid", "test_bleu":"Test"}, "BLEU", mf, mf, set_ylim=False, output_name="bleus")

                    # plot_tfevent(event_files, ["dev_bleu", "test_bleu"], {"dev_bleu":"Valid", "test_bleu":"Test"}, "BLEU", mf, mf, set_ylim=False, output_name="bleus")
 



