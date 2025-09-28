import re
import numpy as np
import matplotlib.pyplot as plt

def parse_summary_file(filename):
    """
    Reads the file line by line, grouping lines under each block that starts with "Running : ...".
    For each block, extracts:
      - Cycle
      - Avg Power
      - Idle FU Cycles
      - Num of Registers
    and returns a list of dictionaries.
    """
    blocks = []
    current_lines = []
    inside_block = False

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            # Start of a new block
            if line_stripped.startswith("Running :"):
                if current_lines:
                    blocks.append("\n".join(current_lines))
                    current_lines = []
                inside_block = True
                current_lines.append(line_stripped)
            else:
                if inside_block:
                    current_lines.append(line_stripped)
        if current_lines:
            blocks.append("\n".join(current_lines))

    cycle_pat = re.compile(r"Cycle\s*:\s*([\d.]+)")
    power_pat = re.compile(r"Avg Power\s*:\s*([\d.]+)")
    idle_pat  = re.compile(r"Idle FU Cycles\s*:\s*([\d.]+)")
    reg_pat   = re.compile(r"Num of Registers.*:\s*([\d.]+)")

    results = []
    for block_text in blocks:
        cycle_match = cycle_pat.search(block_text)
        power_match = power_pat.search(block_text)
        idle_match  = idle_pat.search(block_text)
        reg_match   = reg_pat.search(block_text)
        if cycle_match and power_match and idle_match and reg_match:
            try:
                cycle_val = float(cycle_match.group(1))
                power_val = float(power_match.group(1))
                idle_val  = float(idle_match.group(1))
                reg_val   = float(reg_match.group(1))
                results.append({
                    "Cycle": cycle_val,
                    "Avg Power": power_val,
                    "Idle FU Cycles": idle_val,
                    "Num of Registers": reg_val
                })
            except ValueError:
                pass
    return results

def compute_overall_metrics(blocks):
    """
    Aggregates:
      - Total Cycles = sum of block "Cycle"
      - Weighted Avg Power = sum(Cycle_i * Avg Power_i) / sum(Cycle_i)
      - Total Idle FU Cycles = sum of block "Idle FU Cycles"
      - Max Num of Registers = maximum across blocks
    """
    if not blocks:
        return {
            "Total Cycles": 0.0,
            "Weighted Avg Power": 0.0,
            "Total Idle FU Cycles": 0.0,
            "Max Num of Registers": 0.0
        }

    cycles = np.array([b["Cycle"] for b in blocks])
    powers = np.array([b["Avg Power"] for b in blocks])
    idle   = np.array([b["Idle FU Cycles"] for b in blocks])
    regs   = np.array([b["Num of Registers"] for b in blocks])

    total_cycles = cycles.sum()
    total_idle   = idle.sum()
    max_regs     = regs.max()

    if total_cycles > 0:
        weighted_power = np.sum(cycles * powers) / total_cycles
    else:
        weighted_power = 0.0

    return {
        "Total Cycles": total_cycles,
        "Weighted Avg Power": weighted_power,
        "Total Idle FU Cycles": total_idle,
        "Max Num of Registers": max_regs
    }

def main():
    # Files for VGG pair
    noatt_vgg_file = "nnet_fwd_summary_noatt_vgg.txt"
    att_vgg_file   = "nnet_fwd_summary_att_vgg_2003.txt"
    # Files for Lenet pair
    noatt_lenet_file = "nnet_fwd_summary_noatt_lenet.txt"
    att_lenet_file   = "nnet_fwd_summary_att_lenet.txt"

    # Parse files for VGG
    blocks_noatt_vgg = parse_summary_file(noatt_vgg_file)
    blocks_att_vgg   = parse_summary_file(att_vgg_file)
    # Parse files for Lenet
    blocks_noatt_lenet = parse_summary_file(noatt_lenet_file)
    blocks_att_lenet   = parse_summary_file(att_lenet_file)

    # Compute aggregated metrics for VGG
    agg_noatt_vgg = compute_overall_metrics(blocks_noatt_vgg)
    agg_att_vgg   = compute_overall_metrics(blocks_att_vgg)
    # Compute aggregated metrics for Lenet
    agg_noatt_lenet = compute_overall_metrics(blocks_noatt_lenet)
    agg_att_lenet   = compute_overall_metrics(blocks_att_lenet)

    # Optional: print out metrics for debugging
    print("VGG No Attack overall metrics:")
    for k, v in agg_noatt_vgg.items():
        print(f"  {k}: {v:.1f}")
    print("\nVGG Attack overall metrics:")
    for k, v in agg_att_vgg.items():
        print(f"  {k}: {v:.1f}")
    print("\nLenet No Attack overall metrics:")
    for k, v in agg_noatt_lenet.items():
        print(f"  {k}: {v:.1f}")
    print("\nLenet Attack overall metrics:")
    for k, v in agg_att_lenet.items():
        print(f"  {k}: {v:.1f}")

    # Define the metrics to compare
    metric_names = [
        "Total\nCycles",
        "Weighted\nAvg Power",
        "Total Idle\nFU Cycles",
        "Max Num\nof Registers"
    ]
    # Reference (No Attack) is always 1.0 for each feature
    norm_ref = [1.0 for _ in metric_names]
    norm_vgg = []
    norm_lenet = []

    # For each metric, compute normalized ratio (attack/noattack) for each pair.
    for m in metric_names:
        dict_key = m.replace("\n", " ")
        # VGG normalization
        val_noatt_vgg = agg_noatt_vgg[dict_key]
        val_att_vgg   = agg_att_vgg[dict_key]
        ratio_vgg = (val_att_vgg / val_noatt_vgg) if val_noatt_vgg != 0 else 0.0
        norm_vgg.append(ratio_vgg)
        # Lenet normalization
        val_noatt_lenet = agg_noatt_lenet[dict_key]
        val_att_lenet   = agg_att_lenet[dict_key]
        ratio_lenet = (val_att_lenet / val_noatt_lenet) if val_noatt_lenet != 0 else 0.0
        norm_lenet.append(ratio_lenet)

    # Plot settings: each metric group will have 3 bars:
    # the reference (No Attack), Tiling Attack on Lenet5, and Tiling Attack on VGG16.
    spacing_factor = 0.6
    x = np.arange(len(metric_names)) * spacing_factor
    bar_width = 0.2

    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 7.5})


    # Increase spacing_factor to get more space between groups
    spacing_factor = 1.0
    x = np.arange(len(metric_names)) * spacing_factor
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(4, 2))
    
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(0.5)

    bars_ref = ax.bar(x - bar_width, norm_ref, bar_width,
                    color='grey', edgecolor='black', label="No Attack")
    bars_lenet = ax.bar(x, norm_lenet, bar_width,
                        color='firebrick', edgecolor='black', 
                        label="Tiling Attack\non LeNet-5")
    bars_vgg = ax.bar(x + bar_width, norm_vgg, bar_width,
                    color='purple', edgecolor='black', 
                    label="Tiling Attack\non VGG-16")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Normalized Value")

    # Move legend to the top-right corner (legend's upper-right corner at axes coordinate (1,1))
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def annotate_bars(bars, data_list):
        for bar, val in zip(bars, data_list):
            height = bar.get_height()
            ax.annotate(f"{val:.1f}",
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)

    annotate_bars(bars_ref, norm_ref)
    annotate_bars(bars_lenet, norm_lenet)
    annotate_bars(bars_vgg, norm_vgg)

    plt.tight_layout()
    plt.savefig("nnet_fwd_summary_comparison.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
