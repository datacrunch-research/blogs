import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Configuration - Set to True for plots you want to generate
CONFIG = {
    "cuda_graph_benchmark": True,
    "fp8_latency_comparison": True,
    "speculative_serving": True,
    "dp_attention": True,
    "overlap_scheduler": True,
    "flashinfer_mla": True,
    "fp8_vs_bf16": True,
    "int8_gemm": True,
    "fused_moe_latency": True
}

# Create imgs directory if it doesn't exist
os.makedirs("imgs", exist_ok=True)

# Brand colors 2025
BRAND_COLORS = {
    "primary_main": "#4B7A8A",
    "primary_dark": "#245866",
    "primary_light": "#7FB1C3",
    "info_main": "#0FB4E0",
    "info_dark": "#0B8AAC",
    "info_light": "#9BE4F8",
    "success_main": "#13B91A",
    "success_dark": "#009906",
    "success_light": "#ADE6AF",
    "warning_main": "#EB8C00",
    "warning_dark": "#CC7700",
    "warning_light": "#FFD494",
    "error_main": "#FF4B1A",
    "error_dark": "#D1350A",
    "error_light": "#FFC3B3",
    "grey_500": "#909090",
    "grey_700": "#585858",
    "grey_900": "#353535",
}

# Set style for all plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    BRAND_COLORS["primary_main"], 
    BRAND_COLORS["info_main"],
    BRAND_COLORS["success_main"],
    BRAND_COLORS["warning_main"],
    BRAND_COLORS["error_main"],
    BRAND_COLORS["primary_dark"],
    BRAND_COLORS["info_dark"],
    BRAND_COLORS["success_dark"],
])

# Function to save figures
def save_figure(fig, filename):
    fig.savefig(f"imgs/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot 1: CUDA Graph Benchmark
if CONFIG["cuda_graph_benchmark"]:
    # Benchmark labels
    benchmarks = [
        'No Cuda Graph\n+ Torch Compile',
        'Cuda Graph\n+ Torch Compile',
        'Cuda Graph\n+ Torch Compile(Cuda Graph)'
    ]

    # Correct total latency and throughput values
    total_latency = [7.322, 1.256, 1.011]           # in seconds
    total_throughput = [39.34, 229.27, 284.86]      # in tokens/sec

    x = np.arange(len(benchmarks))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Latency bars (left y-axis)
    bar1 = ax1.bar(x - width/2, total_latency, width, color=BRAND_COLORS["primary_main"], label='Total Latency')
    ax1.set_ylabel('Latency (s)', color=BRAND_COLORS["primary_main"])
    ax1.tick_params(axis='y', labelcolor=BRAND_COLORS["primary_main"])

    # Throughput bars (right y-axis)
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, total_throughput, width, color=BRAND_COLORS["info_main"], label='Total Throughput')
    ax2.set_ylabel('Throughput (tokens/sec)', color=BRAND_COLORS["info_main"])
    ax2.tick_params(axis='y', labelcolor=BRAND_COLORS["info_main"])

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    plt.title("Total Latency & Throughput Across Benchmarks")

    # Create a single combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.tight_layout()
    save_figure(fig, "cuda_graph_benchmark")

# Plot 2: FP8 Latency Comparison
if CONFIG["fp8_latency_comparison"]:
    # Input types
    input_types = [
        "e4m3fn + e4m3fn",         # large latency case
        "e5m2 + e4m3fn",
        "e4m3fn + e5m2fn",
        "e4m3fn + e4m3fn (v2)",
        "e5m2 + e4m3fn (v2)",
        "e4m3fn + e5m2"
    ]

    # Latencies
    bf16_latencies = [0.05708, 0.00007, 0.00004, 0.00005, 0.00004, 0.00004]
    fp8_latencies  = [0.06103, 0.00405, 0.00344, 0.00736, 0.00372, 0.00367]

    x = np.arange(len(input_types))
    width = 0.35

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), 
                                            gridspec_kw={'height_ratios': [1, 2]},
                                            sharex=True)

    # Top axis (large bar only)
    ax_top.bar(x - width/2, bf16_latencies, width, label='BF16 Latency', color=BRAND_COLORS["primary_main"])
    ax_top.bar(x + width/2, fp8_latencies, width, label='FP8 Latency', color=BRAND_COLORS["info_main"])

    # Bottom axis (zoomed in smaller bars)
    ax_bottom.bar(x - width/2, bf16_latencies, width, color=BRAND_COLORS["primary_main"])
    ax_bottom.bar(x + width/2, fp8_latencies, width, color=BRAND_COLORS["info_main"])

    # Set axis limits to break after the first bar
    ax_top.set_ylim(0.02, 0.065)       # Shows the large bar clearly
    ax_bottom.set_ylim(0, 0.008)       # Shows smaller bars clearly

    # Hide spines between the two axes
    ax_top.spines.bottom.set_visible(False)
    ax_bottom.spines.top.set_visible(False)

    # Diagonal lines to indicate broken axis
    d = .005
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels, titles, ticks
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(input_types, rotation=30, ha='right')

    ax_top.set_title("BF16 vs FP8 Latencies with Broken Y-axis")
    ax_bottom.set_ylabel("Latency (s)")

    ax_top.legend()

    plt.tight_layout()
    save_figure(fig, "fp8_latency_comparison")

# Plot 3: Speculative Serving
if CONFIG["speculative_serving"]:
    # Labels for the two benchmarks
    benchmarks = ['Standard\nServing', 'Speculative\nServing']

    # Metrics from logs
    latency_ms = [4556.85, 746.72]            # End-to-End Latency (ms)
    throughput_tokps = [63.18, 385.11]        # Total Token Throughput (tokens/sec)

    x = np.arange(len(benchmarks))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot latency (left Y-axis)
    bars1 = ax1.bar(x - width/2, latency_ms, width, label='E2E Latency (ms)', color=BRAND_COLORS["primary_main"])
    ax1.set_ylabel('Latency (ms)', color=BRAND_COLORS["primary_main"])
    ax1.tick_params(axis='y', labelcolor=BRAND_COLORS["primary_main"])

    # Plot throughput (right Y-axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, throughput_tokps, width, label='Token Throughput (tok/s)', color=BRAND_COLORS["info_main"])
    ax2.set_ylabel('Throughput (tokens/sec)', color=BRAND_COLORS["info_main"])
    ax2.tick_params(axis='y', labelcolor=BRAND_COLORS["info_main"])

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    plt.title("Serving Performance: Standard vs Speculative Decoding")

    # Create a single combined legend (matching first plot's style)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.tight_layout()
    save_figure(fig, "speculative_serving")

# Plot 4: DP Attention
if CONFIG["dp_attention"]:
    # Benchmarks (x-axis labels)
    benchmarks = [
        'Prefill\nStandard',
        'Prefill\nDP Attention',
        'Decode\nStandard',
        'Decode\nDP Attention'
    ]

    # Reorder the data to match the new benchmark order
    latency_ms = [242905.21, 204997.47, 532565.15, 401309.94]  # Reordered
    throughput_tokps = [10770.74, 13132.59, 5518.66, 8966.88]  # Reordered

    x = np.arange(len(benchmarks))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-axis: Latency
    bars1 = ax1.bar(x - width/2, latency_ms, width, label='E2E Latency (ms)', color=BRAND_COLORS["primary_main"])
    ax1.set_ylabel('Latency (ms)', color=BRAND_COLORS["primary_main"])
    ax1.tick_params(axis='y', labelcolor=BRAND_COLORS["primary_main"])

    # Right Y-axis: Throughput
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, throughput_tokps, width, label='Token Throughput (tok/s)', color=BRAND_COLORS["info_main"])
    ax2.set_ylabel('Throughput (tokens/sec)', color=BRAND_COLORS["info_main"])
    ax2.tick_params(axis='y', labelcolor=BRAND_COLORS["info_main"])

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    plt.title("Serving Performance: Standard vs DP-Attention (Prefill & Decode)")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    save_figure(fig, "dp_attention")

# Plot 5: Overlap scheduler with DP
if CONFIG['overlap_scheduler']:
    # Labels
    benchmarks = ["Standard", "Overlap-scheduler"]

    # Data
    times = ["Mean E2E", "Mean TTFT", "Mean TPOT", "Mean ITL"]
    standard_latency = [1080152.26, 724050.93, 348.10, 350.62]
    overlap_latency = [1066166.84, 864850.92, 196.79, 197.96]

    x = np.arange(len(times))
    width = 0.35

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), 
                                            gridspec_kw={'height_ratios': [1, 2]},
                                            sharex=True)

    # Top axis (large bar only)
    ax_top.bar(x - width/2, standard_latency, width, label='Standard Latency', color=BRAND_COLORS["primary_main"])
    ax_top.bar(x + width/2, overlap_latency, width, label='Overlap Scheduler Latency', color=BRAND_COLORS["info_main"])

    # Bottom axis (zoomed in smaller bars)
    ax_bottom.bar(x - width/2, standard_latency, width, color=BRAND_COLORS["primary_main"])
    ax_bottom.bar(x + width/2, overlap_latency, width, color=BRAND_COLORS["info_main"])

    # Set axis limits to break after the first bar
    ax_top.set_ylim(500000, 1100000)       # Shows the large bar clearly
    ax_bottom.set_ylim(0, 400)       # Shows smaller bars clearly

    # Hide spines between the two axes
    ax_top.spines.bottom.set_visible(False)
    ax_bottom.spines.top.set_visible(False)

    # Diagonal lines to indicate broken axis
    d = .005
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels, titles, ticks
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(times, rotation=30, ha='right')

    ax_top.set_title("Standard vs Overlap Scheduler mean latencies with Broken Y-axis")
    ax_bottom.set_ylabel("Latency (s)")

    ax_top.legend()

    plt.tight_layout()
    save_figure(fig, "ovelap_scheduler_latency")

# Plot 6: FlashInfer MLA
if CONFIG["flashinfer_mla"]:
    # Labels
    benchmarks = ['Standard', 'FlashInfer-MLA']

    # Data from logs
    latency_sec = [77.397, 71.480]
    throughput_tokps = [1809.790, 1920.021]

    x = np.arange(len(benchmarks))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Latency (left axis)
    bars1 = ax1.bar(x - width/2, latency_sec, width, label='Latency (s)', color=BRAND_COLORS["primary_main"])
    ax1.set_ylabel('Latency (seconds)', color=BRAND_COLORS["primary_main"])
    ax1.tick_params(axis='y', labelcolor=BRAND_COLORS["primary_main"])

    # Throughput (right axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, throughput_tokps, width, label='Throughput (tok/s)', color=BRAND_COLORS["info_main"])
    ax2.set_ylabel('Output Throughput (tokens/sec)', color=BRAND_COLORS["info_main"])
    ax2.tick_params(axis='y', labelcolor=BRAND_COLORS["info_main"])

    # X-axis and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    plt.title("GSM8K Benchmark: Standard vs FlashInfer-MLA")

    # Set y-axis limits with some padding
    ax1.set_ylim(0, max(latency_sec) * 1.2)  # 20% padding above max latency
    ax2.set_ylim(0, max(throughput_tokps) * 1.2)  # 20% padding above max throughput

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.tight_layout()
    save_figure(fig, "flashinfer_mla")

# Plot 7: FP8 vs BF16
if CONFIG["fp8_vs_bf16"]:
    # New benchmark labels
    benchmarks = ['Baseline BF16', 'Current FP8']

    # New Data from your provided results
    latency_sec = [109.212, 84.355]
    throughput_tokps = [1244.611, 1605.663]

    x = np.arange(len(benchmarks))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Latency (left axis)
    bars1 = ax1.bar(x - width/2, latency_sec, width, label='Latency (s)', color=BRAND_COLORS["primary_main"])
    ax1.set_ylabel('Latency (seconds)', color=BRAND_COLORS["primary_main"])
    ax1.tick_params(axis='y', labelcolor=BRAND_COLORS["primary_main"])

    # Throughput (right axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, throughput_tokps, width, label='Throughput (tok/s)', color=BRAND_COLORS["info_main"])
    ax2.set_ylabel('Output Throughput (tokens/sec)', color=BRAND_COLORS["info_main"])
    ax2.tick_params(axis='y', labelcolor=BRAND_COLORS["info_main"])

    # X-axis and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks)
    plt.title("GSM8K Benchmark: Baseline BF16 vs Current FP8")

    # Set y-axis limits with padding
    ax1.set_ylim(0, max(latency_sec) * 1.2)  
    ax2.set_ylim(0, max(throughput_tokps) * 1.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.tight_layout()
    save_figure(fig, "fp8_vs_bf16")

# Plot 8: INT8 GEMM
if CONFIG["int8_gemm"]:
    # Organizing the data
    data = {
        "N": [3072, 4096, 2048, 576, 21888, 2048, 2816, 2048],
        "K": [2048, 2048, 2048, 2048, 2048, 10944, 2048, 1408],
        "batch_sizes": [1, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "vllm": [
            [1120.091, 18127.45, 36151.01, 72928.93, 121023.5, 217529.4, 428592.9, 655413.2, 1020821],
            [1452.199, 23107.52, 46342.35, 93717.39, 161364.7, 285123.2, 559576.6, 862680.6, 1134484],
            [770.815, 12369.31, 24309.64, 49477.27, 88538.27, 149531.3, 293198.6, 580078.3, 871056.2],
            [233.943, 3778.96, 7606.53, 14513.08, 26431.64, 43406.09, 86812.18, 172437.87, 336445.45],
            [3217.856, 51426.78, 103207.8, 201330.4, 454584.1, 760203.1, 1038487, 1332380, 1436509],
            [2025.341, 32358.7, 64438.44, 134078.2, 231778.6, 329470.3, 614898.4, 1152749, 1250151],
            [1035.576, 16569.21, 32762.92, 66658.86, 109884.2, 200699.6, 395395.3, 598851.1, 944107.1],
            [581.762, 9338.22, 18321.82, 36818.42, 65605.62, 110543.12, 222146.61, 445361.24, 695198.04],
        ],
        "sgl_kernel": [
            [1166.485, 18608.71, 37217.42, 75099.43, 147563.8, 221345.7, 436942.1, 665131.0, 1040553],
            [1501.989, 23895.27, 47926.7, 95581.09, 187958.4, 288794.4, 566644.9, 875305.2, 1158908],
            [803.817, 12782.88, 25488.29, 50822.58, 101953.16, 151552.0, 298399.5, 587677.6, 881035.5],
            [242.38, 3878.09, 7756.17, 14831.55, 30822.57, 44217.42, 87011.75, 174827.30, 340992.0],
            [3380.491, 54448.22, 108567.6, 205707.1, 382121.2, 768324.9, 1043005, 1326849, 1425475],
            [2238.875, 35707.92, 71076.31, 146805.7, 283407.0, 330685.0, 618609.0, 1140837, 1248519],
            [1082.084, 17108.45, 34115.97, 68231.93, 135663.48, 204243.93, 400530.29, 608700.66, 957541.99],
            [595.158, 9553.95, 18616.39, 37473.76, 74465.54, 110279.92, 223217.20, 443230.30, 707138.48],
        ]
    }

    # Flattening into a long dataframe
    records = []
    for i, (n, k) in enumerate(zip(data["N"], data["K"])):
        for j, batch in enumerate(data["batch_sizes"]):
            records.append((f"N={n}, K={k}", batch, data["vllm"][i][j], data["sgl_kernel"][i][j]))

    df = pd.DataFrame(records, columns=["Shape", "Batch Size", "vLLM Int8 GEMM (GB/s)", "SGL-Kernel Int8 GEMM (GB/s)"])

    # Plotting: pick a few shapes for separate subplots
    shapes_to_plot = ["N=3072, K=2048", "N=2048, K=10944", "N=2048, K=1408"]
    fig, axes = plt.subplots(len(shapes_to_plot), 1, figsize=(8, 12))

    for ax, shape in zip(axes, shapes_to_plot):
        subset = df[df["Shape"] == shape]
        ax.plot(subset["Batch Size"], subset["vLLM Int8 GEMM (GB/s)"], marker='o', label="vLLM", color=BRAND_COLORS["primary_main"], linewidth=2)
        ax.plot(subset["Batch Size"], subset["SGL-Kernel Int8 GEMM (GB/s)"], marker='s', label="SGL-Kernel", color=BRAND_COLORS["info_main"], linewidth=2)
        ax.set_title(f"INT8 GEMM Throughput - {shape}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (GB/s)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save_figure(fig, "int8_gemm_comparison")

# Plot 9: Fused MoE Latency (vLLM vs SGLang)
if CONFIG["fused_moe_latency"]:
    fused_moe_df = pd.read_csv("fused-moe-performance.csv")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(fused_moe_df["batch_size"], fused_moe_df["vllm_fused_moe_triton"],
            label="vLLM Fused MoE (Triton)", color=BRAND_COLORS["primary_main"], linewidth=4)
    ax.plot(fused_moe_df["batch_size"], fused_moe_df["sglang_fused_moe_triton"],
            label="SGLang Fused MoE (Triton)", color=BRAND_COLORS["info_main"], linewidth=4)

    ax.set_title("Fused MoE Latency: vLLM vs SGLang (Triton Kernels)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (s)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, "fused_moe_latency_comparison")


print("Plots generated and saved to the 'imgs' directory.")
