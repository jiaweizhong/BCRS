"""
Script to generate high-quality, publication-ready vector PDF diagrams for DES-ESOD.
Self-contained inside the icip/ paper directory.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["mathtext.fontset"] = "cm"


def create_system_overview(save_path):
    fig, ax = plt.subplots(figsize=(12.8, 4.0), dpi=300)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 38)
    ax.axis("off")

    # Color Palette (Publication Grade)
    c_img = "#FFF7ED"
    b_img = "#EA580C"
    c_stem = "#EFF6FF"
    b_stem = "#2563EB"
    c_dual = "#FAF5FF"
    b_dual = "#9333EA"
    c_slicer = "#FEFCE8"
    b_slicer = "#CA8A04"
    c_backbone = "#F0FDF4"
    b_backbone = "#16A34A"
    c_head = "#FDF2F8"
    b_head = "#DB2777"

    def draw_box(
        x,
        y,
        w,
        h,
        bg,
        border,
        title,
        subtitle="",
        radius=1.0,
        title_color="#1E293B",
        sub_color="#475569",
        title_fs=9.5,
        sub_fs=7.5,
        lw=1.5,
    ):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg,
            edgecolor=border,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.62,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_fs,
                color=sub_color,
                zorder=3,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )

    # 1. Input Image
    draw_box(
        1.5,
        13,
        10.5,
        12,
        c_img,
        b_img,
        "High-Res Image",
        r"$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$",
        title_color="#9A3412",
        sub_color="#C2410C",
    )

    # Arrow 1->2
    ax.annotate(
        "",
        xy=(15.2, 19),
        xytext=(12.3, 19),
        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=14),
    )

    # 2. Shallow Stem
    draw_box(
        15.5,
        13,
        11.5,
        12,
        c_stem,
        b_stem,
        "Shallow Stem",
        "Focus & Conv Layers\n"
        + r"$\mathbf{F} \in \mathbb{R}^{H_s \times W_s \times C}$",
        title_color="#1E40AF",
        sub_color="#1D4ED8",
    )

    # Big bounding box encompassing Dual-Evidence Priority Head + AdaSlicer = Dual-Evidence Spatial Selector
    selector_bg = patches.FancyBboxPatch(
        (31.5, 6.5),
        39.5,
        25.5,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FAFAFA",
        edgecolor="#A855F7",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(selector_bg)
    ax.text(
        51.25,
        30.5,
        "Dual-Evidence Spatial Selector (Proposed Subsystem)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#7E22CE",
    )

    # Arrow from Stem into Dual-Evidence Head
    ax.annotate(
        "",
        xy=(33.2, 19),
        xytext=(27.3, 19),
        arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.8, mutation_scale=14),
    )

    # Bypass arrow from Stem (features F) directly into AdaSlicer
    ax.annotate(
        "",
        xy=(59.0, 24.5),
        xytext=(27.3, 23.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2563EB",
            lw=1.2,
            linestyle=":",
            connectionstyle="arc3,rad=-0.15",
            mutation_scale=12,
        ),
    )
    ax.text(
        42,
        27.5,
        r"Shallow Features $\mathbf{F}$",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#1D4ED8",
        style="italic",
    )

    # 3. Dual-Evidence Priority Head
    draw_box(
        33.5,
        8.5,
        18,
        18,
        c_dual,
        b_dual,
        "Dual-Evidence Priority Head",
        r"Semantic Branch ($z^{\mathrm{sem}}$)"
        + "\n+\n"
        + r"Spectral Branch ($z^{\mathrm{spec}}$)"
        + "\n"
        + r"$\mathbf{S} = \sigma(\mathrm{Conv}_{1\times 1}([z^{\mathrm{sem}}\|z^{\mathrm{spec}}]))$",
        title_color="#6B21A8",
        sub_color="#7E22CE",
        title_fs=9.0,
        sub_fs=7.5,
        lw=1.8,
    )

    # Arrow Dual-Evidence Head -> AdaSlicer with Priority Map S label
    ax.annotate(
        "",
        xy=(59.0, 16.5),
        xytext=(51.8, 16.5),
        arrowprops=dict(arrowstyle="-|>", color="#9333EA", lw=1.8, mutation_scale=14),
    )
    ax.text(
        55.4,
        18.5,
        r"$\mathbf{S} \in [0, 1]$",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#9333EA",
    )

    # 4. AdaSlicer (Patch Router)
    draw_box(
        59.5,
        11.5,
        10.0,
        14,
        c_slicer,
        b_slicer,
        "AdaSlicer",
        r"$\mathcal{S} = \{i : s_i > \tau\}$" + "\nDynamic Patch\nSlicing & Routing",
        title_color="#854D0E",
        sub_color="#A16207",
        title_fs=9.0,
        sub_fs=7.2,
    )

    # Arrow AdaSlicer -> Backbone (Selected Patches)
    ax.annotate(
        "",
        xy=(74.5, 19),
        xytext=(70.0, 19),
        arrowprops=dict(arrowstyle="-|>", color="#CA8A04", lw=1.8, mutation_scale=14),
    )
    ax.text(
        72.2,
        21.2,
        r"Patches $\mathcal{P}$",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#854D0E",
    )

    # 5. Deep Backbone & Neck
    draw_box(
        74.8,
        13,
        11.5,
        12,
        c_backbone,
        b_backbone,
        "Deep Backbone\n& PANet Neck",
        "Evaluates Selected\nPatches Only (Sparse)",
        title_color="#166534",
        sub_color="#15803D",
        title_fs=9,
        sub_fs=7.5,
    )

    # Arrow Backbone -> Head
    ax.annotate(
        "",
        xy=(89.5, 19),
        xytext=(86.6, 19),
        arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.8, mutation_scale=14),
    )

    # 6. Decoupled Head + SABL
    draw_box(
        89.8,
        11.5,
        10.5,
        15,
        c_head,
        b_head,
        "ISPP Decoupled\nDetection Head",
        "Partial Conv Stem\n+\n"
        + r"$\mathcal{L}_{\mathrm{box}}$ (SABL) & $\mathcal{L}_{\mathrm{cls}}$",
        title_color="#9D174D",
        sub_color="#BE185D",
        title_fs=8.5,
        sub_fs=7.2,
        lw=1.8,
    )

    # Supervisory loss annotations at the bottom
    loss_box = patches.FancyBboxPatch(
        (20, 1.2),
        62,
        4.2,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(loss_box)
    ax.text(
        51,
        3.3,
        r"$\mathbf{Joint \; Multi-Task \; Training \; Optimization:}$  "
        + r"$\mathcal{L} = \mathcal{L}_{\mathrm{det}} + \lambda_{\mathrm{sel}}(\mathcal{L}_{\mathrm{BCE}} + \lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cover}})$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#334155",
    )

    # Connecting dashed lines from losses to modules
    ax.plot([42.5, 42.5], [8.5, 5.5], color="#9333EA", linestyle="--", lw=1.2)
    ax.plot(
        [95.0, 95.0, 82.0, 82.0],
        [11.5, 3.3, 3.3, 5.5],
        color="#DB2777",
        linestyle="--",
        lw=1.2,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


def create_segmenter_diagram(save_path):
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Colors
    c_in = "#F8FAFC"
    b_in = "#64748B"
    c_sem = "#EFF6FF"
    b_sem = "#2563EB"
    c_spec = "#ECFDF5"
    b_spec = "#059669"
    c_fuse = "#FAF5FF"
    b_fuse = "#7C3AED"

    def draw_node(
        x,
        y,
        w,
        h,
        bg,
        border,
        title,
        subtitle="",
        radius=1.0,
        title_color="#1E293B",
        sub_color="#475569",
        title_fs=8.5,
        sub_fs=7,
        lw=1.3,
    ):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg,
            edgecolor=border,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.62,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_fs,
                color=sub_color,
                zorder=3,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )

    # Input shallow feature
    draw_node(
        1.5,
        17,
        13.5,
        16,
        c_in,
        b_in,
        "Shallow Features",
        r"$\mathbf{F} \in \mathbb{R}^{H_s \times W_s \times C}$",
        title_color="#334155",
        sub_color="#475569",
    )

    # Fork paths
    ax.annotate(
        "",
        xy=(19.5, 38),
        xytext=(15.2, 28),
        arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.5, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(19.5, 12),
        xytext=(15.2, 22),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.5, mutation_scale=12),
    )

    # --- TOP: Semantic Branch ---
    draw_node(
        20,
        32,
        28,
        12,
        c_sem,
        b_sem,
        "Semantic Objectness Branch",
        r"$\mathrm{Conv}_{1\times 1} \to z^{\mathrm{sem}} \in \mathbb{R}^{H_s \times W_s \times N_c}$",
        title_color="#1E40AF",
        sub_color="#1D4ED8",
    )

    # --- BOTTOM: Spectral Branch ---
    # Step 1: Channel-wise pooling
    draw_node(
        20,
        6,
        17,
        12,
        c_spec,
        b_spec,
        "Channel Pooling",
        r"$\mathrm{Max}+\mathrm{Mean} \; (C \to 2)$",
        title_color="#065F46",
        sub_color="#047857",
    )

    ax.annotate(
        "",
        xy=(40.5, 12),
        xytext=(37.2, 12),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.4, mutation_scale=12),
    )

    # Step 2: Depthwise Laplacian & Sobel
    draw_node(
        41,
        6,
        19,
        12,
        c_spec,
        b_spec,
        "Trainable Filter Bank",
        r"$\mathrm{DWConv}_{3\times 3} \; (2 \to 6)$" + "\nLaplacian + Sobel-x/y",
        title_color="#065F46",
        sub_color="#047857",
        title_fs=8,
        sub_fs=6.8,
    )

    ax.annotate(
        "",
        xy=(63.5, 12),
        xytext=(60.2, 12),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.4, mutation_scale=12),
    )

    # Step 3: Projection & Spectral Logits
    draw_node(
        64,
        6,
        17,
        12,
        c_spec,
        b_spec,
        "Spectral Head",
        r"$\mathrm{Conv}_{1\times 1} \to z^{\mathrm{spec}}$",
        title_color="#065F46",
        sub_color="#047857",
    )

    # Join arrows to Concat
    ax.annotate(
        "",
        xy=(84.5, 27),
        xytext=(48.5, 38),
        arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.5, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(84.5, 23),
        xytext=(81.3, 12),
        arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.5, mutation_scale=12),
    )

    # Concat & Fusion
    concat_circle = patches.Circle(
        (86.5, 25),
        radius=2.6,
        facecolor="#EDE9FE",
        edgecolor="#7C3AED",
        lw=1.5,
        zorder=4,
    )
    ax.add_patch(concat_circle)
    ax.text(
        86.5,
        25,
        r"$\mathbf{C}$",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#7C3AED",
        zorder=5,
    )

    ax.annotate(
        "",
        xy=(91.5, 25),
        xytext=(89.2, 25),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.5, mutation_scale=12),
    )

    # Output Fusion
    draw_node(
        91.8,
        17,
        7.8,
        16,
        c_fuse,
        b_fuse,
        r"$\mathrm{Conv}_{1\times 1}$" + "\nFusion",
        r"$s_i = \sigma(u_{i,0})$" + "\n" + r"$\in [0, 1]$",
        title_color="#5B21B6",
        sub_color="#6D28D9",
        title_fs=8,
        sub_fs=7,
        lw=1.5,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


def create_sabl_diagram(save_path):
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    c_box = "#F8FAFC"
    b_box = "#64748B"
    c_w = "#FFF7ED"
    b_w = "#EA580C"
    c_ciou = "#F1F5F9"
    b_ciou = "#475569"
    c_gate = "#FEF3C7"
    b_gate = "#D97706"
    c_sum = "#FDF2F8"
    b_sum = "#DB2777"

    def draw_node(
        x,
        y,
        w,
        h,
        bg,
        border,
        title,
        subtitle="",
        radius=1.0,
        title_color="#1E293B",
        sub_color="#475569",
        title_fs=8.5,
        sub_fs=7,
        lw=1.3,
    ):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg,
            edgecolor=border,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.62,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_fs,
                color=sub_color,
                zorder=3,
            )
        else:
            ax.text(
                x + w / 2,
                y + h / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=3,
            )

    # 1. Inputs
    draw_node(
        1.5,
        27,
        13,
        14,
        c_box,
        b_box,
        "Predicted Box",
        r"$p_{\mathrm{box}} = (c_x, c_y, w, h)$",
        title_color="#334155",
        sub_color="#475569",
    )
    draw_node(
        1.5,
        7,
        13,
        14,
        c_box,
        b_box,
        "Ground Truth",
        r"$t_{\mathrm{box}} = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})$",
        title_color="#334155",
        sub_color="#475569",
    )

    # Arrows to Scale Gate & Wasserstein
    ax.annotate(
        "",
        xy=(20, 24),
        xytext=(14.8, 14),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.4, mutation_scale=12),
    )

    # Scale computation & Gate
    draw_node(
        20.5,
        17,
        21,
        14,
        c_gate,
        b_gate,
        "Scale Gating Factor",
        r"$s = \sqrt{\hat{w}\hat{h}}$" + "\n" + r"$\mu(s) = \exp(-(s/32)^6)$",
        title_color="#92400E",
        sub_color="#B45309",
    )

    # Arrow from Gate to Wasserstein & CIoU
    ax.annotate(
        "",
        xy=(47, 36),
        xytext=(41.8, 26),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.4, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(47, 12),
        xytext=(41.8, 22),
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.4, mutation_scale=12),
    )

    # Direct arrows from boxes to terms
    ax.annotate(
        "",
        xy=(47, 40),
        xytext=(14.8, 34),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.4, mutation_scale=12),
    )

    # Top: Wasserstein Penalty
    draw_node(
        47.5,
        30,
        27,
        14,
        c_w,
        b_w,
        "Normalized Wasserstein Penalty",
        r"$\mathcal{L}_W = \mu(s) \cdot (1 - e^{-D_W/12})$"
        + "\n"
        + r"(Dominant for Tiny Targets, $s \rightarrow 0$)",
        title_color="#9A3412",
        sub_color="#C2410C",
        title_fs=8.5,
        sub_fs=7,
        lw=1.5,
    )

    # Bottom: CIoU Center Distance Penalty
    draw_node(
        47.5,
        4,
        27,
        14,
        c_ciou,
        b_ciou,
        "CIoU Center-Distance Penalty",
        r"$\mathcal{L}_{\mathrm{ctr}} = (1 - \mu(s)) \cdot \ell_{\mathrm{ctr}}$"
        + "\n"
        + r"(Dominant for Large Targets, $s \gg 32$)",
        title_color="#1E293B",
        sub_color="#475569",
        title_fs=8.5,
        sub_fs=7,
    )

    # Base CIoU overlap
    ax.text(
        61,
        23.5,
        r"$+ \; (1 - \mathrm{IoU} + \alpha v)$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#475569",
    )

    # Arrows to Summation
    ax.annotate(
        "",
        xy=(81.5, 25),
        xytext=(75.0, 35),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.5, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(81.5, 23),
        xytext=(75.0, 13),
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.5, mutation_scale=12),
    )

    # Summation node
    sum_circle = patches.Circle(
        (84, 24),
        radius=2.6,
        facecolor="#FCE7F3",
        edgecolor="#DB2777",
        lw=1.6,
        zorder=4,
    )
    ax.add_patch(sum_circle)
    ax.text(
        84,
        24,
        r"$+$",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#DB2777",
        zorder=5,
    )

    ax.annotate(
        "",
        xy=(90.5, 24),
        xytext=(86.8, 24),
        arrowprops=dict(arrowstyle="-|>", color="#DB2777", lw=1.6, mutation_scale=12),
    )

    # Output Box Loss
    draw_node(
        91,
        17,
        8,
        14,
        c_sum,
        b_sum,
        r"$\mathcal{L}_{\mathrm{box}}$",
        "Total Box\nLoss",
        title_color="#831843",
        sub_color="#9D174D",
        title_fs=9.5,
        sub_fs=7.5,
        lw=1.6,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    create_system_overview(os.path.join(base_dir, "hesod_overview.pdf"))
    create_segmenter_diagram(os.path.join(base_dir, "hesod_segmenter_block.pdf"))
    create_sabl_diagram(os.path.join(base_dir, "hesod_sabl_block.pdf"))
