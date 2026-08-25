"""
Script to generate high-quality, publication-ready vector PDF diagrams for HESOD.
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
    import numpy as np

    fig, ax = plt.subplots(figsize=(11.8, 5.0), dpi=300)
    ax.set_xlim(0, 102)
    ax.set_ylim(-5, 50)
    ax.axis("off")

    angle = 30
    rad = np.radians(angle)

    def draw_3d_cube(
        x,
        y,
        w,
        h,
        d,
        c_front,
        c_top,
        c_side,
        edge_col,
        label_top="",
        label_side="",
        caption="",
        sub_caption="",
        title_color="#0F172A",
        sub_color="#334155",
        caption_y_offset=-2.5,
    ):
        dx = d * np.cos(rad)
        dy = d * np.sin(rad)

        # 1. Front face
        front_poly = patches.Polygon(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            closed=True,
            facecolor=c_front,
            edgecolor=edge_col,
            linewidth=1.3,
            zorder=3,
        )
        ax.add_patch(front_poly)

        # 2. Top face
        top_poly = patches.Polygon(
            [
                [x, y + h],
                [x + w, y + h],
                [x + w + dx, y + h + dy],
                [x + dx, y + h + dy],
            ],
            closed=True,
            facecolor=c_top,
            edgecolor=edge_col,
            linewidth=1.3,
            zorder=3,
        )
        ax.add_patch(top_poly)

        # 3. Right side face
        side_poly = patches.Polygon(
            [
                [x + w, y],
                [x + w + dx, y + dy],
                [x + w + dx, y + h + dy],
                [x + w, y + h],
            ],
            closed=True,
            facecolor=c_side,
            edgecolor=edge_col,
            linewidth=1.3,
            zorder=3,
        )
        ax.add_patch(side_poly)

        # Labels on Top face
        if label_top:
            ax.text(
                x + w / 2 + dx / 2,
                y + h + dy / 2 + 0.5,
                label_top,
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=edge_col,
                zorder=5,
            )
        # Labels on Side face
        if label_side:
            ax.text(
                x + w + dx / 2 + 0.5,
                y + h / 2 + dy / 2,
                label_side,
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#334155",
                zorder=5,
            )

        # Caption
        if caption:
            ax.text(
                x + w / 2 + dx / 2,
                y + caption_y_offset,
                caption,
                ha="center",
                va="top",
                fontsize=8.8,
                fontweight="bold",
                color=title_color,
                zorder=5,
            )
        if sub_caption:
            ax.text(
                x + w / 2 + dx / 2,
                y + caption_y_offset - 2.4,
                sub_caption,
                ha="center",
                va="top",
                fontsize=7.8,
                fontweight="bold",
                color=sub_color,
                zorder=5,
            )

        center_east = (x + w + dx, y + h / 2 + dy / 2)
        center_west = (x, y + h / 2)
        return center_east, center_west

    def draw_operator_pill(
        x,
        y,
        w,
        h,
        bg,
        border,
        title,
        subtitle="",
        title_color="#1E293B",
        sub_color="#475569",
        radius=0.8,
        title_fs=8.2,
        sub_fs=7.2,
    ):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg,
            edgecolor=border,
            linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.65,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                zorder=5,
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_fs,
                fontweight="bold",
                color=sub_color,
                zorder=5,
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
                zorder=5,
            )

    # Palettes
    p_in = ("#93C5FD", "#DBEAFE", "#60A5FA", "#1E3A8A")
    p_sem = ("#60A5FA", "#BFDBFE", "#3B82F6", "#1D4ED8")
    p_pool = ("#6EE7B7", "#D1FAE5", "#34D399", "#065F46")
    p_filt = ("#86EFAC", "#DCFCE7", "#4ADE80", "#166534")
    p_sstem = ("#5EEAD4", "#CCFBF1", "#2DD4BF", "#0F766E")
    p_shead = ("#6EE7B7", "#D1FAE5", "#34D399", "#065F46")
    p_fuse = ("#FDE68A", "#FEF3C7", "#FCD34D", "#B45309")

    # Branch Banners
    ax.text(
        38,
        46.5,
        "Semantic Objectness Branch",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1E40AF",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.3
        ),
    )
    ax.text(
        49,
        18.0,
        r"Channel-Pooled Spectral Branch ($\mathcal{O}(1)$ Complexity)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#065F46",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#059669", lw=1.3
        ),
    )

    # 1. Input Shallow Feature Map (x=2, y=14)
    e_in, _ = draw_3d_cube(
        2.0,
        13.5,
        4.2,
        13.0,
        4.8,
        *p_in,
        label_top=r"$C{=}256$",
        label_side=r"$H_s{\times}W_s$",
        caption="Shallow Feature",
        sub_caption=r"$\mathbf{F} \in \mathbb{R}^{H_s \times W_s \times 256}$",
        title_color="#1E3A8A",
        sub_color="#1E40AF",
    )

    # --- TOP PATH: Semantic Branch ---
    # Layer: 1x1 Conv Operator Block (x=22, y=32)
    draw_operator_pill(
        20.5,
        32.5,
        11.5,
        7.0,
        "#DBEAFE",
        "#2563EB",
        r"$\mathbf{1\times 1 \; Conv}$",
        r"$C \to N_c$",
        title_color="#1E40AF",
        sub_color="#1D4ED8",
    )

    # Arrow Input -> 1x1 Conv
    ax.annotate(
        "",
        xy=(20.0, 36.0),
        xytext=(e_in[0] + 0.5, e_in[1] + 2.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1D4ED8",
            lw=1.8,
            connectionstyle="arc3,rad=-0.12",
            mutation_scale=13,
        ),
    )

    # Output: Semantic Logits Tensor (x=38, y=30.5)
    e_sem, w_sem = draw_3d_cube(
        38.0,
        30.5,
        2.8,
        10.5,
        2.2,
        *p_sem,
        label_top=r"$N_c$",
        label_side=r"$H_s{\times}W_s$",
        caption="Semantic Logits",
        sub_caption=r"$z^{\mathrm{sem}} \in \mathbb{R}^{H_s \times W_s \times N_c}$",
        title_color="#1E3A8A",
        sub_color="#1E40AF",
    )

    # Arrow 1x1 Conv -> z_sem
    ax.annotate(
        "",
        xy=(w_sem[0] - 0.8, w_sem[1] + 1),
        xytext=(32.5, 36.0),
        arrowprops=dict(arrowstyle="-|>", color="#1D4ED8", lw=1.8, mutation_scale=13),
    )

    # --- BOTTOM PATH: Spectral Branch ---
    # Operator 1: Parallel MaxPool & AvgPool along Channels (x=15, y=3.0)
    draw_operator_pill(
        14.0,
        7.5,
        9.0,
        4.8,
        "#D1FAE5",
        "#059669",
        "Max Pool",
        r"$\max_c(\mathbf{F})$",
        title_color="#065F46",
        sub_color="#047857",
        title_fs=7.8,
        sub_fs=7.0,
    )
    draw_operator_pill(
        14.0,
        1.5,
        9.0,
        4.8,
        "#D1FAE5",
        "#059669",
        "Avg Pool",
        r"$\mathrm{mean}_c(\mathbf{F})$",
        title_color="#065F46",
        sub_color="#047857",
        title_fs=7.8,
        sub_fs=7.0,
    )

    # Fork Arrow to Max & Avg Pool
    ax.annotate(
        "",
        xy=(13.5, 9.8),
        xytext=(e_in[0] + 0.5, e_in[1] - 1.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#047857",
            lw=1.8,
            connectionstyle="arc3,rad=0.12",
            mutation_scale=13,
        ),
    )
    ax.annotate(
        "",
        xy=(13.5, 4.0),
        xytext=(e_in[0] + 0.5, e_in[1] - 2.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#047857",
            lw=1.8,
            connectionstyle="arc3,rad=0.15",
            mutation_scale=13,
        ),
    )

    # Output of Pooling: 2-Channel Tensor (x=27.5, y=3.5)
    e_pool, w_pool = draw_3d_cube(
        27.5,
        3.5,
        2.2,
        9.5,
        1.8,
        *p_pool,
        label_top=r"$2$",
        label_side=r"$H_s{\times}W_s$",
        caption="Pooled Map",
        sub_caption=r"$\mathbf{F}_{\mathrm{pool}} \in \mathbb{R}^{2}$",
        title_color="#064E3B",
        sub_color="#047857",
    )

    # Convergence from Max+Avg to Pooled Tensor
    ax.annotate(
        "",
        xy=(w_pool[0] - 0.8, w_pool[1] + 2),
        xytext=(23.5, 9.8),
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.6, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(w_pool[0] - 0.8, w_pool[1] - 1),
        xytext=(23.5, 4.0),
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.6, mutation_scale=12),
    )

    # Operator 2: Trainable Filter Bank DWConv 3x3 (x=37.5, y=4.5)
    draw_operator_pill(
        37.5,
        4.0,
        12.0,
        8.5,
        "#DCFCE7",
        "#166534",
        r"$\mathbf{DWConv}_{3\times 3}$" + "\n" + r"$\mathbf{Filter \; Bank}$",
        r"$\mathbf{K}_{\mathrm{Lap}}, \mathbf{K}_{\mathrm{Sob\text{-}x}}, \mathbf{K}_{\mathrm{Sob\text{-}y}}$",
        title_color="#14532D",
        sub_color="#166534",
        title_fs=7.8,
        sub_fs=7.0,
    )

    # Arrow Pooled -> Filter Bank
    ax.annotate(
        "",
        xy=(37.0, 8.2),
        xytext=(e_pool[0] + 0.8, e_pool[1] + 1),
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.8, mutation_scale=13),
    )

    # Output of Filter Bank: 6-Channel Tensor (x=53.5, y=3.5)
    e_filt, w_filt = draw_3d_cube(
        53.5,
        3.5,
        2.8,
        9.5,
        2.6,
        *p_filt,
        label_top=r"$6$",
        label_side=r"$H_s{\times}W_s$",
        caption="Gradients",
        sub_caption=r"$\mathbf{F}_{\mathrm{filt}} \in \mathbb{R}^{6}$",
        title_color="#064E3B",
        sub_color="#047857",
    )

    ax.annotate(
        "",
        xy=(w_filt[0] - 0.8, w_filt[1] + 1),
        xytext=(50.0, 8.2),
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.8, mutation_scale=13),
    )

    # Operator 3: 1x1 Conv + BN + SiLU + 1x1 Head (x=63.5, y=4.5)
    draw_operator_pill(
        63.0,
        4.0,
        12.0,
        8.5,
        "#CCFBF1",
        "#0F766E",
        r"$\mathbf{1\times 1 \; Conv}$" + "\n" + r"$\mathbf{+ \; BN \; + \; SiLU}$",
        r"$6 \to 256 \to N_c$",
        title_color="#134E4A",
        sub_color="#0F766E",
        title_fs=7.8,
        sub_fs=7.0,
    )

    ax.annotate(
        "",
        xy=(62.5, 8.2),
        xytext=(e_filt[0] + 0.8, e_filt[1] + 1),
        arrowprops=dict(arrowstyle="-|>", color="#047857", lw=1.8, mutation_scale=13),
    )

    # Output: Spectral Logits Tensor (x=79.0, y=3.5)
    e_shead, w_shead = draw_3d_cube(
        79.0,
        3.5,
        2.8,
        9.5,
        2.0,
        *p_shead,
        label_top=r"$N_c$",
        label_side=r"$H_s{\times}W_s$",
        caption="Spectral Logits",
        sub_caption=r"$z^{\mathrm{spec}} \in \mathbb{R}^{H_s \times W_s \times N_c}$",
        title_color="#064E3B",
        sub_color="#047857",
    )

    ax.annotate(
        "",
        xy=(w_shead[0] - 0.8, w_shead[1] + 1),
        xytext=(75.5, 8.2),
        arrowprops=dict(arrowstyle="-|>", color="#0F766E", lw=1.8, mutation_scale=13),
    )

    # --- Evidence Concat & Fusion ---
    concat_x, concat_y = 89.0, 20.0
    concat_circle = patches.Circle(
        (concat_x, concat_y),
        radius=2.8,
        facecolor="#EDE9FE",
        edgecolor="#7C3AED",
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(concat_circle)
    ax.text(
        concat_x,
        concat_y,
        r"$\mathbf{C}$",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#7C3AED",
        zorder=6,
    )
    ax.text(
        concat_x,
        concat_y - 4.2,
        r"$\mathbf{Concat}$" + "\n" + r"$(2N_c)$",
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#6D28D9",
    )

    # Connection from Semantic to Concat
    ax.annotate(
        "",
        xy=(concat_x - 2.0, concat_y + 2.0),
        xytext=(e_sem[0] + 0.8, e_sem[1] + 1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1D4ED8",
            lw=2.0,
            connectionstyle="arc3,rad=-0.1",
            mutation_scale=14,
        ),
    )

    # Connection from Spectral to Concat
    ax.annotate(
        "",
        xy=(concat_x - 2.0, concat_y - 2.0),
        xytext=(e_shead[0] + 0.8, e_shead[1] + 1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#047857",
            lw=2.0,
            connectionstyle="arc3,rad=0.1",
            mutation_scale=14,
        ),
    )

    # Fusion Conv 1x1 + Sigmoid Output Priority Map S (x=96)
    e_fuse, w_fuse = draw_3d_cube(
        96.0,
        14.0,
        2.5,
        12.0,
        1.5,
        *p_fuse,
        label_top=r"$1$",
        label_side=r"$H_s{\times}W_s$",
        caption="Priority Map",
        sub_caption=r"$\mathbf{S} \in [0, 1]$",
        title_color="#92400E",
        sub_color="#B45309",
    )

    # Arrow Concat -> Fusion
    ax.annotate(
        "",
        xy=(w_fuse[0] - 1.0, w_fuse[1] + 1),
        xytext=(concat_x + 3.0, concat_y),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.0, mutation_scale=14),
    )
    ax.text(
        92.5,
        24.0,
        r"$\mathbf{1\times 1 \; Conv}$" + "\n" + r"$\mathbf{+ \; \sigma(\cdot)}$",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#7C3AED",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


def create_sabl_diagram(save_path):
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 47)
    ax.axis("off")

    c_box = "#F8FAFC"
    b_box = "#475569"
    c_w = "#FFF7ED"
    b_w = "#EA580C"
    c_ciou = "#F1F5F9"
    b_ciou = "#334155"
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
        radius=1.2,
        title_color="#0F172A",
        sub_color="#334155",
        title_fs=9.2,
        sub_fs=7.8,
        lw=1.5,
    ):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.25,rounding_size={radius}",
            facecolor=bg,
            edgecolor=border,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.74,
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
                y + h * 0.36,
                subtitle,
                ha="center",
                va="center",
                fontsize=sub_fs,
                fontweight="bold",
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
        14,
        14,
        c_box,
        b_box,
        "Predicted Box",
        r"$p_{\mathrm{box}} = (c_x, c_y, w, h)$",
        title_color="#1E293B",
        sub_color="#334155",
    )
    draw_node(
        1.5,
        7,
        14,
        14,
        c_box,
        b_box,
        "Ground Truth",
        r"$t_{\mathrm{box}} = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})$",
        title_color="#1E293B",
        sub_color="#334155",
    )

    # Arrows to Scale Gate & Wasserstein
    ax.annotate(
        "",
        xy=(21, 24),
        xytext=(15.8, 14),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.8, mutation_scale=13),
    )

    # Scale computation & Gate
    draw_node(
        21.5,
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
        xy=(48, 36),
        xytext=(42.8, 26),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.8, mutation_scale=13),
    )
    ax.annotate(
        "",
        xy=(48, 12),
        xytext=(42.8, 22),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8, mutation_scale=13),
    )

    # Direct arrows from boxes to terms
    ax.annotate(
        "",
        xy=(48, 40),
        xytext=(15.8, 34),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.8, mutation_scale=13),
    )

    # Top: Wasserstein Penalty
    draw_node(
        48.5,
        28.5,
        28.5,
        15.5,
        c_w,
        b_w,
        "Normalized Wasserstein",
        r"$\mathcal{L}_W = \mu(s) \cdot (1 - e^{-D_W/12})$"
        + "\n"
        + r"(Dominates for Tiny Targets, $s \to 0$)",
        title_color="#9A3412",
        sub_color="#C2410C",
        title_fs=9.2,
        sub_fs=7.8,
        lw=1.6,
    )

    # Bottom: CIoU Center Distance Penalty
    draw_node(
        48.5,
        3.5,
        28.5,
        15.5,
        c_ciou,
        b_ciou,
        "CIoU Center Penalty",
        r"$\mathcal{L}_{\mathrm{ctr}} = (1 - \mu(s)) \cdot \ell_{\mathrm{ctr}}$"
        + "\n"
        + r"(Dominates for Large Targets, $s \gg 32$)",
        title_color="#1E293B",
        sub_color="#334155",
        title_fs=9.2,
        sub_fs=7.8,
        lw=1.6,
    )

    # Base CIoU overlap
    ax.text(
        62.5,
        23.5,
        r"$+ \; (1 - \mathrm{IoU} + \alpha v)$",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#334155",
    )

    # Arrows to Summation
    ax.annotate(
        "",
        xy=(81.5, 25),
        xytext=(76.8, 35),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.8, mutation_scale=14),
    )
    ax.annotate(
        "",
        xy=(81.5, 23),
        xytext=(76.8, 13),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8, mutation_scale=14),
    )

    # Summation node
    sum_circle = patches.Circle(
        (84.5, 24),
        radius=2.8,
        facecolor="#FCE7F3",
        edgecolor="#DB2777",
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(sum_circle)
    ax.text(
        84.5,
        24,
        r"$+$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#DB2777",
        zorder=5,
    )

    ax.annotate(
        "",
        xy=(91.0, 24),
        xytext=(87.5, 24),
        arrowprops=dict(arrowstyle="-|>", color="#DB2777", lw=1.8, mutation_scale=14),
    )

    # Output Box Loss
    draw_node(
        91.5,
        17,
        7.8,
        14,
        c_sum,
        b_sum,
        r"$\mathcal{L}_{\mathrm{box}}$",
        "Total Box\nLoss",
        title_color="#831843",
        sub_color="#9D174D",
        title_fs=10.0,
        sub_fs=8.0,
        lw=1.8,
    )

    plt.tight_layout()
    if save_path.endswith(".jpg"):
        plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


def create_ispphead_diagram(save_path):
    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=300)
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def iso_proj(x, y, z, d_scale=0.45):
        return x + z * d_scale * 0.866, y + z * d_scale * 0.5

    def draw_3d_tensor(
        x0,
        y0,
        w,
        h,
        d,
        c_front,
        c_top,
        c_side,
        edge_color="#1E293B",
        lw=1.3,
        alpha=0.95,
        dashed=False,
        draw_kernel_grid=False,
    ):
        p0 = iso_proj(x0, y0, 0)
        p1 = iso_proj(x0 + w, y0, 0)
        p2 = iso_proj(x0 + w, y0 + h, 0)
        p3 = iso_proj(x0, y0 + h, 0)

        p4 = iso_proj(x0, y0, d)
        p5 = iso_proj(x0 + w, y0, d)
        p6 = iso_proj(x0 + w, y0 + h, d)
        p7 = iso_proj(x0, y0 + h, d)

        # Drop Shadow
        shadow_poly = patches.Polygon(
            [
                (p0[0] - 0.4, p0[1] - 0.6),
                (p1[0] + 0.4, p1[1] - 0.6),
                (p5[0] + 0.4, p5[1] - 0.6),
                (p4[0] - 0.4, p4[1] - 0.6),
            ],
            closed=True,
            facecolor="#94A3B8",
            edgecolor="none",
            alpha=0.25,
            zorder=1,
        )
        ax.add_patch(shadow_poly)

        # Top Face
        top_poly = patches.Polygon(
            [p3, p2, p6, p7],
            closed=True,
            facecolor=c_top,
            edgecolor=edge_color,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(top_poly)

        # Right Face
        side_poly = patches.Polygon(
            [p1, p5, p6, p2],
            closed=True,
            facecolor=c_side,
            edgecolor=edge_color,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(side_poly)

        # Front Face
        front_poly = patches.Polygon(
            [p0, p1, p2, p3],
            closed=True,
            facecolor=c_front,
            edgecolor=edge_color,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            alpha=alpha,
            zorder=4,
        )
        ax.add_patch(front_poly)

        # Optional 3x3 Conv grid on Front Face
        if draw_kernel_grid:
            for i in range(1, 3):
                gx = x0 + i * (w / 3.0)
                gy = y0 + i * (h / 3.0)
                ax.plot(
                    [gx, gx],
                    [y0, y0 + h],
                    color="#FFFFFF",
                    lw=0.9,
                    linestyle=":",
                    zorder=5,
                )
                ax.plot(
                    [x0, x0 + w],
                    [gy, gy],
                    color="#FFFFFF",
                    lw=0.9,
                    linestyle=":",
                    zorder=5,
                )

        return p2, p6, p5

    # --- 1. Input Tensor X ---
    draw_3d_tensor(
        2.5,
        17.0,
        3.6,
        14.0,
        7.0,
        c_front="#0284C7",
        c_top="#38BDF8",
        c_side="#0369A1",
        edge_color="#0C4A6E",
    )
    ax.text(
        4.3,
        24.0,
        r"$C_1$",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#FFFFFF",
        zorder=6,
    )
    ax.text(
        4.3,
        34.5,
        r"$\mathbf{X} \in \mathbb{R}^{C_1 \times H \times W}$",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#0369A1",
        zorder=6,
    )
    ax.text(
        4.3,
        13.5,
        "Input Feature\n(Neck $P_3/P_4/P_5$)",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#334155",
        zorder=6,
    )

    # Arrow Input -> Expand
    ax.annotate(
        "",
        xy=(15.2, 24.0),
        xytext=(8.8, 24.0),
        arrowprops=dict(arrowstyle="-|>", color="#0284C7", lw=1.8, mutation_scale=13),
        zorder=6,
    )
    ax.text(
        12.0,
        26.5,
        r"$1\times 1\text{ Conv}$",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#0284C7",
        zorder=6,
    )

    # --- 2. Inverted Expansion (2 * C1) ---
    draw_3d_tensor(
        16.2,
        17.0,
        3.6,
        14.0,
        14.0,
        c_front="#059669",
        c_top="#34D399",
        c_side="#047857",
        edge_color="#064E3B",
    )
    ax.text(
        18.0,
        24.0,
        r"$2C_1$",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#FFFFFF",
        zorder=6,
    )
    ax.text(
        18.0,
        13.5,
        r"$\mathbf{1\times 1\ Expansion}$" + "\n" + r"($C_{\mathrm{exp}} = 2C_1$)",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#334155",
        zorder=6,
    )

    # Channel Split Arrows (Diverging)
    ax.annotate(
        "",
        xy=(31.8, 34.0),
        xytext=(26.2, 26.5),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.8, mutation_scale=13),
        zorder=6,
    )
    ax.text(
        27.2,
        33.2,
        r"$1/4\ \text{Split}$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#EA580C",
        zorder=6,
    )

    ax.annotate(
        "",
        xy=(31.8, 13.5),
        xytext=(26.2, 21.5),
        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=13),
        zorder=6,
    )
    ax.text(
        27.2,
        14.8,
        r"$3/4\ \text{Split}$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#475569",
        zorder=6,
    )

    # --- 3A. Active Partial Conv Branch (1/4 Channels, Cp = 0.5 C1) ---
    draw_3d_tensor(
        32.6,
        27.5,
        3.6,
        11.5,
        4.0,
        c_front="#EA580C",
        c_top="#FB923C",
        c_side="#C2410C",
        edge_color="#7C2D12",
        draw_kernel_grid=True,
    )
    ax.text(
        34.4,
        33.2,
        r"$C_p$",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#FFFFFF",
        zorder=6,
    )
    ax.text(
        34.4,
        42.2,
        r"$C_p = 0.25 C_{\mathrm{exp}} = 0.5 C_1$",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#C2410C",
        zorder=6,
    )
    ax.text(
        34.4,
        24.6,
        r"$\mathbf{3\times 3\ Partial\ Conv\ (PConv)}$" + "\n(Spatial Context)",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#9A3412",
        zorder=6,
    )

    # --- 3B. Identity Passthrough Branch (3/4 Channels, 1.5 C1) ---
    draw_3d_tensor(
        32.6,
        6.5,
        3.6,
        11.5,
        10.5,
        c_front="#94A3B8",
        c_top="#CBD5E1",
        c_side="#64748B",
        edge_color="#334155",
        alpha=0.75,
        dashed=True,
    )
    ax.text(
        34.4,
        12.2,
        r"$1.5 C_1$",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#FFFFFF",
        zorder=6,
    )
    ax.text(
        34.4,
        20.8,
        r"$0.75 C_{\mathrm{exp}} = 1.5 C_1$",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#475569",
        zorder=6,
    )
    ax.text(
        34.4,
        3.5,
        r"$\mathbf{Identity\ Passthrough}$" + "\n(0 FLOPs / 0 Params)",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#475569",
        zorder=6,
    )

    # Channel Merge Arrows to Concat + Project
    ax.annotate(
        "",
        xy=(50.8, 26.5),
        xytext=(41.5, 33.5),
        arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=1.8, mutation_scale=13),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(50.8, 22.0),
        xytext=(41.5, 14.5),
        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=13),
        zorder=6,
    )

    # --- 4. Concat + Linear Projection (1x1 Conv, 2C1 -> C1) ---
    draw_3d_tensor(
        51.6,
        17.0,
        3.6,
        14.0,
        7.0,
        c_front="#9333EA",
        c_top="#C084FC",
        c_side="#7E22CE",
        edge_color="#581C87",
    )
    ax.text(
        53.4,
        24.0,
        r"$C_1$",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#FFFFFF",
        zorder=6,
    )
    ax.text(
        53.4,
        13.5,
        r"$\mathbf{Concat + 1\times 1\ Proj}$"
        + "\n"
        + r"($\mathbf{F}_{\mathrm{proj}} \in \mathbb{R}^{C_1 \times H \times W}$)",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#334155",
        zorder=6,
    )

    # Arrow to Sum Node
    ax.annotate(
        "",
        xy=(63.0, 24.0),
        xytext=(58.8, 24.0),
        arrowprops=dict(arrowstyle="-|>", color="#9333EA", lw=1.8, mutation_scale=13),
        zorder=6,
    )

    # --- 5. Residual Sum Node (+) ---
    sum_circle = patches.Circle(
        (65.5, 24.0),
        radius=2.4,
        facecolor="#FCE7F3",
        edgecolor="#DB2777",
        linewidth=1.8,
        zorder=7,
    )
    ax.add_patch(sum_circle)
    ax.text(
        65.5,
        24.0,
        r"$+$",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#DB2777",
        zorder=8,
    )

    # Residual Skip Connection Arc from Input X to Sum Node (+) (Clearance above blocks)
    ax.annotate(
        "",
        xy=(65.5, 26.8),
        xytext=(4.3, 31.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2563EB",
            lw=2.2,
            connectionstyle="arc3,rad=-0.48",
            linestyle="-",
            mutation_scale=14,
        ),
        zorder=6,
    )
    # Residual Label Pill Badge
    res_badge = patches.FancyBboxPatch(
        (25.0, 48.0),
        26.0,
        3.5,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        linewidth=1.2,
        zorder=7,
    )
    ax.add_patch(res_badge)
    ax.text(
        38.0,
        49.75,
        r"$\mathbf{+}$ Residual Skip Connection ($\mathbf{X}$)",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#1E40AF",
        zorder=8,
    )

    # Arrows from Sum to Decoupled Heads
    ax.annotate(
        "",
        xy=(73.5, 38.0),
        xytext=(68.2, 25.5),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.6, mutation_scale=12),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(73.5, 24.0),
        xytext=(68.2, 24.0),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.6, mutation_scale=12),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(73.5, 10.0),
        xytext=(68.2, 22.5),
        arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.6, mutation_scale=12),
        zorder=6,
    )

    # --- 6. Decoupled 3D Heads (Cls / Reg / Obj) ---
    # 6A. Classification Branch
    draw_3d_tensor(
        74.5,
        33.5,
        3.2,
        9.0,
        5.0,
        c_front="#7C3AED",
        c_top="#A78BFA",
        c_side="#6D28D9",
        edge_color="#4C1D95",
    )
    ax.text(
        76.1,
        45.8,
        r"$\mathbf{Cls\ Head}$" + "\n" + r"$1\times 1 \to N_c \cdot N_a$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#5B21B6",
        zorder=6,
    )

    # 6B. Bounding Box Regression Branch (SABL)
    draw_3d_tensor(
        74.5,
        19.5,
        3.2,
        9.0,
        5.0,
        c_front="#D97706",
        c_top="#FBBF24",
        c_side="#B45309",
        edge_color="#78350F",
    )
    ax.text(
        76.1,
        31.8,
        r"$\mathbf{Reg\ Head\ (SABL)}$" + "\n" + r"$1\times 1 \to 4 \cdot N_a$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#92400E",
        zorder=6,
    )

    # 6C. Objectness Branch
    draw_3d_tensor(
        74.5,
        5.5,
        3.2,
        9.0,
        4.0,
        c_front="#16A34A",
        c_top="#4ADE80",
        c_side="#15803D",
        edge_color="#14532D",
    )
    ax.text(
        76.1,
        17.8,
        r"$\mathbf{Obj\ Head}$" + "\n" + r"$1\times 1 \to 1 \cdot N_a$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#166534",
        zorder=6,
    )

    # Output Arrows to Final Unified Tensor
    ax.annotate(
        "",
        xy=(90.0, 27.5),
        xytext=(82.5, 38.0),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.6, mutation_scale=12),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(90.0, 24.0),
        xytext=(82.5, 24.0),
        arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.6, mutation_scale=12),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(90.0, 20.5),
        xytext=(82.5, 10.0),
        arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.6, mutation_scale=12),
        zorder=6,
    )

    # --- 7. Final Prediction Tensor Output ---
    draw_3d_tensor(
        91.0,
        14.0,
        4.2,
        20.0,
        9.0,
        c_front="#DB2777",
        c_top="#F472B6",
        c_side="#BE185D",
        edge_color="#831843",
    )
    ax.text(
        93.1,
        39.5,
        r"$\mathbf{Y}_{\mathrm{pred}}$",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#9D174D",
        zorder=6,
    )
    ax.text(
        93.1,
        9.5,
        r"$(N_c + 5)N_a$" + "\n" + r"$\times H \times W$" + "\n(Detections)",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#831843",
        zorder=6,
    )

    # --- 8. Bottom Key Metrics & Efficiency Callout Banner ---
    callout_bg = patches.FancyBboxPatch(
        (2.5, 0.4),
        103.0,
        2.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(callout_bg)
    ax.text(
        54.0,
        1.65,
        "Whole-Model Impact (SeaPerson): -27.6% Params (35.78M -> 25.92M)  |  -20.7% GFLOPs (263.7 -> 209.0)  |  80.7 FPS Real-Time  |  77.14% Very Tiny Recall",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="#1E293B",
        zorder=4,
    )

    plt.tight_layout()
    if save_path.endswith(".jpg"):
        plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Created {save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    create_ispphead_diagram(os.path.join(fig_dir, "ISPPHead.jpg"))
    create_ispphead_diagram(os.path.join(fig_dir, "ISPPHead.pdf"))
    print("ISPPHead figures successfully generated in figures/ directory.")
