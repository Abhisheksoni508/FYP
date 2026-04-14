import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup figure
fig, ax = plt.subplots(figsize=(18, 10), dpi=400) # Larger canvas, high resolution
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors - Modern sleek palette for academic poster
C_BG = '#FFFFFF'
C_L1 = '#F0F9FF'  
C_L1_BR = '#0284C7'
C_L2 = '#F0FDF4'  
C_L2_BR = '#059669'
C_L3 = '#FEF2F2'  
C_L3_BR = '#DC2626'
C_DATA = '#F8FAFC'
C_DATA_BR = '#475569'
C_OUT = '#FFFBEB'
C_OUT_BR = '#D97706'
C_TEXT = '#0F172A'

fig.patch.set_facecolor(C_BG)

def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", fontsize=14):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.08, y-0.08), w, h, boxstyle="round,pad=0.2",
                                    facecolor='black', alpha=0.08, edgecolor='none')
    ax.add_patch(shadow)
    # Main Box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=2.5)
    ax.add_patch(box)
    
    # Text
    title_y = y + h/2 + (0.4 if subtitle else 0)
    ax.text(x + w/2, title_y, title, 
            ha='center', va='center', fontsize=fontsize+2, fontweight='bold', color=C_TEXT, family='sans-serif')
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.4, subtitle, 
                ha='center', va='center', fontsize=fontsize, color='#334155', family='sans-serif', linespacing=1.5)

def draw_arrow(x1, y1, x2, y2, text="", rad=0.0, color=C_TEXT, ls='-', text_y_offset=0):
    # Arrow
    ax.annotate("",
                xy=(x2, y2), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="->,head_length=0.9,head_width=0.45", 
                                color=color, lw=3, ls=ls,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        # Midpoint for text
        if rad == 0:
            mx, my = (x1+x2)/2, (y1+y2)/2
        else:
            mx, my = (x1+x2)/2, (y1+y2)/2
            my += (rad * abs(x2-x1) * 0.4) 
        
        my += text_y_offset
            
        # Bounding box for text for readability
        bbox = dict(boxstyle="round,pad=0.5", fc=C_BG, ec=color, alpha=0.95, lw=1)
        if ls == '--' or color != C_TEXT:
             bbox['facecolor'] = '#FFFFFF'
        else:
             bbox['edgecolor'] = 'none'

        ax.text(mx, my, text, ha='center', va='center', fontsize=12, fontweight='bold', 
                color=color, bbox=bbox, family='sans-serif')

# --- Drawing the Architecture (No overlaps) ---

# 1. Data Source
draw_box(0.5, 4.0, 2.5, 2.0, C_DATA, C_DATA_BR, "NASA C-MAPSS\nSensors", "30-Cycle Sliding Window\n+ Noise Injection", fontsize=12)

# 2. Layer 1 (Perception)
draw_box(4.5, 3.5, 3.5, 3.0, C_L1, C_L1_BR, "LAYER 1: PERCEPTION", "LSTM Deep Ensemble (\u00d75)\nBootstrap Sampled\n\n\u2192 Extracts Epistemic\nUncertainty (\u03c3)")

# 3. Layer 2 (Cognition)
draw_box(10.0, 6.0, 4.0, 2.5, C_L2, C_L2_BR, "LAYER 2: COGNITION", "Uncertainty-Aware DQN\n\n\u2192 Reward Penalty on High \u03c3\nNear End-of-Life")

# 4. Layer 3 (Reflex)
draw_box(10.0, 1.5, 4.0, 2.5, C_L3, C_L3_BR, "LAYER 3: REFLEX", "\u03c3-Gated Safety Supervisor\n\n\u2192 Hard Confidence Bounds\non RUL < 15 cycles")

# 5. Output / Action
draw_box(15.5, 4.0, 1.8, 2.0, C_OUT, C_OUT_BR, "FINAL\nACTION", "WAIT or\nMAINTAIN", fontsize=13)

# --- Connections ---

# Data -> Layer 1
draw_arrow(3.2, 5.0, 4.3, 5.0, "[24 Features]")

# Layer 1 -> Layer 2
draw_arrow(8.2, 5.5, 9.8, 7.25, "State [ \u03bc_RUL, \u03c3_now, \u03c3_roll, trend ]", rad=-0.15, text_y_offset=0.2)

# Layer 1 -> Layer 3
draw_arrow(8.2, 4.5, 9.8, 2.75, "Predictions [ \u03bc_RUL, \u03c3_roll ]", rad=0.15, text_y_offset=-0.3)

# Layer 2 -> Output Action
draw_arrow(14.2, 7.25, 15.3, 5.5, "DQN Policy:\nWAIT / MAINTAIN", rad=-0.2, text_y_offset=0.4, color=C_L2_BR)

# Layer 3 -> Output (Override)
draw_arrow(14.2, 2.75, 15.3, 4.5, "Emergency Override\n(Forces MAINTAIN)", rad=0.2, text_y_offset=-0.4, color=C_L3_BR, ls='--')

# Overarching title
plt.title("A Robust Predictive Maintenance Framework for IoT Systems via Uncertainty-Aware Reinforcement Learning\n3-Layer Hybrid Architecture Dashboard", 
          fontsize=18, fontweight='bold', color=C_TEXT, pad=20, family='sans-serif')

plt.tight_layout()
plt.savefig('figures/architecture_poster_diagram.png', dpi=400, bbox_inches='tight', facecolor=C_BG)
print("Diagram generated successfully at figures/architecture_poster_diagram.png")
