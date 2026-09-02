import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("frames", exist_ok=True)

grid_size = 4
block_size = 1
iz_blocks = [(1, 2), (2, 1), (2, 3)]  # IZ units in grid coordinates

for i in range(10):
    fig, ax = plt.subplots(figsize=(6, 10.67))  # 1080x1920 aspect ratio
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.axis('off')
    
    # Draw neighborhood blocks
    for x in range(grid_size):
        for y in range(grid_size):
            color = "#b0b0b0"  # default building
            if i >= 2 and (x, y) in iz_blocks:
                color = "#2ecc71"  # highlight IZ units after frame 2
            rect = patches.Rectangle((x, y), block_size, block_size,
                                     linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
    
    # Add resident icons
    if i >= 3:
        for idx, (x, y) in enumerate(iz_blocks):
            if i >= 3 + idx:  # stagger residents moving in
                circle = patches.Circle((x+0.5, y+0.5), 0.1, color='blue')
                ax.add_patch(circle)
    
    # Optional: text overlay for frames 7+
    if i >= 6:
        ax.text(2, 4.2, "Inclusionary Zoning", ha='center', fontsize=20, color='black')
    if i >= 7:
        ax.text(2, 3.8, "Keeps teachers, seniors, and workers in the neighborhood",
                ha='center', fontsize=14, color='black', wrap=True)
    
    # Save frame
    plt.savefig(f"frames/frame_{i+1:02d}.png", dpi=150, bbox_inches='tight')
    plt.close()
