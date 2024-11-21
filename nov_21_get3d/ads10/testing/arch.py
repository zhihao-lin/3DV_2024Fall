import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def add_layer(ax, layer_type, x, y, layer_name="", output_shape="", color="skyblue", width=1.5, height=0.8):
    """
    Adds a visual representation of a layer to the plot.
    """
    # Create a box for the layer
    layer_box = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.1",
        edgecolor="black", facecolor=color
    )
    ax.add_patch(layer_box)  # Add the box to the axis

    # Add text inside the box
    ax.text(
        x + width / 2, y + height / 2, layer_type, color="black",
        fontsize=8, ha="center", va="center", weight="bold"
    )

    # Add optional layer name and output shape
    if layer_name:
        ax.text(x + width / 2, y - 0.2, layer_name, color="black", fontsize=7, ha="center")
    if output_shape:
        ax.text(x + width / 2, y + height + 0.2, output_shape, color="gray", fontsize=7, ha="center")

    return x + width + 0.5  # Return the new x position after the layer

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    """
    Adds an arrow to connect layers.
    """
    ax.annotate(
        "", xy=(x_end, y_end), xytext=(x_start, y_start),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )

def plot_generator():
    """
    Creates a diagram of the Generator architecture.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")  # Turn off the axis

    # Starting coordinates
    x, y = 0, 5

    # Encoder: Input to Dense layers
    x = add_layer(ax, "Input", x, y, "Input", "1x64x64", color="lightgray")
    x = add_layer(ax, "Conv2D", x, y, "Conv2D", "64x32x32", color="lightblue")
    x = add_layer(ax, "BatchNorm", x, y, "BatchNorm", "64x32x32", color="lightgreen")
    x = add_layer(ax, "LeakyReLU", x, y, "LeakyReLU", "64x32x32", color="orange")
    x = add_layer(ax, "Flatten", x, y, "Flatten", "65536", color="lightpink")
    x = add_layer(ax, "Dense", x, y, "Dense", "512", color="lightblue")

    # Decoder: Dense to Reshape layers
    x = add_layer(ax, "Dense", x, y, "Dense", "32768", color="violet")
    x = add_layer(ax, "Reshape", x, y, "Reshape", "32x32x32", color="coral")

    # Draw arrows between layers
    x_coords = [1.75 * i for i in range(8)]
    for i in range(len(x_coords) - 1):
        draw_arrow(ax, x_coords[i] + 2, y + 0.4, x_coords[i + 1] + 0.5, y + 0.4)

    # Set axis limits to ensure all elements are visible
    ax.set_xlim(-1, x + 1)
    ax.set_ylim(4, 6.5)

    # Set the title
    plt.title("Generator Architecture", fontsize=16)
    plt.show()

# Plot the Generator architecture
plot_generator()
