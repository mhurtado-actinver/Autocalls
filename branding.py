# Actinver brand palette - Extracted from 'Template' provided by Hector
BLUE_1 = '#041020'
BLUE_2 = '#041E42'
BLUE_3 = '#003057'
BLUE_4 = '#003B71'
GOLD_1 = '#B3A369'
GOLD_2 = '#C6B784'
GOLD_3 = '#E1DAC3'
GOLD_4 = '#F7F6F0'

# Define standard color cycle
COLOR_CYCLE = [
    BLUE_1,
    BLUE_2,
    BLUE_3,
    BLUE_4,
    GOLD_1,
    GOLD_2,
    GOLD_3,
    GOLD_4
]


ACTINVER_BLUES = [BLUE_1, BLUE_2, BLUE_3, BLUE_4]

ACTINVER_GOLDS = [GOLD_1, GOLD_2, GOLD_3, GOLD_4]



def show_palette(palette, title="Color Palette"):
    import matplotlib.pyplot as plt

    n = len(palette)
    fig, ax = plt.subplots(figsize=(n, 1))
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title(title, pad=10)
    plt.show()
