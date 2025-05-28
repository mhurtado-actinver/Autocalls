import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
import numpy as np
import math
import os
import locale
import pandas as pd




FONTSIZE = 7






##################
# Plotting Paths #
##################

def configure_plot_style(fontname):
    """
    Set the general seaborn and matplotlib style for plots.

    Parameters
    ----------
    fontname : str
        Font name to use throughout the plot.
    """
    sns.set(style="white")
    plt.rcParams["font.family"] = fontname

def plot_monte_carlo_paths(ax, paths, palette, drawn_paths):
    """
    Plot Monte Carlo simulation paths on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    paths : np.ndarray
        2D array of shape (timesteps x simulations) of asset price paths.
    palette : list
        List of colors to use for each path.
    drawn_paths : int
        Number of paths to draw.
    """
    n_paths = min(drawn_paths, paths.shape[1])
    for i in range(n_paths):
        color = palette[i] if i < len(palette) else palette[-1]
        ax.plot(paths[:, i], linewidth=1.0, color=color, alpha=0.7)

def format_x_axis_with_dates(ax, dates, fontname, num_ticks=12):
    """
    Format the x-axis using Spanish locale and provided date labels.
    Shows evenly spaced ticks and avoids missing dates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to modify.
    dates : list or pd.Series
        Sequence of date values matching the timesteps.
    fontname : str
        Font to apply to the labels.
    num_ticks : int
        Approximate number of ticks to show on the x-axis.
    """
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES.utf8')  # Unix/macOS
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')  # Windows
        except:
            print("⚠️ No se pudo establecer el idioma español para las fechas.")

    total_points = len(dates)
    tick_indices = np.linspace(0, total_points - 1, num=num_ticks, dtype=int)
    ax.set_xticks(tick_indices)

    labels = [
        f"{dt.strftime('%d')}-{dt.strftime('%b').replace('.', '').capitalize()}"
        for dt in pd.to_datetime(dates.iloc[tick_indices])
    ]

    ax.set_xticklabels(labels, fontname=fontname, rotation=90, ha='center')
    ax.spines['bottom'].set_visible(True)  # 🔧 Ensure axis line is visible
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', which='major', direction='out', length=5,  
                   width=0.5,
            bottom=True,
            top=False,
            labelsize=8)
    ax.xaxis.set_tick_params(labelsize=FONTSIZE)

def format_y_axis(ax, paths, drawn_paths, fontname='Century Gothic', fontsize=8):
    """
    Format the y-axis using percentiles and rounded steps, based only on drawn paths.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to modify.
    paths : np.ndarray
        2D array of price paths.
    drawn_paths : int
        Number of paths actually being plotted.
    """
    n_paths = min(drawn_paths, paths.shape[1])
    visible_paths = paths[:, :n_paths]

    ymin, ymax = np.percentile(visible_paths, [0.1, 99.9])
    yrange = ymax - ymin
    rough_step = yrange / 6
    magnitude = 10 ** math.floor(math.log10(rough_step))
    step = math.ceil(rough_step / magnitude) * magnitude

    ymin_rounded = math.floor(ymin / step) * step
    ymax_rounded = math.ceil(ymax / step) * step
    ax.set_ylim(ymin_rounded, ymax_rounded)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))

     # ✅ Apply font style to y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)


def add_reference_line(ax, paths, line_color, fontname):
    """
    Draw a horizontal reference line (normally for initial value) with annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to draw on.
    paths : np.ndarray
        2D array of asset paths.
    line_color : str
        Color for the reference line.
    fontname : str
        Font to use in the annotation.
    """
    start_price = paths[0, 0]
    ax.axhline(start_price, color=line_color, linestyle="--", linewidth=2)
    ax.text(
        x=paths.shape[0] + 15,
        y=start_price,
        s=f"Valor Inicial: {start_price:,.0f}",
        va='center',
        ha='left',
        fontname=fontname,
        fontsize=FONTSIZE,
        color=line_color,
        bbox=dict(boxstyle="round,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7)
    )

def finalize_and_save_plot(fig, ax, filename_prefix):
    """
    Finalizes the layout, applies cosmetic changes, and saves the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axis object.
    filename_prefix : str
        Base name for saving the image files.
    """
    for spine in ['top', 'left']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'right']:
        ax.spines[spine].set_linewidth(0.5)

    ax.grid(False)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.95)

    # Ensure 'images' directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)

    safe_name = filename_prefix.lower().replace(" ", "_")
    png_path = os.path.join(output_dir, f"{safe_name}.png")
    svg_path = os.path.join(output_dir, f"{safe_name}.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

def plot_paths(
    paths,
    drawn_paths=100,
    palette=None,
    line_color=None,
    ylabel="Precio",
    fontname="Century Gothic",
    filename_prefix="monte_carlo_paths",
    dates=None,
    show_axis_titles=True
):
    """
    Main function to plot Monte Carlo paths using a modular, customizable setup.

    Parameters
    ----------
    paths : np.ndarray
        2D array (timesteps x simulations) of asset price paths.
    drawn_paths : int, optional
        Number of paths to draw (default is 100).
    palette : list, optional
        Color palette to use for path lines. If None, uses 'Blues' palette.
    line_color : str, optional
        Color used for the horizontal reference line at initial price.
    filename_prefix : str, optional
        Prefix for the output file names (default: 'monte_carlo_paths').
    ylabel : str, optional
        Label for the y-axis.
    fontname : str, optional
        Font used for all text elements.
    dates : list or pd.Series, optional
        List of datetime objects or strings corresponding to each timestep.
    use_dates : bool, optional
        If True, display x-axis using dates instead of day indices.
    """

    # Apply global plot style
    configure_plot_style(fontname)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define color palette
    if palette is None:
        palette = sns.color_palette("Blues", n_colors=min(drawn_paths, paths.shape[1]))

    # Plot the paths
    plot_monte_carlo_paths(ax, paths, palette, drawn_paths)

    # Always format the x-axis correctly based on data type
    if dates is not None:
        if len(dates) != paths.shape[0]:
            raise ValueError("Length of 'dates' must match number of timesteps in 'paths'")
        format_x_axis_with_dates(ax, dates, fontname)
    
    # Only show axis titles if enabled
    if show_axis_titles:
        ax.set_xlabel("Fechas" if dates is not None else "Días", fontname=fontname)
        ax.set_ylabel(ylabel, fontname=fontname)



    # Format Y-axis with rounded scale
    format_y_axis(ax, paths, drawn_paths, fontname=fontname, fontsize=FONTSIZE)
    ax.yaxis.tick_right()                      # Puts ticks on the right
    ax.yaxis.set_label_position("right")       # Puts label ("Precio") on the right

    # Add horizontal reference line for start price
    add_reference_line(ax, paths, line_color, fontname)

    # Ensure x-axis tick marks are visible and well styled
    ax.tick_params(axis='x', which='major', direction='out', length=5, width=0.5)
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')  # explicitly draw ticks at bottom
    ax.xaxis.set_tick_params(labelsize=FONTSIZE)


    # Save the figure and finalize layout
    finalize_and_save_plot(fig, ax, filename_prefix)

    # Show plot
    plt.show()

#########################
# End of Plotting Paths #
#########################






def compute_global_ylim(*paths_list, low=0.1, high=99.9):
    all_lows = []
    all_highs = []
    for paths in paths_list:
        p_low, p_high = np.percentile(paths, [low, high])
        all_lows.append(p_low)
        all_highs.append(p_high)
    return min(all_lows), max(all_highs)

def snap_ylim_to_last_tick(ax):
    """
    Adjusts y-axis so it ends at the last major tick.
    """
    ticks = ax.get_yticks()
    ax.set_ylim(top=ticks[-1])





def plot_fan_chart(
    paths,
    percentiles=[5, 25, 50, 75, 95],
    colors=["#d0d1e6", "#a6bddb", "#3690c0"],  # outer to inner
    line_color=None,
    fontname="Century Gothic",
    ylabel="Precio",
    filename_prefix="fan_chart",
    dates=None,
    show_axis_titles=True,
    fixed_ylim=None
):
    """
    Plot a fan chart showing percentile bands over time for a Monte Carlo simulation.

    Parameters
    ----------
    paths : np.ndarray
        Array of shape (timesteps x simulations).
    percentiles : list
        Percentiles to compute and plot (must include 50 for median).
    colors : list
        Colors for shading percentile bands (outer to inner).
        Must be len = (len(percentiles) - 1) // 2.
    fontname : str
        Font to use.
    ylabel : str
        Label for y-axis.
    filename_prefix : str
        Output filename prefix.
    dates : list or pd.Series, optional
        Dates to use on x-axis.
    show_axis_titles : bool
        Whether to show axis labels.
    """
    # Style and figure
    configure_plot_style(fontname)
    fig, ax = plt.subplots(figsize=(4.527559, 2.362205)) # Figsize original is (10,5)

    # Compute percentiles
    percentiles_sorted = sorted(percentiles)
    pct = np.percentile(paths, percentiles_sorted, axis=1)

    # Plot shaded bands from outermost to innermost
    n_bands = (len(percentiles_sorted) - 1) // 2
    for i in range(n_bands):
        low = pct[i]
        high = pct[-(i + 1)]
        ax.fill_between(
            np.arange(paths.shape[0]),
            low,
            high,
            color=colors[i],
            alpha=0.8
        )

    # Plot median line
    if 50 in percentiles_sorted:
        median_idx = percentiles_sorted.index(50)
        ax.plot(pct[median_idx], color="black", linewidth=2, label="Mediana")

    # X-axis formatting
    if dates is not None:
        if len(dates) != paths.shape[0]:
            raise ValueError("Length of 'dates' must match number of timesteps in 'paths'")
        format_x_axis_with_dates(ax, pd.Series(dates), fontname)
        if show_axis_titles:
            ax.set_xlabel("Fechas", fontname=fontname)
    elif show_axis_titles:
        ax.set_xlabel("Días", fontname=fontname)

    # Y-axis
    if show_axis_titles:
        ax.set_ylabel(ylabel, fontname=fontname, fontsize=FONTSIZE)

    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
        ax.tick_params(axis='y', labelsize=FONTSIZE,  width=0.5)
    else:
        format_y_axis(ax, paths, drawn_paths=paths.shape[1], fontname=fontname, fontsize=FONTSIZE) # full path coverage


    

    # Snap ymax to last visible tick
    snap_ylim_to_last_tick(ax)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Reference line at start price
    add_reference_line(ax, paths, line_color=line_color, fontname=fontname)

    # Save and finalize
    finalize_and_save_plot(fig, ax, filename_prefix)
    ax.legend(frameon=False)
    plt.show()




######################
# Presentation Plots #
######################

def apply_return_plot_style(ax, 
                            show_title=True, 
                            title_text=None, 
                            fontname="Century Gothic", 
                            show_axis_titles=True,
                            show_legend=True, 
                            snap_y_ticks=True):
    plt.style.use('default')
    ax.set_facecolor("white")
    ax.grid(False)

    # Hide top and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Keep right and bottom axes only
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Legend styling
    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.,
            fontsize=8
            )
    else:
        ax.get_legend().remove() if ax.get_legend() else None


    # Font styling
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(8)

    # Title
    if show_title and title_text:
        ax.set_title(title_text, fontname=fontname, fontsize=10)
    else:
        ax.set_title("")

    # Axis labels visibility
    if not show_axis_titles:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Snap x-axis to start at first visible tick
    x_ticks = ax.get_xticks()
    if len(x_ticks) > 0:
        ax.set_xlim(left=x_ticks[0])

    
    if snap_y_ticks:
        y_max = ax.get_ylim()[1]
        y_step = 0.1
        y_max_rounded = min(1.0, ceil(y_max / y_step) * y_step)
        ax.set_ylim(top=y_max_rounded)
        ax.yaxis.set_major_locator(MultipleLocator(y_step))



def plot_return_histogram_comparison(
    real_prices,
    simulated_paths,
    asset_name="Asset",
    use_log_returns=True,
    seed=None,
    bins=40,
    fontname="Century Gothic",
    show_title=True,
    show_axis_titles=True,
    show_legend=True,
    filename="histogram_comparison",
    fixed_xlim=None,
    snap_y_ticks=True
):
    """
    Plot and save a histogram comparison of real vs. simulated returns.

    Parameters
    ----------
    real_prices : array-like (1D)
        Historical real price series.
    simulated_paths : ndarray (2D)
        Simulated price paths (timesteps x simulations).
    asset_name : str
        Asset name for title/label.
    use_log_returns : bool
        Whether to use log or simple returns.
    seed : int, optional
        Random seed for reproducibility.
    bins : int
        Number of histogram bins.
    fontname : str
        Font to use in the plot.
    show_title : bool
        Whether to show the title.
    show_axis_titles : bool
        Whether to show x/y axis labels.
    show_legend : bool
        Whether to display the legend.
    filename : str
        Name of the output file (no extension). Saved to ./images/
    """

    real_prices = np.asarray(real_prices)
    simulated_paths = np.asarray(simulated_paths)
    simulated_paths = simulated_paths[1:]

    # Compute returns
    if use_log_returns:
        returns_real = np.log(real_prices[1:] / real_prices[:-1])
        log_returns_sim = np.log(simulated_paths[1:] / simulated_paths[:-1])
    else:
        returns_real = (real_prices[1:] / real_prices[:-1]) - 1
        log_returns_sim = (simulated_paths[1:] / simulated_paths[:-1]) - 1

    log_returns_sim_flat = log_returns_sim.flatten()

    if seed is not None:
        np.random.seed(seed)

    sim_sample = np.random.choice(log_returns_sim_flat, size=len(returns_real), replace=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(returns_real, bins=bins, kde=False, color="#888888",
                 label="Real Returns", stat='density', element='step', ax=ax)
    sns.histplot(sim_sample, bins=bins, kde=False, color="#1f77b4",
                 label="Simulated Sample", stat='density', element='step', ax=ax)

    if show_axis_titles:
        ax.set_xlabel("Log Returns" if use_log_returns else "Simple Returns", fontname=fontname)
        ax.set_ylabel("Density", fontname=fontname)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)

    apply_return_plot_style(
        ax,
        show_title=show_title,
        title_text=f"Histogram Comparison: {asset_name}",
        fontname=fontname,
        show_axis_titles=show_axis_titles,
        show_legend=show_legend,
        snap_y_ticks=False
    )

    # Force y-axis to end on a visible tick
    ymax = ax.get_ylim()[1]
    ticks = ax.get_yticks()
    tick_spacing = ticks[1] - ticks[0] if len(ticks) > 1 else 0.1
    ymax_rounded = np.ceil(ymax / tick_spacing) * tick_spacing
    ax.set_ylim(top=ymax_rounded)


    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    fig.savefig(os.path.join("images", f"{filename}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("images", f"{filename}.svg"), format="svg", bbox_inches="tight")

    plt.show()
    plt.close(fig)





def plot_return_kde_comparison(
    real_prices,
    simulated_paths,
    asset_name="Asset",
    use_log_returns=True,
    seed=None,
    fontname="Century Gothic",
    bandwidth_adjust=1.0,
    show_title=True,
    show_axis_titles=True,
    show_legend=True,
    filename="kde_comparison",
    fixed_xlim=None
):
    """
    Plot and save KDE (smoothed density estimate) comparison of real vs. simulated returns.

    Parameters
    ----------
    real_prices : array-like (1D)
        Historical real price series.
    simulated_paths : array-like (2D)
        Simulated price paths of shape (timesteps x simulations).
    asset_name : str
        Name of the asset for display title.
    use_log_returns : bool
        Whether to use log returns (True) or simple returns (False).
    seed : int, optional
        Random seed for reproducibility.
    fontname : str
        Font used in plot.
    bandwidth_adjust : float
        KDE bandwidth scaling factor.
    show_title : bool
        Whether to display title.
    show_axis_titles : bool
        Whether to show axis labels.
    show_legend : bool
        Whether to show the legend.
    filename : str
        Output file name (saved in ./images).
    fixed_xlim : tuple or None
        If set, forces the same x-axis limits across models.
    """

    real_prices = np.asarray(real_prices)
    simulated_paths = np.asarray(simulated_paths)
    simulated_paths = simulated_paths[1:]

    if use_log_returns:
        returns_real = np.log(real_prices[1:] / real_prices[:-1])
        log_returns_sim = np.log(simulated_paths[1:] / simulated_paths[:-1])
    else:
        returns_real = (real_prices[1:] / real_prices[:-1]) - 1
        log_returns_sim = (simulated_paths[1:] / simulated_paths[:-1]) - 1

    log_returns_sim_flat = log_returns_sim.flatten()
    if seed is not None:
        np.random.seed(seed)
    sim_sample = np.random.choice(log_returns_sim_flat, size=len(returns_real), replace=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(returns_real, label="Real Returns", color="#555", linewidth=2, bw_adjust=bandwidth_adjust, ax=ax)
    sns.kdeplot(sim_sample, label="Simulated Sample", color="#1f77b4", linewidth=2, linestyle="--", bw_adjust=bandwidth_adjust, ax=ax)

    if show_axis_titles:
        ax.set_xlabel("Log Returns" if use_log_returns else "Simple Returns", fontname=fontname)
        ax.set_ylabel("Density", fontname=fontname)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)

    apply_return_plot_style(
        ax,
        show_title=show_title,
        title_text=f"KDE Comparison: {asset_name}",
        fontname=fontname,
        show_axis_titles=show_axis_titles,
        show_legend=show_legend,
        snap_y_ticks=False
    )

    # Snap y-axis to a clean top tick
    ymax = ax.get_ylim()[1]
    yticks = ax.get_yticks()
    if len(yticks) >= 2:
        spacing = yticks[1] - yticks[0]
        ax.set_ylim(top=np.ceil(ymax / spacing) * spacing)

    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    fig.savefig(os.path.join("images", f"{filename}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("images", f"{filename}.svg"), format="svg", bbox_inches="tight")

    plt.show()
    plt.close(fig)



def plot_return_qq(
    real_prices,
    simulated_paths,
    asset_name="Asset",
    use_log_returns=True,
    seed=None,
    fontname="Century Gothic",
    show_title=True,
    show_axis_titles=True,
    show_legend=True,
    filename="qq_plot"
):
    """
    Generate and save a QQ plot comparing real and simulated returns.

    Parameters
    ----------
    real_prices : array-like (1D)
        Real price series.
    simulated_paths : ndarray (2D)
        Simulated paths of shape (timesteps x simulations).
    asset_name : str
        Label for plot title.
    use_log_returns : bool
        Whether to compute log returns (True) or simple returns (False).
    seed : int, optional
        Random seed for reproducibility.
    fontname : str
        Font to use for all labels.
    show_title : bool
        Whether to display the title.
    show_axis_titles : bool
        Whether to display x/y axis titles.
    show_legend : bool
        Whether to display the legend.
    filename : str
        Name of output file (saved as PNG and SVG in `images/`).
    """

    real_prices = np.asarray(real_prices)
    simulated_paths = np.asarray(simulated_paths)
    simulated_paths = simulated_paths[1:]

    if use_log_returns:
        returns_real = np.log(real_prices[1:] / real_prices[:-1])
        returns_sim = np.log(simulated_paths[1:] / simulated_paths[:-1])
    else:
        returns_real = (real_prices[1:] / real_prices[:-1]) - 1
        returns_sim = (simulated_paths[1:] / simulated_paths[:-1]) - 1

    returns_sim_flat = returns_sim.flatten()
    if seed is not None:
        np.random.seed(seed)
    sim_sample = np.random.choice(returns_sim_flat, size=len(returns_real), replace=False)

    real_sorted = np.sort(returns_real)
    sim_sorted = np.sort(sim_sample)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(real_sorted, sim_sorted, "o", alpha=0.6, label="Quantile Comparison")

    lims = [min(real_sorted.min(), sim_sorted.min()), max(real_sorted.max(), sim_sorted.max())]
    ax.plot(lims, lims, "--", color="gray", label="Perfect Match Line")

    if show_axis_titles:
        ax.set_xlabel("Real Returns Quantiles", fontname=fontname)
        ax.set_ylabel("Simulated Returns Quantiles", fontname=fontname)

    apply_return_plot_style(
        ax,
        show_title=show_title,
        title_text=f"QQ Plot: {asset_name}",
        fontname=fontname,
        show_axis_titles=show_axis_titles,
        show_legend=show_legend
    )

    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    fig.savefig(os.path.join("images", f"{filename}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join("images", f"{filename}.svg"), format="svg", bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_return_ecdf_comparison(
    real_prices,
    simulated_paths,
    asset_name="Asset",
    use_log_returns=True,
    seed=None,
    fontname="Century Gothic", 
    show_title=True,
    show_axis_titles=True,
    show_legend=True,
    filename="ecdf_comparison",
    fixed_xlim=None
):
    """
    Plot ECDF comparison between real and simulated return distributions and save the figure.

    Parameters
    ----------
    real_prices : array-like (1D)
        Observed historical prices.
    simulated_paths : ndarray (2D)
        Simulated price paths of shape (timesteps x simulations).
    asset_name : str
        Label for title and legend.
    use_log_returns : bool
        Use log returns if True, simple returns if False.
    seed : int, optional
        Random seed for reproducibility.
    fontname : str
        Font for plot labels.
    show_title : bool
        Whether to show the plot title.
    show_axis_titles : bool
        Whether to show axis titles.
    show_legend : bool
        Whether to display the legend.
    filename : str
        Name of the output file (without extension); saved to images/ as PNG and SVG.
    """

    real_prices = np.asarray(real_prices)
    simulated_paths = np.asarray(simulated_paths)
    simulated_paths = simulated_paths[1:]

    # Compute returns
    if use_log_returns:
        returns_real = np.log(real_prices[1:] / real_prices[:-1])
        returns_sim = np.log(simulated_paths[1:] / simulated_paths[:-1])
    else:
        returns_real = (real_prices[1:] / real_prices[:-1]) - 1
        returns_sim = (simulated_paths[1:] / simulated_paths[:-1]) - 1

    returns_sim_flat = returns_sim.flatten()

    if seed is not None:
        np.random.seed(seed)

    sim_sample = np.random.choice(returns_sim_flat, size=len(returns_real), replace=False)

    real_sorted = np.sort(returns_real)
    sim_sorted = np.sort(sim_sample)
    ecdf_real = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    ecdf_sim = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(real_sorted, ecdf_real, label="Real Returns", lw=2, color="#444")
    ax.plot(sim_sorted, ecdf_sim, label="Simulated Returns", lw=2, linestyle="--", color="#1f77b4")

    if show_axis_titles:
        ax.set_xlabel("Log Returns" if use_log_returns else "Simple Returns", fontname=fontname)
        ax.set_ylabel("Cumulative Probability", fontname=fontname)

    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)

    apply_return_plot_style(
        ax,
        show_title=show_title,
        title_text=f"ECDF Comparison: {asset_name}",
        fontname=fontname,
        show_axis_titles=show_axis_titles,
        show_legend=show_legend
    )


    # Ensure 'images/' directory exists
    os.makedirs("images", exist_ok=True)

    # Save both PNG and SVG
    png_path = os.path.join("images", f"{filename}.png")
    svg_path = os.path.join("images", f"{filename}.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')


    plt.show()       # Always display the plot
    plt.close(fig)   # Clean up memory after showing







#################
# Miscellaneous #
#################

def list_available_fonts(fontname):
    # List all available fonts with a string similar to the argument 'fontname'
    available_fonts = [f.name for f in fm.fontManager.ttflist if fontname in f.name]
    print(available_fonts)