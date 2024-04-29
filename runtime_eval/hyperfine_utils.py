import matplotlib.pyplot as plt
import matplotlib

# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("file", help="JSON file with benchmark results")
# parser.add_argument("--title", help="Plot Title")
# parser.add_argument("--sort-by", choices=['median'], help="Sort method")
# parser.add_argument(
#     "--labels", help="Comma-separated list of entries for the plot legend"
# )
# parser.add_argument(
#     "-o", "--output", help="Save image to the given filename."
# )

# args = parser.parse_args()


def save_whisker_plot(hyperfine_results: dict[str, dict], result_filename=None, sort_by='name', title=""):
    matplotlib.use('Agg')

    labels = list(sorted(hyperfine_results.keys()))
    times = [hyperfine_results[name]["times"] for name in labels]

    if sort_by == 'median':
        medians = [hyperfine_results[name]["median"] for name in labels]
        indices = sorted(range(len(labels)), key=lambda k: medians[k])
        labels = [labels[i] for i in indices]
        times = [times[i] for i in indices]
    elif sort_by == 'name':
        indices = sorted(range(len(labels)), key=lambda k: labels[k])
        labels = [labels[i] for i in indices]
        times = [times[i] for i in indices]

    plt.figure(figsize=(10, 6), constrained_layout=True)
    boxplot = plt.boxplot(times, vert=True, patch_artist=True)
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(val / len(times)) for val in range(len(times))]

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)

    if title:
        plt.title(title)
    plt.legend(handles=boxplot["boxes"], labels=labels, loc="best", fontsize="medium")
    plt.ylabel("Time [s]")
    plt.ylim(0, None)
    plt.xticks(list(range(1, len(labels)+1)), labels, rotation=45)
    if result_filename:
        plt.savefig(result_filename)
    else:
        plt.show()