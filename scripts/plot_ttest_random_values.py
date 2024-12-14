import numpy as np


def ttest_figure(ttest_dicts):
    samples = list(ttest_dicts.values())
    if len(samples) != 6:
        return False

    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.palettes.color_palette("muted")

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), tight_layout=True, sharey=True)
    axs = axs.flatten()

    for i in range(6):
        ttest_dict = samples[i]
        ax = axs[i]

        frame_means = np.array(
            [np.mean(frame_arr) for frame_arr in ttest_dict["frame_labels"]]
        )

        ax.plot(
            frame_means, ".-", label="Average probability of frame", color=colors[0]
        )
        if "early_stop" in ttest_dict:
            ax.vlines(
                ttest_dict["early_stop"],
                ymin=0,
                ymax=1,
                label="Early stop iteration",
                color=colors[1],
            )
            ax.annotate(
                f"Predicted: {ttest_dict['predicted_label']:.2f}",
                arrowprops=dict(arrowstyle="->"),
                xytext=(ttest_dict["early_stop"] + 0.22 * len(frame_means), 0.6),
                xy=(ttest_dict["early_stop"], 0.5),
                ha="center",
                va="center",
            )

        ax.hlines(
            y=ttest_dict["actual_label"],
            xmin=0,
            xmax=len(frame_means),
            label="Actual label",
            color=colors[2],
        )
        ax.set_xlabel("1 out of 10 frames")
        if i % 3 == 0:
            ax.set_ylabel("Probability")

        ax.set_ylim([-0.05, 1.05])

    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        labelspacing=0.0,
        fancybox=True,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.savefig("ttest_viz.png", bbox_inches="tight")
    fig.savefig("ttest_viz.eps", bbox_inches="tight")
    plt.show()

    return True


def main():
    plot_dict = dict()
    # test with random
    for i in range(6):
        plot_dict[i] = dict()
        plot_dict[i]["actual_label"] = int(i)
        plot_dict[i]["predicted_label"] = np.random.rand()
        plot_dict[i]["early_stop"] = np.random.randint(0, 10)
        plot_dict[i]["frame_labels"] = [np.random.rand(10) for _ in range(10)]


if __name__ == "__main__":
    main()
