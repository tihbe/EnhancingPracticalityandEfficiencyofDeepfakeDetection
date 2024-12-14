import os
import matplotlib.pyplot as plt
from dfdetect.utils import video_to_frames, crop_frames
from dfdetect.preprocessing.face_detection import waterfall


celeb_df_syn = "/mnt/d/datasets/Celeb-DF-v2/Celeb-synthesis"
dfdc_sample_train = "/mnt/d/datasets/deepfake-detection-challenge/train_sample_videos"


fig, axs = plt.subplots(2, 4, tight_layout=True)


line_1 = [
    (os.path.join(celeb_df_syn, "id0_id1_0007.mp4"), 5),  # path, frame nb
    (os.path.join(celeb_df_syn, "id48_id42_0009.mp4"), 12),
    (os.path.join(celeb_df_syn, "id29_id34_0009.mp4"), 9),
    (os.path.join(celeb_df_syn, "id58_id49_0007.mp4"), 6),
]
line_2 = [
    (os.path.join(dfdc_sample_train, "bctvsmddgq.mp4"), 5),  # path, frame nb
    (os.path.join(dfdc_sample_train, "etmcruaihe.mp4"), 5),
    (os.path.join(dfdc_sample_train, "cfyduhpbps.mp4"), 5),
    (os.path.join(dfdc_sample_train, "dkrvorliqc.mp4"), 5),
]


lines = [line_1, line_2]

for line_nb in range(2):
    line = lines[line_nb]
    for i, (path, frame_nb) in enumerate(line):
        frames = list(video_to_frames(path, frame_nb + 1)[0])
        axs[line_nb, i].imshow(
            list(crop_frames([frames[frame_nb]], waterfall(frames[frame_nb])))[0]
        )
        axs[line_nb, i].get_xaxis().set_visible(False)
        axs[line_nb, i].get_yaxis().set_ticks([])
        if i == 0:
            axs[line_nb, i].set_ylabel("Celeb-DF V2" if line_nb == 0 else "DFDC")


fig.savefig("facial_crops_datasets.png", bbox_inches="tight", pad_inches=0)
plt.show()
