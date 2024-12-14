Enhancing Practicality and Efficiency of Deepfake Detection
==========================


# Preprocessing

Example for DFDC training part 0:
```bash
part_name=dfdc_train_part_0
export DFDC_DATASET_PATH=./dfdc2020/train_videos/$part_name
export DFDC_PREPROCESSED_DATASET_PATH=./dfdc_preprocessed/$part_name
python $HOME/dev/scripts/preprocessing_dfdc_videos_to_videos.py
```

Example for DFDC test:
```bash
echo "Processing test set"
export DFDC_DATASET_PATH=.#dfdc_test_set
export DFDC_PREPROCESSED_DATASET_PATH=./dfdc_preprocessed/test_set
python ./scripts/preprocessing_dfdc_videos_to_videos.py --is_test True
```

Example for CelebDF:
```bash
python scripts/preprocessing_celebdfv2_videos_to_images.py
```

# Training
```bash
python scripts/train_network_single_image.py
```

# Testing
```bash
python scripts/train_network_single_image.py --test
```

