import os

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10
MIN_LR = 1e-7
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATA_ROOT = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation" 

IMAGE_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth"

OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = "cuda"