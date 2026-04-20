import os

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10
MIN_LR = 1e-7
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATA_ROOT = "/kaggle/input/isic-2018-challenge"
IMAGE_DIR = os.path.join(DATA_ROOT, "ISIC2018_Task1-2_Training_Input_x2")
MASK_DIR = os.path.join(DATA_ROOT, "ISIC2018_Task1_Training_GroundTruth_x2")

OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = "cuda"
