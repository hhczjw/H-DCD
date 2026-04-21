"""
Testing script for H-DCD
"""
from run import H_DCD_run

# Test on IEMOCAP (emotion classification)
H_DCD_run(
    dataset_name='iemocap',
    seeds=[1111],
    model_save_dir="./checkpoints",
    res_save_dir="./results",
    log_dir="./logs",
    mode='test',
    gpu_ids=[0],
    num_workers=4,
    verbose_level=1
)

# Uncomment below to test on other datasets:

# Test on MOSI (sentiment regression)
# H_DCD_run(
#     dataset_name='mosi',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='test',
#     gpu_ids=[0]
# )

# Test on MOSEI (sentiment regression)
# H_DCD_run(
#     dataset_name='mosei',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='test',
#     gpu_ids=[0]
# )

# Test on MELD (emotion classification)
# H_DCD_run(
#     dataset_name='meld',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='test',
#     gpu_ids=[0]
# )
