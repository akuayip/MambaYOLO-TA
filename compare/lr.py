import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the files
df_0_01 = pd.read_csv('../output_dir/freeze-backbone/1_baseline_lr0.01_b8_optSGD_e100/results.csv')
df_0_001 = pd.read_csv('../output_dir/freeze-backbone/2_lr0.001_b8_optSGD_e100/results.csv')
df_0_005 = pd.read_csv('../output_dir/freeze-backbone/3_lr0.005_b8_optSGD_e100/results.csv')

# Strip whitespace from column names
df_0_01.columns = [c.strip() for c in df_0_01.columns]
df_0_001.columns = [c.strip() for c in df_0_001.columns]
df_0_005.columns = [c.strip() for c in df_0_005.columns]

# Output directory
output_dir = '../output_dir/compare'
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store results
results_summary = {}

for name, df in zip(['LR 0.01', 'LR 0.001', 'LR 0.005'], [df_0_01, df_0_001, df_0_005]):
    summary = {
        'Max mAP50': df['metrics/mAP50(B)'].max(),
        'Max mAP50-95': df['metrics/mAP50-95(B)'].max(),
        'Min Val Box Loss': df['val/box_loss'].min(),
        'Min Val Cls Loss': df['val/cls_loss'].min(),
        'Final Epoch mAP50': df['metrics/mAP50(B)'].iloc[-1],
        'Final Epoch mAP50-95': df['metrics/mAP50-95(B)'].iloc[-1]
    }
    results_summary[name] = summary

summary_df = pd.DataFrame(results_summary).T
print(summary_df.to_string())

# Plot mAP50(B) for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_0_01['epoch'], df_0_01['metrics/mAP50(B)'], label='LR 0.01')
plt.plot(df_0_001['epoch'], df_0_001['metrics/mAP50(B)'], label='LR 0.001')
plt.plot(df_0_005['epoch'], df_0_005['metrics/mAP50(B)'], label='LR 0.005')
plt.title('Comparison of mAP50(B) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP50(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'lr_mAP50_comparison.svg'))

# Plot mAP50-95(B) for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_0_01['epoch'], df_0_01['metrics/mAP50-95(B)'], label='LR 0.01')
plt.plot(df_0_001['epoch'], df_0_001['metrics/mAP50-95(B)'], label='LR 0.001')
plt.plot(df_0_005['epoch'], df_0_005['metrics/mAP50-95(B)'], label='LR 0.005')
plt.title('Comparison of mAP50-95(B) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'lr_mAP50-95_comparison.svg'))

# Plot Validation Loss for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_0_01['epoch'], df_0_01['val/box_loss'], label='LR 0.01')
plt.plot(df_0_001['epoch'], df_0_001['val/box_loss'], label='LR 0.001')
plt.plot(df_0_005['epoch'], df_0_005['val/box_loss'], label='LR 0.005')
plt.title('Comparison of Val Box Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Val Box Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'lr_val_loss_comparison.svg'))

# Plot Validation Cls Loss comparison
plt.figure(figsize=(10, 6))
plt.plot(df_0_01['epoch'], df_0_01['val/cls_loss'], label='LR 0.01')
plt.plot(df_0_001['epoch'], df_0_001['val/cls_loss'], label='LR 0.001')
plt.plot(df_0_005['epoch'], df_0_005['val/cls_loss'], label='LR 0.005')
plt.title('Comparison of Val Cls Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Val Cls Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'lr_cls_loss_comparison.svg'))