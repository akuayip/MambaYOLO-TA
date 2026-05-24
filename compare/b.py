import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the files
df_8 = pd.read_csv('../output_dir/freeze-backbone/7_lr0.01_b8_optSGD_e200/results.csv')
df_16 = pd.read_csv('../output_dir/freeze-backbone/8_lr0.01_b16_optSGD_e200/results.csv')
df_32 = pd.read_csv('../output_dir/freeze-backbone/9_lr0.01_b32_optSGD_e200/results.csv')

# Strip whitespace from column names
df_8.columns = [c.strip() for c in df_8.columns]
df_16.columns = [c.strip() for c in df_16.columns]
df_32.columns = [c.strip() for c in df_32.columns]

# Output directory
output_dir = '../output_dir/compare'
os.makedirs(output_dir, exist_ok=True)

# Summary of key metrics
summary = {}
for name, df in zip(['8 Batch Size', '16 Batch Size', '32 Batch Size'], [df_8, df_16, df_32]):
    summary[name] = {
        'Max mAP50': df['metrics/mAP50(B)'].max(),
        'Final mAP50': df['metrics/mAP50(B)'].iloc[-1],
        'Max mAP50-95': df['metrics/mAP50-95(B)'].max(),
        'Final mAP50-95': df['metrics/mAP50-95(B)'].iloc[-1],
        'Min Val Box Loss': df['val/box_loss'].min(),
        'Final Val Box Loss': df['val/box_loss'].iloc[-1],
        'Min Val Cls Loss': df['val/cls_loss'].min(),
        'Final Val Cls Loss': df['val/cls_loss'].iloc[-1]
    }

summary_df = pd.DataFrame(summary).T
print(summary_df.to_string())

# Plot mAP50(B) for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_8['epoch'], df_8['metrics/mAP50(B)'], label='8 Batch Size')
plt.plot(df_16['epoch'], df_16['metrics/mAP50(B)'], label='16 Batch Size')
plt.plot(df_32['epoch'], df_32['metrics/mAP50(B)'], label='32 Batch Size')
plt.title('mAP50(B) Comparison across different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('mAP50(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'batch_size_map50_comparison.svg'))

# Plot mAP50-95(B) for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_8['epoch'], df_8['metrics/mAP50-95(B)'], label='8 Batch Size')
plt.plot(df_16['epoch'], df_16['metrics/mAP50-95(B)'], label='16 Batch Size')
plt.plot(df_32['epoch'], df_32['metrics/mAP50-95(B)'], label='32 Batch Size')
plt.title('mAP50-95(B) Comparison across different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'batch_size_map50-95_comparison.svg'))

# Plot Validation Loss for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_8['epoch'], df_8['val/box_loss'], label='8 Batch Size')
plt.plot(df_16['epoch'], df_16['val/box_loss'], label='16 Batch Size')
plt.plot(df_32['epoch'], df_32['val/box_loss'], label='32 Batch Size')
plt.title('Val Box Loss Comparison across different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Val Box Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'batch_size_val_box_loss_comparison.svg'))

# Plot Classification Loss for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_8['epoch'], df_8['val/cls_loss'], label='8 Batch Size')
plt.plot(df_16['epoch'], df_16['val/cls_loss'], label='16 Batch Size')
plt.plot(df_32['epoch'], df_32['val/cls_loss'], label='32 Batch Size')
plt.title('Val Cls Loss Comparison across different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel('Val Cls Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'batch_size_val_cls_loss_comparison.svg'))