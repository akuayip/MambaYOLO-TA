import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the files
df_100 = pd.read_csv('../output_dir/freeze-backbone/1_baseline_lr0.01_b8_optSGD_e100/results.csv')
df_150 = pd.read_csv('../output_dir/freeze-backbone/6_lr0.01_b8_optSGD_e150/results.csv')
df_200 = pd.read_csv('../output_dir/freeze-backbone/7_lr0.01_b8_optSGD_e200/results.csv')

# Strip whitespace from column names
df_100.columns = [c.strip() for c in df_100.columns]
df_150.columns = [c.strip() for c in df_150.columns]
df_200.columns = [c.strip() for c in df_200.columns]

# Output directory
output_dir = '../output_dir/compare'
os.makedirs(output_dir, exist_ok=True)

# Summary of key metrics
summary = {}
for name, df in zip(['100 Epochs', '150 Epochs', '200 Epochs'], [df_100, df_150, df_200]):
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
plt.plot(df_100['epoch'], df_100['metrics/mAP50(B)'], label='100 Epochs')
plt.plot(df_150['epoch'], df_150['metrics/mAP50(B)'], label='150 Epochs')
plt.plot(df_200['epoch'], df_200['metrics/mAP50(B)'], label='200 Epochs')
plt.title('mAP50(B) Comparison across different Epoch counts')
plt.xlabel('Epoch')
plt.ylabel('mAP50(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epoch_map50_comparison.svg'))

# Plot mAP50-95(B) for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_100['epoch'], df_100['metrics/mAP50-95(B)'], label='100 Epochs')
plt.plot(df_150['epoch'], df_150['metrics/mAP50-95(B)'], label='150 Epochs')
plt.plot(df_200['epoch'], df_200['metrics/mAP50-95(B)'], label='200 Epochs')
plt.title('mAP50-95(B) Comparison across different Epoch counts')
plt.xlabel('Epoch')
plt.ylabel('maAP50-95(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epoch_map50-95_comparison.svg'))

# Plot Validation Loss for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_100['epoch'], df_100['val/box_loss'], label='100 Epochs')
plt.plot(df_150['epoch'], df_150['val/box_loss'], label='150 Epochs')
plt.plot(df_200['epoch'], df_200['val/box_loss'], label='200 Epochs')
plt.title('Val Box Loss Comparison across different Epoch counts')
plt.xlabel('Epoch')
plt.ylabel('Val Box Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epoch_val_loss_comparison.svg'))

# Plot Classification Loss for comparison
plt.figure(figsize=(10, 6))
plt.plot(df_100['epoch'], df_100['val/box_loss'], label='100 Epochs')
plt.plot(df_150['epoch'], df_150['val/cls_loss'], label='150 Epochs')
plt.plot(df_200['epoch'], df_200['val/cls_loss'], label='200 Epochs')
plt.title('Comparison of Val Cls Loss across different Epoch counts')
plt.xlabel('Epoch')
plt.ylabel('Val Cls Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'epoch_cls_loss_comparison.svg'))