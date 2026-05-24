import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the files
df_sgd = pd.read_csv('../output_dir/freeze-backbone/1_baseline_lr0.01_b8_optSGD_e100/results.csv')
df_adamw = pd.read_csv('../output_dir/freeze-backbone/4_lr0.01_b8_optAdamW_e100/results.csv')
df_rmsprop = pd.read_csv('../output_dir/freeze-backbone/5_lr0.01_b8_optRMSProp_e100/results.csv')

# Strip whitespace from column names
df_adamw.columns = [c.strip() for c in df_adamw.columns]
df_sgd.columns = [c.strip() for c in df_sgd.columns]
df_rmsprop.columns = [c.strip() for c in df_rmsprop.columns]

# Output directory
output_dir = '../output_dir/compare'
os.makedirs(output_dir, exist_ok=True)

# Summarize best values
summary = {}
for name, df in zip(['SGD', 'AdamW',  'RMSprop'], [df_sgd, df_adamw, df_rmsprop]):
    summary[name] = {
        'Max mAP50': df['metrics/mAP50(B)'].max(),
        'Max mAP50-95': df['metrics/mAP50-95(B)'].max(),
        'Min Val Box Loss': df['val/box_loss'].min(),
        'Min Val Cls Loss': df['val/cls_loss'].min(),
        'Final mAP50': df['metrics/mAP50(B)'].iloc[-1],
        'Final mAP50-95': df['metrics/mAP50-95(B)'].iloc[-1]
    }

summary_df = pd.DataFrame(summary).T
print(summary_df.to_string())

# Plot mAP50 comparison
plt.figure(figsize=(10, 6))
plt.plot(df_sgd['epoch'], df_sgd['metrics/mAP50(B)'], label='SGD')
plt.plot(df_adamw['epoch'], df_adamw['metrics/mAP50(B)'], label='AdamW')
plt.plot(df_rmsprop['epoch'], df_rmsprop['metrics/mAP50(B)'], label='RMSprop')
plt.title('Comparison of mAP50(B):  SGD vs AdamW vs RMSprop')
plt.xlabel('Epoch')
plt.ylabel('mAP50(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'optim_mAP50_comparison.svg'))

# Plot mAP50-95 comparison
plt.figure(figsize=(10, 6))
plt.plot(df_sgd['epoch'], df_sgd['metrics/mAP50-95(B)'], label='SGD')
plt.plot(df_adamw['epoch'], df_adamw['metrics/mAP50-95(B)'], label='AdamW')
plt.plot(df_rmsprop['epoch'], df_rmsprop['metrics/mAP50-95(B)'], label='RMSprop')
plt.title('Comparison of mAP50-95(B):  SGD vs AdamW vs RMSprop')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95(B)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'optim_mAP50-95_comparison.svg'))

# Plot Validation Box Loss comparison
plt.figure(figsize=(10, 6))
plt.plot(df_sgd['epoch'], df_sgd['val/box_loss'], label='SGD')
plt.plot(df_adamw['epoch'], df_adamw['val/box_loss'], label='AdamW')
plt.plot(df_rmsprop['epoch'], df_rmsprop['val/box_loss'], label='RMSprop')
plt.title('Comparison of Val Box Loss: AdamW vs SGD vs RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Val Box Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'optim_val_loss_comparison.svg'))

# Plot Validation Cls Loss comparison
plt.figure(figsize=(10, 6))
plt.plot(df_sgd['epoch'], df_sgd['val/cls_loss'], label='SGD')
plt.plot(df_adamw['epoch'], df_adamw['val/cls_loss'], label='AdamW')
plt.plot(df_rmsprop['epoch'], df_rmsprop['val/cls_loss'], label='RMSprop')
plt.title('Comparison of Val Cls Loss: AdamW vs SGD vs RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Val Cls Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'optim_cls_loss_comparison.svg'))