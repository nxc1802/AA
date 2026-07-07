import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configure premium custom matplotlib style (no seaborn dependency)
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'grid.color': '#e9ecef',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'figure.facecolor': '#ffffff'
})

def make_paper_assets(csv_path='results/final_results.csv', output_dir='results/paper_assets'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Filter out Clean for progression plots to keep focus
    df_plot = df[df['Attack'] != 'Clean'].copy()
    df_plot['PSNR'] = df_plot['PSNR'].replace(float('inf'), np.nan)
    
    # Define premium colors for attacks
    color_map = {
        'FGSM': '#e63946',
        'BIM': '#f4a261',
        'PGD': '#e76f51',
        'Sparse (k=0.1)': '#a8dadc',
        'Sparse (k=0.2)': '#457b9d',
        'Sparse (k=0.3)': '#1d3557',
        'Sparse (k=0.4)': '#2a9d8f',
        'Sparse (k=0.5)': '#264653',
        'Sparse (k=0.6)': '#6a0dad',
        'Sparse (k=0.7)': '#b5179e',
        'Sparse (k=0.8)': '#7209b7',
        'Sparse (k=0.9)': '#3f37c9',
        'Sparse (k=1.0)': '#4895ef'
    }
    
    metrics = ['ASR', 'Accuracy', 'SSIM', 'PSNR', 'LPIPS']
    models = df['Model'].unique()
    
    # 1. PROGRESSION PLOTS (ASR, Accuracy, SSIM, PSNR vs Iteration)
    for model in models:
        df_model = df_plot[df_plot['Model'] == model].copy()
        
        # Format labels
        def get_label(row):
            if row['Attack'] == 'Sparse':
                return f"Sparse (k={row['K-Ratio']:.1f})"
            return row['Attack']
        
        df_model['Method'] = df_model.apply(get_label, axis=1)
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.grid(True)
            
            # Group by Method and Iteration to plot means
            grouped = df_model.groupby(['Method', 'Iteration'])[metric].mean().reset_index()
            
            for method in sorted(grouped['Method'].unique()):
                method_data = grouped[grouped['Method'] == method]
                color = color_map.get(method, '#333333')
                
                # Markers and styles
                marker = 'o' if 'Sparse' in method else 's' if 'PGD' in method else '^'
                linestyle = '-' if 'Sparse' in method else '--'
                
                ax.plot(method_data['Iteration'], method_data[metric], 
                        label=method, color=color, marker=marker, 
                        linestyle=linestyle, linewidth=2, markersize=5)
            
            ax.set_title(f'{metric} vs Iterations ({model} Model)', fontsize=13, fontweight='bold', pad=15)
            ax.set_ylabel(metric, fontsize=11, fontweight='semibold')
            ax.set_xlabel('Iteration', fontsize=11, fontweight='semibold')
            
            if metric in ['ASR', 'Accuracy', 'SSIM']:
                ax.set_ylim(-0.05, 1.05)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'progression_{metric.lower()}_{model.lower()}.png'), bbox_inches='tight')
            plt.close()

    # 2. BAR CHARTS AT ITERATION 10 (Acc, ASR, SSIM, PSNR vs K-Ratio)
    df_iter10 = df_plot[(df_plot['Attack'] == 'Sparse') & (df_plot['Iteration'] == 10)].copy()
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True, axis='y')
        
        # Calculate means grouped by K-Ratio and Model
        summary_bar = df_iter10.groupby(['K-Ratio', 'Model'])[metric].mean().unstack()
        
        # Grouped bar width and indices
        x = np.arange(len(summary_bar.index))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, summary_bar.get('Standard', 0.0), width, label='Standard Model', color='#457b9d', edgecolor='#dee2e6')
        rects2 = ax.bar(x + width/2, summary_bar.get('Robust', 0.0), width, label='Robust Model', color='#e76f51', edgecolor='#dee2e6')
        
        ax.set_title(f'Final {metric} vs K-Ratio (Iteration 10)', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel(metric, fontsize=11, fontweight='semibold')
        ax.set_xlabel('K-Ratio (Proportion of Perturbed Pixels)', fontsize=11, fontweight='semibold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{val:.1f}" for val in summary_bar.index])
        
        if metric in ['ASR', 'Accuracy', 'SSIM']:
            ax.set_ylim(0, 1.1)
            
        ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dee2e6')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bar_kratio_{metric.lower()}.png'), bbox_inches='tight')
        plt.close()

    # 3. LATEX TABLES
    for mtype in models:
        df_m = df[df['Model'] == mtype]
        df_final = df_m[((df_m['Iteration'] == 10) | 
                         (df_m['Attack'] == 'Clean') | 
                         (df_m['Attack'] == 'FGSM'))]
        
        summary = df_final.groupby(['Attack', 'K-Ratio']).mean(numeric_only=True).reset_index()
        
        # Format strings for LaTeX
        summary['Accuracy'] = (summary['Accuracy'] * 100).round(2).astype(str) + r'\%'
        summary['ASR'] = (summary['ASR'] * 100).round(2).astype(str) + r'\%'
        summary['Sparsity'] = (summary['Sparsity'] * 100).round(1).astype(str) + r'\%'
        summary['SSIM'] = summary['SSIM'].round(4)
        summary['LPIPS'] = summary['LPIPS'].round(4)
        summary['PSNR'] = summary['PSNR'].round(2).replace(float('inf'), r'\infty')
        
        # Filter unwanted columns
        cols_to_keep = ['Attack', 'K-Ratio', 'Accuracy', 'ASR', 'L0', 'Sparsity', 'L2', 'Linf', 'SSIM', 'PSNR', 'LPIPS']
        summary = summary[cols_to_keep]
        
        tex_path = os.path.join(output_dir, f'table_{mtype.lower()}.tex')
        with open(tex_path, 'w') as f:
            f.write(summary.to_latex(index=False, escape=False))

    print(f"Expanded paper assets generated successfully in {output_dir}")

if __name__ == "__main__":
    make_paper_assets()
