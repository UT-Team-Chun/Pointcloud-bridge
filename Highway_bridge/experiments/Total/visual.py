import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# --- Configuration ---
BASE_DIR = '.'  # Run from the 'Total' directory
OUTPUT_DIR = 'visualizations_output' # Directory to save plots
METRICS_SUBDIR = 'metrics'
CONF_MATRIX_FILE = 'global_metrics_confusion_matrix_normalized.csv'
CLASS_IOU_FILE = 'global_metrics_class_iou.csv'
PERFORMANCE_FILE = 'global_metrics_performance.csv'

FONT_SIZE = 14
TITLE_FONT_SIZE = 16
DPI = 300
# Use a low-contrast, professional color palette
# Examples: 'pastel', 'muted', 'colorblind', 'Blues', 'Purples'
COLOR_PALETTE = "pastel"
CMAP_COLOR = "Blues" # Colormap for confusion matrix

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Use a style suitable for papers and set font size globally
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE - 2, # Slightly smaller legend
    'figure.titlesize': TITLE_FONT_SIZE,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'pdf.fonttype': 42, # Ensure fonts are editable in PDF editors
    'ps.fonttype': 42
})
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn") # Ignore seaborn warnings about palette


# --- Helper Functions ---

def find_case_folders(base_dir, metrics_subdir):
    """Finds subdirectories containing the specified metrics subdirectory."""
    case_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        metrics_path = os.path.join(item_path, metrics_subdir)
        if os.path.isdir(item_path) and os.path.isdir(metrics_path):
            case_folders.append(item)
    return sorted(case_folders) # Sort for consistent order

def load_data(case_folders, base_dir, metrics_subdir):
    """Loads metrics data from specified case folders."""
    all_conf_matrices = {}
    all_class_ious = {}
    all_performances = {}
    class_names_consistent = None

    print("Loading data...")
    for case_name in case_folders:
        metrics_path = os.path.join(base_dir, case_name, metrics_subdir)
        conf_matrix_path = os.path.join(metrics_path, CONF_MATRIX_FILE)
        class_iou_path = os.path.join(metrics_path, CLASS_IOU_FILE)
        performance_path = os.path.join(metrics_path, PERFORMANCE_FILE)

        try:
            # Load Confusion Matrix
            if os.path.exists(conf_matrix_path):
                cm_df = pd.read_csv(conf_matrix_path, index_col=0)
                # Clean potential whitespace in column/index names
                cm_df.columns = cm_df.columns.str.strip()
                cm_df.index = cm_df.index.str.strip()
                all_conf_matrices[case_name] = cm_df
                current_class_names = cm_df.columns.tolist()
                if class_names_consistent is None:
                    class_names_consistent = current_class_names
                elif class_names_consistent != current_class_names:
                    print(f"Warning: Inconsistent class names found in {case_name}. Using names from first case: {class_names_consistent}")

            else:
                print(f"Warning: File not found - {conf_matrix_path}")

            # Load Class IoU
            if os.path.exists(class_iou_path):
                iou_df = pd.read_csv(class_iou_path)
                iou_df.columns = iou_df.columns.str.strip() # Clean column names
                if 'Class' not in iou_df.columns or 'IoU' not in iou_df.columns:
                     print(f"Warning: 'Class' or 'IoU' column missing in {class_iou_path}")
                else:
                    all_class_ious[case_name] = iou_df
            else:
                print(f"Warning: File not found - {class_iou_path}")

            # Load Performance Metrics
            if os.path.exists(performance_path):
                perf_df = pd.read_csv(performance_path)
                perf_df.columns = perf_df.columns.str.strip() # Clean column names
                if 'Metric' not in perf_df.columns or 'Value' not in perf_df.columns:
                     print(f"Warning: 'Metric' or 'Value' column missing in {performance_path}")
                else:
                    all_performances[case_name] = perf_df
            else:
                print(f"Warning: File not found - {performance_path}")

        except Exception as e:
            print(f"Error loading data for case '{case_name}': {e}")

    print("Data loading complete.")
    return all_conf_matrices, all_class_ious, all_performances, class_names_consistent


def plot_confusion_matrix(cm_df, case_name, class_names, output_dir):
    """Plots and saves a single confusion matrix."""
    if cm_df.empty or not class_names:
        print(f"Skipping empty confusion matrix for {case_name}")
        return

    # Multiply by 100 for percentage display
    cm_df_percent = cm_df * 100

    plt.figure(figsize=(8, 6)) # Adjust size as needed
    sns.heatmap(cm_df_percent, annot=True, fmt=".1f", cmap=CMAP_COLOR, # Display one decimal place
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, linecolor='lightgray', cbar=True, square=True)
                # Removed cbar_kws={'edgecolor': 'black', 'linewidth': 1} as it caused errors
    plt.title(f'Normalized Confusion Matrix (%) - {case_name}', fontsize=TITLE_FONT_SIZE)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(pad=1.5) # Add padding to prevent label overlap

    # Save plots
    base_filename = os.path.join(output_dir, f'confusion_matrix_{case_name}')
    try:
        plt.savefig(f'{base_filename}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'{base_filename}.png', format='png', dpi=DPI, bbox_inches='tight')
        print(f"Saved confusion matrix for {case_name}")
    except Exception as e:
        print(f"Error saving confusion matrix for {case_name}: {e}")
    plt.close() # Close the figure to free memory


def plot_combined_bar(data_df, x_col, y_col, hue_col, title, xlabel, ylabel, output_filename_base, output_dir, palette, rotate_x_labels=False):
    """Plots and saves a combined bar chart."""
    if data_df.empty:
        print(f"Skipping empty combined plot: {title}")
        return

    # Adjust dynamic width calculation: further reduce multiplier and adjust cap
    num_categories = len(data_df[x_col].unique())
    num_hues = len(data_df[hue_col].unique())
    # Further reduced multiplier, adjusted min/max width
    dynamic_width = max(6, min(16, num_categories * num_hues * 0.35))
    plt.figure(figsize=(dynamic_width, 6))

    ax = sns.barplot(data=data_df, x=x_col, y=y_col, hue=hue_col, palette=palette,
                     edgecolor='black', linewidth=0.8, # Add black edges to bars
                     errorbar=None) # errorbar=None if no std dev needed
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add value labels on top of bars - format as integer
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=FONT_SIZE - 4, padding=3) # Integer format

    if rotate_x_labels:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=0)

    # Adjust legend position - place explicitly in lower right
    # plt.legend(title=hue_col, loc='best', borderaxespad=0.5) # Original 'best' placement
    plt.legend(title=hue_col, loc='lower right', borderaxespad=0.5, fontsize=FONT_SIZE - 4) # Place legend in lower right, slightly smaller font

    # plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend outside - remove or adjust if legend is inside
    plt.tight_layout(pad=1.0) # Use standard tight_layout

    # Save plots
    try:
        plt.savefig(os.path.join(output_dir, f'{output_filename_base}.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{output_filename_base}.png'), format='png', dpi=DPI, bbox_inches='tight')
        print(f"Saved combined plot: {output_filename_base}")
    except Exception as e:
        print(f"Error saving combined plot {output_filename_base}: {e}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    case_folders = find_case_folders(BASE_DIR, METRICS_SUBDIR)
    if not case_folders:
        print(f"Error: No subfolders containing a '{METRICS_SUBDIR}' directory found in '{os.path.abspath(BASE_DIR)}'.")
        exit()

    print(f"Found {len(case_folders)} case folders: {', '.join(case_folders)}")

    all_conf_matrices, all_class_ious, all_performances, class_names = load_data(
        case_folders, BASE_DIR, METRICS_SUBDIR
    )

    # --- Determine Final Ordered Class Names for Plotting ---
    final_plot_class_names = []
    if class_names:
        temp_names = class_names[:] # Copy
        has_abutment = 'Abutment' in temp_names
        has_pier = 'Pier' in temp_names
        processed_combined = False
        for name in temp_names:
            if name == 'Abutment' or name == 'Pier':
                if not processed_combined:
                    final_plot_class_names.append('Abut./Pier')
                    processed_combined = True
            else:
                final_plot_class_names.append(name)
        if not final_plot_class_names: # Fallback if class_names was empty
             print("Warning: Could not determine final class names.")
             final_plot_class_names = []
    # ---------------------------------------------------------


    # --- Plot Individual Confusion Matrices ---
    if all_conf_matrices and final_plot_class_names: # Use final_plot_class_names
        print("\nPlotting individual confusion matrices...")
        # No need to create plot_class_names here, use final_plot_class_names

        for case_name, cm_df in all_conf_matrices.items():
            try:
                # --- Combine Abutment/Pier rows/columns in the DataFrame ---
                cm_df_processed = cm_df.copy()
                original_cols = cm_df_processed.columns.tolist()
                original_idx = cm_df_processed.index.tolist()

                cols_to_combine = [col for col in ['Abutment', 'Pier'] if col in original_cols]
                rows_to_combine = [row for row in ['Abutment', 'Pier'] if row in original_idx]

                # Combine Columns
                if len(cols_to_combine) > 1:
                    cm_df_processed['Abut./Pier'] = cm_df_processed[cols_to_combine].sum(axis=1)
                    cm_df_processed = cm_df_processed.drop(columns=cols_to_combine)
                elif len(cols_to_combine) == 1 and cols_to_combine[0] not in final_plot_class_names: # Rename only if target name is different
                     cm_df_processed = cm_df_processed.rename(columns={cols_to_combine[0]: 'Abut./Pier'})

                # Combine Rows
                if len(rows_to_combine) > 1:
                    # Ensure columns align before summing, especially after potential column combination
                    current_cols = cm_df_processed.columns
                    combined_row = cm_df_processed.loc[rows_to_combine, current_cols].sum(axis=0)
                    combined_row.name = 'Abut./Pier'
                    cm_df_processed = cm_df_processed.drop(index=rows_to_combine)
                    cm_df_processed = pd.concat([cm_df_processed, pd.DataFrame(combined_row).T])
                elif len(rows_to_combine) == 1 and rows_to_combine[0] not in final_plot_class_names: # Rename only if target name is different
                    cm_df_processed = cm_df_processed.rename(index={rows_to_combine[0]: 'Abut./Pier'})
                # ----------------------------------------------------------

                # Reindex using the final ordered class names
                cm_df_reordered = cm_df_processed.reindex(index=final_plot_class_names, columns=final_plot_class_names, fill_value=0)
                plot_confusion_matrix(cm_df_reordered, case_name, final_plot_class_names, OUTPUT_DIR) # Pass final names
            except Exception as e:
                 print(f"Error processing/plotting confusion matrix for {case_name}: {e}")
                 # import traceback
                 # traceback.print_exc()
    else:
        print("\nSkipping individual confusion matrix plots (no data or consistent class names found).")

    # --- Prepare and Plot Combined Class IoU ---
    if all_class_ious and final_plot_class_names: # Check final_plot_class_names
        print("\nProcessing and plotting combined class IoU...")
        combined_iou_list = []
        for case_name, df in all_class_ious.items():
            df_copy = df.copy() # Avoid modifying original dict entry
            df_copy['Case'] = case_name
            combined_iou_list.append(df_copy)

        if combined_iou_list:
            combined_iou_df = pd.concat(combined_iou_list, ignore_index=True)

            # --- Combine 'Abutment' and 'Pier' ---
            class_replace_map = {'Abutment': 'Abut./Pier', 'Pier': 'Abut./Pier'}
            combined_iou_df['Class'] = combined_iou_df['Class'].replace(class_replace_map)

            # If combining classes, average their IoU values per Case
            if 'Abut./Pier' in combined_iou_df['Class'].unique():
                 print("Combining 'Abutment' and 'Pier' into 'Abut./Pier' and averaging IoU.")
                 combined_iou_df = combined_iou_df.groupby(['Case', 'Class'], as_index=False)['IoU'].mean()

            # --- Ensure correct class order for plotting ---
            # Use final_plot_class_names to define the category order
            combined_iou_df['Class'] = pd.Categorical(combined_iou_df['Class'], categories=final_plot_class_names, ordered=True)
            # Sort by the ordered Class column, then by Case for consistent hue grouping
            combined_iou_df = combined_iou_df.sort_values(by=['Class', 'Case'])
            # Drop rows with NaN categories if any class from final_plot_class_names wasn't in the data after merge
            combined_iou_df.dropna(subset=['Class'], inplace=True)
            # ---------------------------------------------

            # Optional: Convert IoU to percentage if it's not already
            combined_iou_df['IoU'] = combined_iou_df['IoU'] # Keep as original scale unless specified
            plot_combined_bar(
                data_df=combined_iou_df,
                x_col='Class',
                y_col='IoU',
                hue_col='Case',
                title='Class IoU Comparison Across Cases',
                xlabel='Class',
                ylabel='IoU', # Add (%) if converted
                output_filename_base='combined_class_iou',
                output_dir=OUTPUT_DIR,
                palette=COLOR_PALETTE,
                rotate_x_labels=True
            )
        else:
             print("No valid class IoU dataframes found to combine.")
    else:
        print("\nSkipping combined class IoU plot (no data found).")

    # --- Prepare and Plot Combined Overall Performance ---
    if all_performances:
        print("\nProcessing and plotting combined overall performance...")
        combined_perf_list = []
        for case_name, df in all_performances.items():
            df_copy = df.copy()
            df_copy['Case'] = case_name
            combined_perf_list.append(df_copy)

        if combined_perf_list:
            combined_perf_df = pd.concat(combined_perf_list, ignore_index=True)
            # Optional: Convert Value to percentage if it's not already
            combined_perf_df['Value'] = combined_perf_df['Value'] # Keep as original scale unless specified
            plot_combined_bar(
                data_df=combined_perf_df,
                x_col='Metric',
                y_col='Value',
                hue_col='Case',
                title='Overall Performance Comparison Across Cases',
                xlabel='Metric',
                ylabel='Score', # Add (%) if converted
                output_filename_base='combined_overall_performance',
                output_dir=OUTPUT_DIR,
                palette=COLOR_PALETTE,
                rotate_x_labels=True # Keep rotation for performance metrics as they can be long
            )
        else:
             print("No valid performance dataframes found to combine.")
    else:
        print("\nSkipping combined overall performance plot (no data found).")

    print(f"\nVisualizations saved to '{os.path.abspath(OUTPUT_DIR)}' directory.")
    print("Script finished.")
