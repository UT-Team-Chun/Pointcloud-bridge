import os
import pandas as pd
import glob
import re
import numpy as np

# --- Configuration ---
BASE_DIR = '.'  # Run from the 'Total' directory
METRICS_SUBDIR = 'metrics'
GLOBAL_PERFORMANCE_FILE = 'global_metrics_performance.csv'
CLASS_IOU_PIVOT_FILE = 'class_iou_pivot.csv'
FILE_PERFORMANCE_PATTERN = 'file_metrics_*.csv'
OUTPUT_CSV_FILE = 'combined_quantitative_metrics.csv'
OUTPUT_LATEX_FILE = 'combined_quantitative_metrics.tex'
DATASET_PREFIX = 'DS-'
MEAN_SUFFIX = ' Mean' # Use Mean for academic papers

# --- Helper Functions ---

def find_dataset_folders(base_dir, prefix):
    """Finds dataset folders matching the prefix."""
    folders = []
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith(prefix):
                metrics_path = os.path.join(item_path, METRICS_SUBDIR)
                if os.path.isdir(metrics_path):
                    folders.append(item)
                else:
                    print(f"Warning: Found dataset folder '{item}' but missing '{METRICS_SUBDIR}' subdirectory. Skipping.")
    except FileNotFoundError:
        print(f"Error: Base directory '{os.path.abspath(base_dir)}' not found.")
        return []
    return sorted(folders)

def extract_bridge_name(filename):
    """Extracts bridge name like 'bridge-3.h5' from filename."""
    match = re.search(r'file_metrics_(.+)_performance\.csv', os.path.basename(filename))
    return match.group(1) if match else None

def generate_latex_table(df, caption="Summary of Quantitative Evaluation Metrics.", label="tab:quantitative_summary"):
    """Generates a LaTeX table string from a DataFrame using booktabs."""
    num_cols = len(df.columns)
    # Adjust column format if names are long, maybe use 'p' columns? For now, stick with 'r'.
    col_format = "l" + "r" * num_cols  # l for metric names, r for values

    latex_string = "\\begin{table*}[!htbp]\n"
    latex_string += "  \\centering\n"
    latex_string += f"  \\caption{{{caption}}}\n"
    latex_string += f"  \\label{{{label}}}\n"
    # Consider adjusting resizebox if column names make table too wide even when resized
    latex_string += "  \\resizebox{\\textwidth}{!}{%\n" # Make table fit text width
    latex_string += f"  \\begin{{tabular}}{{{col_format}}}\n"
    latex_string += "    \\toprule\n"

    # Header row - replace underscores and handle potential multi-word names
    header_names = []
    for col in df.columns:
        # Basic replacement, might need more sophisticated handling for complex names
        processed_col = col.replace('_', ' ').replace('.h5', '')
        header_names.append(processed_col)
    header = "Metric & " + " & ".join(header_names) + " \\\\\n"
    latex_string += "    " + header
    latex_string += "    \\midrule\n"

    # Data rows
    first_iou_written = False
    iou_separator_added = False # Flag to add midrule only once before IoUs

    for index, row in df.iterrows():
        metric_name = index.replace('_', ' ') # Nicer display name

        # Add separator before the first IoU metric appears
        if index.startswith('IoU_') and not iou_separator_added:
             latex_string += "    \\midrule\n"
             iou_separator_added = True

        # Specific handling for combined metric name
        if index == 'IoU_Abutment/Pier':
            metric_name = 'IoU Abutment/Pier' # Display name in table

        # Add units (%) - assuming values are 0-100
        # Check if the metric is one of the standard ones needing (%)
        percent_metrics = ['mIoU', 'OA', 'mAcc', 'Precision', 'Recall', 'F1 Score'] + [idx for idx in df.index if idx.startswith('IoU_')]
        unit = " (\\%)" if index in percent_metrics else ""

        row_values = [f"{val:.2f}" if isinstance(val, (float, np.number)) and not pd.isna(val) else ('-' if pd.isna(val) else str(val)) for val in row]
        latex_string += f"    {metric_name}{unit} & " + " & ".join(row_values) + " \\\\\n"


    latex_string += "    \\bottomrule\n"
    latex_string += "  \\end{tabular}%\n"
    latex_string += "  }\n" # End resizebox
    latex_string += "\\end{table*}\n"

    return latex_string

# --- Main Script ---
if __name__ == "__main__":
    dataset_folders = find_dataset_folders(BASE_DIR, DATASET_PREFIX)

    if not dataset_folders:
        print(f"Error: No dataset folders found starting with '{DATASET_PREFIX}' and containing a '{METRICS_SUBDIR}' subdir in '{os.path.abspath(BASE_DIR)}'.")
        exit()

    print(f"Found {len(dataset_folders)} dataset folders: {', '.join(dataset_folders)}")

    all_metrics_data = {} # Dictionary to hold Series/DataFrames before combining

    for ds_folder in dataset_folders:
        print(f"\nProcessing dataset: {ds_folder}")
        metrics_path = os.path.join(BASE_DIR, ds_folder, METRICS_SUBDIR)

        # 1. Load Global Performance for the dataset
        global_perf_path = os.path.join(metrics_path, GLOBAL_PERFORMANCE_FILE)
        if os.path.exists(global_perf_path):
            try:
                perf_df = pd.read_csv(global_perf_path)
                perf_df.columns = perf_df.columns.str.strip()
                # Ensure correct columns and set Metric as index
                if 'Metric' in perf_df.columns and 'Value' in perf_df.columns:
                     perf_series = perf_df.set_index('Metric')['Value']
                     # Rename F1_score for consistency if needed
                     perf_series = perf_series.rename(index={'F1_score': 'F1 Score'})
                     all_metrics_data[ds_folder] = perf_series
                     print(f"  Loaded global performance for {ds_folder}.")
                else:
                    print(f"  Warning: Missing 'Metric' or 'Value' column in {global_perf_path}. Skipping global performance.")
            except Exception as e:
                print(f"  Error reading {global_perf_path}: {e}")
        else:
            print(f"  Warning: Global performance file not found: {global_perf_path}")

        # 2. Load Class IoU Pivot for individual files within the dataset
        class_iou_path = os.path.join(metrics_path, CLASS_IOU_PIVOT_FILE)
        if os.path.exists(class_iou_path):
            try:
                iou_pivot_df = pd.read_csv(class_iou_path)
                iou_pivot_df.columns = iou_pivot_df.columns.str.strip()
                if 'Class' in iou_pivot_df.columns:
                    iou_pivot_df = iou_pivot_df.set_index('Class')
                    # Add prefix to bridge file columns and prefix 'IoU_' to index
                    iou_pivot_df.columns = [f"{ds_folder}/{col}" for col in iou_pivot_df.columns]
                    iou_pivot_df.index = 'IoU_' + iou_pivot_df.index
                    # Merge into the main dictionary
                    for col in iou_pivot_df.columns:
                        all_metrics_data[col] = iou_pivot_df[col]
                    print(f"  Loaded class IoU pivot for {ds_folder}.")
                else:
                     print(f"  Warning: Missing 'Class' column in {class_iou_path}. Skipping class IoU pivot.")
            except Exception as e:
                print(f"  Error reading {class_iou_path}: {e}")
        else:
            print(f"  Warning: Class IoU pivot file not found: {class_iou_path}")

        # 3. Load File Performance for individual files within the dataset
        file_perf_paths = glob.glob(os.path.join(metrics_path, FILE_PERFORMANCE_PATTERN))
        if file_perf_paths:
            for file_path in file_perf_paths:
                bridge_name = extract_bridge_name(file_path)
                if bridge_name:
                    col_name = f"{ds_folder}/{bridge_name}"
                    try:
                        file_perf_df = pd.read_csv(file_path)
                        file_perf_df.columns = file_perf_df.columns.str.strip()
                        if 'Metric' in file_perf_df.columns and 'Value' in file_perf_df.columns:
                            file_perf_series = file_perf_df.set_index('Metric')['Value']
                            file_perf_series = file_perf_series.rename(index={'F1_score': 'F1 Score'})
                            # Merge file performance (only if not already present from IoU pivot handling)
                            if col_name not in all_metrics_data:
                                all_metrics_data[col_name] = pd.Series(dtype=float) # Initialize if needed
                            # Update existing series or add new metrics
                            all_metrics_data[col_name] = all_metrics_data[col_name].combine_first(file_perf_series)
                            print(f"    Loaded file performance for {bridge_name}.")
                        else:
                            print(f"    Warning: Missing 'Metric' or 'Value' column in {file_path}. Skipping.")
                    except Exception as e:
                        print(f"    Error reading {file_path}: {e}")
                else:
                    print(f"  Warning: Could not extract bridge name from {file_path}")
        else:
             print(f"  Warning: No file performance files found matching '{FILE_PERFORMANCE_PATTERN}' in {metrics_path}")


    # Combine all data into a single DataFrame
    if not all_metrics_data:
        print("\nError: No metrics data was loaded successfully. Exiting.")
        exit()

    combined_df = pd.DataFrame(all_metrics_data)

    # Combine Abutment and Pier IoU
    has_abutment = 'IoU_Abutment' in combined_df.index
    has_pier = 'IoU_Pier' in combined_df.index
    combined_iou_row = pd.Series(name='IoU_Abutment/Pier', dtype=float)

    if has_abutment or has_pier:
        print("\nCombining Abutment and Pier IoU...")
        for col in combined_df.columns:
            abutment_val = combined_df.loc['IoU_Abutment', col] if has_abutment else np.nan
            pier_val = combined_df.loc['IoU_Pier', col] if has_pier else np.nan

            valid_values = [v for v in [abutment_val, pier_val] if pd.notna(v)]

            if len(valid_values) > 0:
                combined_iou_row[col] = np.mean(valid_values)
            else:
                combined_iou_row[col] = np.nan

        # Add the new combined row
        combined_df = pd.concat([combined_df, combined_iou_row.to_frame().T])

        # Remove original rows
        rows_to_drop = []
        if has_abutment:
            rows_to_drop.append('IoU_Abutment')
        if has_pier:
            rows_to_drop.append('IoU_Pier')
        combined_df = combined_df.drop(index=rows_to_drop)
        print("  Combined IoU row 'IoU_Abutment/Pier' created.")
    else:
        print("\nNo 'IoU_Abutment' or 'IoU_Pier' found to combine.")


    # Reorder metrics for better readability
    metric_order = [
        'mIoU', 'OA', 'mAcc', 'Precision', 'Recall', 'F1 Score'
    ]
    # Define desired IoU order
    iou_order = ['IoU_Background']
    if 'IoU_Abutment/Pier' in combined_df.index:
        iou_order.append('IoU_Abutment/Pier')
    # Add remaining IoU metrics alphabetically, excluding ones already placed
    remaining_ious = sorted([idx for idx in combined_df.index if idx.startswith('IoU_') and idx not in iou_order])
    iou_order.extend(remaining_ious)

    final_metric_order = metric_order + iou_order
    # Include any other metrics found that weren't explicitly ordered
    other_metrics = [idx for idx in combined_df.index if idx not in final_metric_order]
    final_metric_order.extend(other_metrics)

    # Filter out metrics not present in the DataFrame before reindexing
    final_metric_order = [m for m in final_metric_order if m in combined_df.index]
    combined_df = combined_df.reindex(index=final_metric_order)
    print(f"\nReordered metrics. Final order: {', '.join(final_metric_order)}")


    # Calculate Overall Average across datasets (DS-*)
    dataset_cols = [col for col in combined_df.columns if col in dataset_folders]
    if dataset_cols:
        # Calculate mean only for numeric columns, handle potential NaNs
        numeric_df = combined_df[dataset_cols].apply(pd.to_numeric, errors='coerce')
        combined_df['Overall Average'] = numeric_df.mean(axis=1, skipna=True)
        print("\nCalculated 'Overall Average' across datasets.")
    else:
        print("\nWarning: No dataset-level columns found to calculate 'Overall Average'.")

    # Rename Dataset Columns
    rename_mapping = {ds_folder: f"{ds_folder}{MEAN_SUFFIX}" for ds_folder in dataset_folders}
    combined_df = combined_df.rename(columns=rename_mapping)
    print(f"Renamed dataset columns with suffix '{MEAN_SUFFIX}'.")


    # Sort columns logically: Files first, then Dataset Mean, then Overall Average
    sorted_columns = []
    processed_cols = set() # Keep track of columns already added

    for ds_folder in dataset_folders:
        # Find files belonging to this dataset
        ds_file_cols = sorted([col for col in combined_df.columns if col.startswith(f"{ds_folder}/")])
        sorted_columns.extend(ds_file_cols)
        processed_cols.update(ds_file_cols)

        # Add the renamed dataset mean column
        ds_mean_col = f"{ds_folder}{MEAN_SUFFIX}"
        if ds_mean_col in combined_df.columns:
            sorted_columns.append(ds_mean_col)
            processed_cols.add(ds_mean_col)
        else:
             print(f"Warning: Renamed mean column '{ds_mean_col}' not found for sorting.")

    # Add Overall Average if it exists
    if 'Overall Average' in combined_df.columns:
         # Ensure it hasn't been accidentally processed
         if 'Overall Average' not in processed_cols:
              sorted_columns.append('Overall Average')
              processed_cols.add('Overall Average')

    # Add any remaining columns that weren't caught by the logic (shouldn't happen ideally)
    remaining_cols = [col for col in combined_df.columns if col not in processed_cols]
    if remaining_cols:
        print(f"Warning: Adding remaining unprocessed columns to the end: {', '.join(remaining_cols)}")
        sorted_columns.extend(sorted(remaining_cols)) # Sort remaining alphabetically


    combined_df = combined_df[sorted_columns]
    print(f"\nReordered columns. Final order: {', '.join(sorted_columns)}")


    # Save to CSV
    try:
        # Format numbers to 2 decimal places for CSV readability
        combined_df_formatted = combined_df.copy()
        for col in combined_df_formatted.columns:
             # Apply formatting only to numeric columns
             if pd.api.types.is_numeric_dtype(combined_df_formatted[col]):
                  combined_df_formatted[col] = combined_df_formatted[col].map('{:.2f}'.format, na_action='ignore')

        combined_df_formatted.to_csv(OUTPUT_CSV_FILE)
        print(f"\nSuccessfully saved combined metrics to: {os.path.abspath(OUTPUT_CSV_FILE)}")
    except Exception as e:
        print(f"\nError saving CSV file: {e}")

    # Generate and Save LaTeX Table
    try:
        # Pass the original DataFrame with float precision to LaTeX function
        latex_code = generate_latex_table(combined_df)
        with open(OUTPUT_LATEX_FILE, 'w') as f:
            f.write(latex_code)
        print(f"Successfully saved LaTeX table to: {os.path.abspath(OUTPUT_LATEX_FILE)}")
        print("\nLaTeX table requires the following packages in your preamble:")
        print("  \\usepackage{booktabs}")
        print("  \\usepackage{graphicx} % For resizebox")
        print("  \\usepackage{amsmath} % Often useful for math symbols like %")

    except Exception as e:
        print(f"\nError generating or saving LaTeX file: {e}")

    print("\nScript finished.")
