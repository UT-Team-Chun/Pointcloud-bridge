import numpy as np
import laspy
import open3d as o3d
import os
import glob
import logging
import re

# Setup logging (can be configured externally if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_npz_block_data(npz_file):
    """Loads point cloud data from a single NPZ block file."""
    logging.debug(f"Attempting to load data from {npz_file}")
    try:
        # Use allow_pickle=False for security unless you specifically need it
        with np.load(npz_file, allow_pickle=False) as data:
            if 'points' not in data or 'colors' not in data:
                logging.warning(f"Skipping {npz_file}: Missing 'points' or 'colors' key.")
                return None, None

            points = data['points'].astype(np.float64) # Use float64 for precision
            colors = data['colors'].astype(np.float64) # Use float64, will be normalized

            if points.ndim != 2 or points.shape[1] != 3:
                logging.warning(f"Skipping {npz_file}: 'points' array has incorrect shape {points.shape}.")
                return None, None
            if colors.ndim != 2 or colors.shape[1] != 3:
                logging.warning(f"Skipping {npz_file}: 'colors' array has incorrect shape {colors.shape}.")
                return None, None
            if points.shape[0] != colors.shape[0]:
                 logging.warning(f"Skipping {npz_file}: Mismatch between number of points ({points.shape[0]}) and colors ({colors.shape[0]}).")
                 return None, None
            if points.shape[0] == 0:
                 logging.warning(f"Skipping {npz_file}: Contains zero points.")
                 return None, None


            logging.debug(f"Successfully loaded {points.shape[0]} points from {npz_file}")

            # Normalize colors to [0, 1] for Open3D, assuming original might be [0, 255] or [0, 1]
            if colors.size > 0 and np.max(colors) > 1.0:
                logging.debug("Normalizing colors from [0, 255] to [0, 1] range.")
                colors /= 255.0
            # Clamp values just in case
            colors = np.clip(colors, 0.0, 1.0)

            return points, colors

    except Exception as e:
        logging.error(f"Error loading {npz_file}: {e}")
        return None, None

def save_to_las(points, colors, output_las_file):
    """Saves point cloud data to a LAS file."""
    if points is None or colors is None:
        logging.error(f"Cannot save LAS file {output_las_file}: No point data provided.")
        return False

    logging.info(f"Preparing to save LAS file to {output_las_file}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_las_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

        # 1. Create Header (LAS 1.2, Point Format 3 for XYZ+RGB)
        header = laspy.LasHeader(version="1.2", point_format=3)
        min_coords = np.min(points, axis=0)
        header.offsets = min_coords
        # Use a small scale factor (e.g., 0.001 for mm precision if coords are in meters)
        header.scales = np.array([0.001, 0.001, 0.001]) # Adjust if needed based on coordinate range

        # 2. Create LAS data object
        las = laspy.LasData(header)

        # 3. Assign coordinates
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        # 4. Assign colors (scaling to 0-65535 for LAS Point Format 3)
        if colors.size > 0:
             # Ensure colors are in [0, 1] before scaling
            if np.max(colors) > 1.0: # Double check normalization
                 colors /= 255.0
            colors_uint16 = (np.clip(colors, 0.0, 1.0) * 65535).astype(np.uint16)
            las.red = colors_uint16[:, 0]
            las.green = colors_uint16[:, 1]
            las.blue = colors_uint16[:, 2]
        else:
            # Handle case with no color data if necessary, e.g., set to black or white
             las.red = np.zeros(len(points), dtype=np.uint16)
             las.green = np.zeros(len(points), dtype=np.uint16)
             las.blue = np.zeros(len(points), dtype=np.uint16)


        # 5. Write file
        las.write(output_las_file)
        logging.info(f"Successfully saved LAS file: {output_las_file}")
        return True

    except Exception as e:
        logging.error(f"Error saving LAS file {output_las_file}: {e}")
        return False

def visualize_point_cloud(points, colors, vis_config, block_name="Point Cloud Block"):
    """Visualizes the point cloud using Open3D with professional settings."""
    if points is None or colors is None:
        logging.error(f"Cannot visualize {block_name}: No point data.")
        return

    logging.info(f"Preparing Open3D visualization for {block_name}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors) # Expects colors in [0, 1]

    # Extract visualization settings from config
    point_size = vis_config.get('point_size', 1.0)
    bg_color = vis_config.get('background_color', (1.0, 1.0, 1.0)) # Default white
    screenshot_path = vis_config.get('screenshot_path', None) # This should be the specific path for this block
    show_ui = vis_config.get('show_ui', False) # Option to show Open3D's UI controls
    close_on_screenshot = vis_config.get('close_on_screenshot', True) # Close after screenshot?
    interactive_mode = vis_config.get('interactive', False) # Explicit flag for interactive window

    window_name = f"{vis_config.get('window_name_prefix', 'Block Vis:')} {block_name}"

    # --- Visualization Logic ---
    if screenshot_path:
        # Ensure output directory exists for screenshot
        screenshot_dir = os.path.dirname(screenshot_path)
        if screenshot_dir and not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir, exist_ok=True)
            logging.info(f"Created screenshot directory: {screenshot_dir}")

        vis = o3d.visualization.Visualizer()
        # Create window possibly hidden if only capturing and closing immediately
        vis.create_window(window_name=window_name, visible= (not close_on_screenshot) or show_ui)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray(bg_color)
        opt.point_size = point_size
        opt.light_on = True

        # Capture image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(screenshot_path, do_render=True)
        logging.info(f"Screenshot saved to {screenshot_path}")

        if close_on_screenshot:
            vis.destroy_window()
        else:
            # If not closing, make sure it's visible and run interactively
            vis.set_visible(True)
            logging.info(f"Displaying interactive window for {block_name}. Press 'Q' to close.")
            vis.run()
            vis.destroy_window()

    elif interactive_mode: # Interactive visualization without screenshot
        logging.info(f"Starting interactive visualization for {block_name}. Press 'Q' to close.")
        logging.info("Rotate (left mouse), zoom (scroll), pan (right mouse).")
        # Using draw_geometries for simplicity in interactive mode per block
        o3d.visualization.draw_geometries([pcd],
                                          window_name=window_name,
                                          point_show_normal=False,
                                          width=vis_config.get('window_width', 800),
                                          height=vis_config.get('window_height', 600))
        logging.info(f"Visualization finished for {block_name}.")

    else:
        logging.debug(f"Visualization skipped for {block_name} as neither interactive mode nor screenshot path was specified.")


# --- Main Processing Function ---
def process_point_cloud_blocks_from_config(config):
    """
    Processes individual NPZ point cloud blocks based on a configuration dictionary.
    Converts each block to a separate LAS file and optionally visualizes/screenshots each.

    Args:
        config (dict): A dictionary containing processing parameters.
            Required keys:
                'cache_dir' (str): Directory with NPZ block files.
                'output_las_dir' (str): Directory to save the output LAS files.
            Optional keys:
                'npz_pattern' (str): Glob pattern for NPZ files (default: '*_block_*.npz').
                'visualize' (bool): Master switch, set to True if any visualization needed (interactive or screenshot).
                'vis_config' (dict): Dictionary with visualization settings:
                    'interactive' (bool): Show an interactive Open3D window for each block (default: False).
                                          WARNING: Can open many windows.
                    'screenshot_dir' (str): Directory to save screenshots. If set, enables screenshotting.
                    'screenshot_format' (str): Format for screenshots, e.g., 'png', 'jpg' (default: 'png').
                    'point_size' (float): Default 1.0.
                    'background_color' (tuple): RGB tuple, e.g., (1.0, 1.0, 1.0) for white.
                    'window_name_prefix' (str): Prefix for visualization/screenshot window titles.
                    'show_ui' (bool): Show Open3D's visualization UI controls (default: False, relevant if not closing on screenshot).
                    'close_on_screenshot' (bool): Close window after saving screenshot (default: True).
                    'window_width', 'window_height' (int): Dimensions for interactive window.
    Returns:
        tuple: (number_of_files_processed, number_of_las_saved_successfully)
    """
    cache_dir = config.get('cache_dir')
    output_las_dir = config.get('output_las_dir')

    if not cache_dir or not output_las_dir:
        logging.error("Configuration must include 'cache_dir' and 'output_las_dir'.")
        return 0, 0

    if not os.path.isdir(cache_dir):
        logging.error(f"Cache directory not found: {cache_dir}")
        return 0, 0

    # Ensure output LAS directory exists
    if not os.path.exists(output_las_dir):
        os.makedirs(output_las_dir, exist_ok=True)
        logging.info(f"Created output LAS directory: {output_las_dir}")

    npz_pattern = config.get('npz_pattern', '*_block_*.npz')
    npz_file_pattern = os.path.join(cache_dir, npz_pattern)
    npz_files = sorted(glob.glob(npz_file_pattern)) # Sort for consistent processing order

    if not npz_files:
        logging.warning(f"No NPZ files found matching pattern '{npz_file_pattern}' in directory '{cache_dir}'.")
        return 0, 0

    logging.info(f"Found {len(npz_files)} NPZ files to process individually.")

    # Visualization settings
    should_visualize = config.get('visualize', False)
    vis_config = config.get('vis_config', {})
    interactive_vis = vis_config.get('interactive', False)
    screenshot_dir = vis_config.get('screenshot_dir', None)
    screenshot_format = vis_config.get('screenshot_format', 'png')

    if interactive_vis and len(npz_files) > 10: # Warn if potentially opening many windows
         logging.warning("Interactive visualization is enabled for potentially many blocks. This may open numerous windows.")
    if screenshot_dir:
        if not os.path.exists(screenshot_dir):
             os.makedirs(screenshot_dir, exist_ok=True)
             logging.info(f"Created screenshot directory: {screenshot_dir}")


    processed_count = 0
    saved_count = 0

    for npz_file in npz_files:
        processed_count += 1
        logging.info(f"--- Processing block file {processed_count}/{len(npz_files)}: {os.path.basename(npz_file)} ---")

        # Derive output filenames
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        output_las_file = os.path.join(output_las_dir, f"{base_name}.las")

        # Load data for the current block
        points, colors = load_npz_block_data(npz_file)

        if points is None:
            logging.warning(f"Skipping LAS saving and visualization for {base_name} due to loading issues.")
            continue

        # Save the individual block to LAS
        if save_to_las(points, colors, output_las_file):
            saved_count += 1

        # Visualize or screenshot if requested
        if should_visualize:
            current_vis_config = vis_config.copy() # Use a copy to modify screenshot path per block
            current_vis_config['interactive'] = interactive_vis # Pass interactive flag

            if screenshot_dir:
                screenshot_file = os.path.join(screenshot_dir, f"{base_name}.{screenshot_format}")
                current_vis_config['screenshot_path'] = screenshot_file # Set specific path for this block
            else:
                 current_vis_config['screenshot_path'] = None # Ensure it's None if no dir specified


            # Only call visualize if interactive mode is on OR a screenshot needs to be saved
            if interactive_vis or screenshot_dir:
                 visualize_point_cloud(points, colors, current_vis_config, block_name=base_name)
            else:
                 logging.debug(f"Visualization skipped for {base_name}: Neither interactive mode nor screenshot directory specified.")


    logging.info(f"--- Processing Summary ---")
    logging.info(f"Attempted to process: {processed_count} NPZ files.")
    logging.info(f"Successfully saved: {saved_count} LAS files.")
    return processed_count, saved_count


# --- Example Usage (if run as main script) ---
if __name__ == "__main__":
    # This demonstrates how you would call the function from another script
    # Replace with your actual paths and desired settings
    example_config = {
        'cache_dir': '/path/to/your/cache_dir', # REQUIRED: Change this
        'output_las_dir': '/path/to/output/las_blocks', # REQUIRED: Change this
        'npz_pattern': '*_block_*.npz', # Optional: Default is '*_block_*.npz'

        'visualize': True, # Master switch for visualization/screenshotting

        'vis_config': {
            # Option 1: Interactive visualization for each block
            # 'interactive': True,
            # 'point_size': 1.0,
            # 'background_color': (1.0, 1.0, 1.0), # White background
            # 'window_name_prefix': 'Block Vis:',

            # Option 2: Generate screenshot for each block
            'interactive': False, # Can be False if only screenshotting
            'screenshot_dir': '/path/to/output/block_screenshots', # REQUIRED for screenshots: Change this
            'screenshot_format': 'png', # Optional: 'png' or 'jpg'
            'point_size': 1.5,
            'background_color': (0.1, 0.1, 0.1), # Dark background example
            'close_on_screenshot': True, # Optional: Keep False to view after screenshot

            # General vis settings
            'window_name_prefix': 'Block Screenshot:', # Used for hidden window title during screenshot
        }
    }

    # Minimal config for just LAS conversion
    # example_config_minimal = {
    #     'cache_dir': '/path/to/your/cache_dir', # REQUIRED: Change this
    #     'output_las_dir': '/path/to/output/las_blocks', # REQUIRED: Change this
    # }

    # --- Run the processing ---
    print(f"Starting point cloud block processing with config...")
    total_processed, total_saved = process_point_cloud_blocks_from_config(example_config)

    print(f"Block processing finished. Processed {total_processed} files, successfully saved {total_saved} LAS files.")

