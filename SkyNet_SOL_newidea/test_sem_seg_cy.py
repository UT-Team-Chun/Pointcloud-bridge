import argparse
import os
from data_utils.BridgeDataLoader import ScannetDatasetWholeScene
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import laspy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# with noise
classes = ['abutment', 'girder', 'deck', 'parapet', 'noise']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}

class2color = {
    'abutment': [229, 158, 221],
    'girder': [0, 11, 195],
    'deck': [173, 219, 225],
    'parapet': [230, 0, 0],
    'noise': [0, 169, 58]
}
label2color = {classes.index(cls): class2color[cls] for cls in classes}

trained_result = 'best-lin'

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=trained_result, help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='Whether visualize result')
    parser.add_argument('--num_votes', type=int, default=5, help='Aggregate segmentation scores with voting')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def save_to_las(xyz, rgb, labels, output_path):
    """
    Save point cloud data to LAS format
    Args:
        xyz: point coordinates (N, 3)
        rgb: color values (N, 3)
        labels: predicted labels (N,)
        output_path: path to save the las file
    """
    header = laspy.LasHeader(point_format=3, version="1.3")
    las = laspy.LasData(header)
    
    # Set coordinates
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    # Set colors (need to convert to uint16)
    las.red = (rgb[:, 0] * 65535).astype(np.uint16)
    las.green = (rgb[:, 1] * 65535).astype(np.uint16)
    las.blue = (rgb[:, 2] * 65535).astype(np.uint16)
    
    # Store classification in classification field
    las.classification = labels.astype(np.uint8)
    
    # Save to file
    las.write(output_path)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = Path(experiment_dir) / 'visual'
    visual_dir.mkdir(exist_ok=True)

    # Logger setup
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = len(classes)
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # Load dataset
    root = 'data/bridges_5cls_0.05_partition_10m_95ol_norm'
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', 
                                                       block_points=NUM_POINT, 
                                                       num_class=NUM_CLASSES)
    log_string(f"The number of test file is: {len(TEST_DATASET_WHOLE_SCENE)}")

    # Load model
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    with torch.no_grad():
        scene_id = [x[:-4] for x in TEST_DATASET_WHOLE_SCENE.file_list]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print(f"Processing [{batch_idx + 1}/{num_batches}] {scene_id[batch_idx]} ...")
            
            # Initialize metrics
            total_seen_class_tmp = [0] * NUM_CLASSES
            total_correct_class_tmp = [0] * NUM_CLASSES
            total_iou_deno_class_tmp = [0] * NUM_CLASSES

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))

            # Voting
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    
                    torch_data = torch.FloatTensor(batch_data).cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, 
                                             batch_point_index[0:real_batch_size, ...],
                                             batch_pred_label[0:real_batch_size, ...],
                                             batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            # Calculate metrics
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] = np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] = np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] = np.sum((pred_label == l) | (whole_scene_label == l))
                
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            # Calculate and log IoU
            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string(f'Mean IoU of {scene_id[batch_idx]}: {tmp_iou:.4f}')

            if args.visual:
                # Prepare data for saving
                xyz = whole_scene_data[:, :3]
                rgb = np.zeros((whole_scene_data.shape[0], 3))
                
                # Convert labels to colors
                for i, label in enumerate(pred_label):
                    rgb[i] = np.array(label2color[label]) / 255.0
                
                # Save as LAS file
                output_filepath = visual_dir / f"{scene_id[batch_idx]}_pred.las"
                save_to_las(xyz, rgb, pred_label, str(output_filepath))
                print(f"{scene_id[batch_idx]} LAS file saved!")

        # Calculate and log final IoU for all classes
        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += f'class {seg_label_to_cat[l]:<14}, IoU: {IoU[l]:.6f}\n'
        log_string(iou_per_class_str)
        log_string(f'Mean IoU: {np.mean(IoU):.6f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
