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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# denoised
# classes = ['abutment', 'girder', 'deck', 'parapet']
# class2label = {cls: i for i, cls in enumerate(classes)} # {'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3}
# seg_classes = class2label
# seg_label_to_cat = {} # {0: 'abutment', 1: 'girder', 2: 'deck', 3: 'parapet'}
# for i, cat in enumerate(seg_classes.keys()):
#     seg_label_to_cat[i] = cat

# class2color = {'abutment': [229, 158, 221], 'girder':[0, 11, 195], 'deck': [173, 219, 225], 'parapet': [230, 0, 0]}
# label2color = {classes.index(cls): class2color[cls] for cls in classes}

# with noise
classes = ['abutment', 'girder', 'deck', 'parapet', 'noise']
class2label = {cls: i for i, cls in enumerate(classes)} # {'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3, 'noise': 4}
seg_classes = class2label
seg_label_to_cat = {} # {0: 'abutment', 1: 'girder', 2: 'deck', 3: 'parapet', 4: 'noise'}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

class2color = {'abutment': [229, 158, 221], 'girder':[0, 11, 195], 'deck': [173, 219, 225], 'parapet': [230, 0, 0], 'noise': [0, 169, 58]}
label2color = {classes.index(cls): class2color[cls] for cls in classes}

trained_result = 'PointNet2_0.05_SOL_a=200_10m_95ol_norm'
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default=trained_result, help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='Whether visualize result [default: False]')
    parser.add_argument('--num_votes', type=int, default=5, help='Aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool
 
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
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
    BATCH_SIZE = args.batch_size # 32
    NUM_POINT = args.num_point # 4096

    root = 'data/bridges_5cls_0.05_partition_10m_95ol_norm'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', block_points=NUM_POINT, num_class=NUM_CLASSES)
    log_string("The number of test file is: %d" %  len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[: -4] for x in scene_id] # delete '.npy', like 'Area_5_conferenceRoom_1'
        num_batches = len(TEST_DATASET_WHOLE_SCENE) # every file in test folder is a batch

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        # every file in test folder is a batch
        for batch_idx in range(num_batches):
            print("visualize [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]

            # ATTENTION: 'whole'!
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx] # xyzrgb
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx] # label
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            # args.num_votes = 5, every file is processed 5 times and the answer appeared for most times will be taken
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                # 'TEST_DATASET_WHOLE_SCENE' returns data_bridge, label_bridge, sample_weight, index_bridge
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE # maybe something like sub_batch_num
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9)) # [32, 4096, 9]
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT)) # [32, 4096]
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT)) # [32, 4096]
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT)) # [32, 4096]
                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0: real_batch_size, ...] = scene_data[start_idx: end_idx, ...]
                    batch_label[0: real_batch_size, ...] = scene_label[start_idx: end_idx, ...]
                    batch_smpw[0: real_batch_size, ...] = scene_smpw[start_idx: end_idx, ...]
                    batch_point_index[0: real_batch_size, ...] = scene_point_index[start_idx: end_idx, ...]
                    batch_data[:, :, 3: 6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool,
                                               batch_point_index[0: real_batch_size, ...],
                                               batch_pred_label[0: real_batch_size, ...],
                                               batch_smpw[0: real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l)) # '&': AND
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l))) # '|': OR
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            if args.visual:
                cordinates_xyz = whole_scene_data[: , : 3]
                color_tmp = np.zeros((whole_scene_data.shape[0], 3))
                pred_label_tmp = pred_label.reshape(-1, 1)
                pred_all_info = np.concatenate([cordinates_xyz, color_tmp, pred_label_tmp], axis=1)
                # allocate color to each point according to prediction
                for row in pred_all_info:
                    label = row[-1]
                    row[3] = label2color[label][0] / 255.0
                    row[4] = label2color[label][1] / 255.0
                    row[5] = label2color[label][2] / 255.0
                # write labeled_color_array into a 'txt' file
                output_filepath = os.path.join(visual_dir, scene_id[batch_idx] + '_pred.txt')
                np.savetxt(fname=output_filepath, X=pred_all_info, fmt='%.7f %.7f %.7f %.7f %.7f %.7f %d')
                print(scene_id[batch_idx] + ' Labeled Color File SAVED!')

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.6f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)