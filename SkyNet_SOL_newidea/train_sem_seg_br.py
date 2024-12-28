import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import provider
from data_utils.BridgeDataLoader import LWBridgeDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['abutment', 'girder', 'deck', 'parapet', 'noise']
#classes = [ 'noise', 'abutment', 'girder', 'deck', 'parapet'] #for CB
class2label = {cls: i for i, cls in enumerate(classes)}
# {'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3, 'noise': 4}
seg_classes = class2label
seg_label_to_cat = {} # {0: 'abutment', 1: 'girder', 2: 'deck', 3: 'parapet', 4: 'noise'}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

# pointnet_sem_seg, pointnet2_sem_seg, pointnet2_sem_seg_msg

#log_dir = 'PointNet2_0.05_SOL_a=200_10m_95ol_norm'
log_dir = 'test-cy-CB-sec'

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--models', type=str, default='pointnet2_sem_seg_msg', help='models name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=log_dir, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(os.path.join(ROOT_DIR, 'log'))
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('CB')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.models))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = os.path.join(ROOT_DIR, 'data/CB')
    NUM_CLASSES = len(classes)
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    deta_path='../data/CB/section'

    print("start loading training data ...")
    TRAIN_DATASET = LWBridgeDataset(split='train', data_root=deta_path, num_point=NUM_POINT, block_size=2.0, sample_rate=1.0, num_class=NUM_CLASSES, transform=None)
    print("start loading val data ...")
    TEST_DATASET = LWBridgeDataset(split='val', data_root=deta_path, num_point=NUM_POINT, block_size=2.0, sample_rate=1.0, num_class=NUM_CLASSES, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    # WCEL needs the amount vector to calculate the weights
    num_per_cls = torch.Tensor(TRAIN_DATASET.num_per_cls).cuda() # WCEL

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.models)
    shutil.copy(os.path.join(ROOT_DIR, 'models/%s.py' % args.models), str(experiment_dir))
    # shutil.copy('models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    # weights initialization
    def weights_init(m):
        classname = m.__class__.__name__
        # 'find(xxx)': return 0 if found and return -1 if not found
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            # make bias to 0
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrained models')
    except:
        log_string('No existing models, start training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # replace the original 'momentum' of BatchNorm1d or BatchNorm2d with the 'momentum' input by the function
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9)
        for i, data in pbar:

            points, target = data # target.shape: [16, 4096]

            #print(f"points: {points.shape}, target: {target.shape}")

            # for my loss function, strcture-oriented loss (SOL)
            points_raw = points.float().cuda() # output.shape: [16, 4096, 9]
            target_SOL = target.long().cuda() # output.shape: [16, 4096]


            points = points.data.numpy() # points.shape: [16, 4096, 9]
            points[: , : , : 3] = provider.rotate_point_cloud_z(points[: , : , : 3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()

            # adjust the shape of 'points' to suit the input_size of 'classifier'
            points = points.transpose(2, 1) # output.shape: [16, 9, 4096]

            optimizer.zero_grad()
            classifier = classifier.train()
            # seg_pred --> segmentation prediction, shape: [BATCH_SIZE, NUM_POINT, NUM_CLASSES] ([16, 4096, NUM_CLASSES])
            seg_pred, trans_feat = classifier(points)

            # for my loss function, strcture-oriented loss (SOL)
            seg_pred_SOL = seg_pred # output.shape: [16, 4096, NUM_CLASSES]

            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) # output.shape: ([16 * 4096, NUM_CLASSES])
            batch_label = target.view(-1, 1)[ : , 0].cpu().data.numpy()
            # target.view(-1, 1): [16 * 4096, 1], target.view(-1, 1)[ : , 0]: [16 * 4096]
            target = target.view(-1, 1)[ : , 0]

            # SOL
            loss = criterion(pred=seg_pred_SOL, target=target_SOL, points=points_raw, pred_previous=seg_pred, target_previous=target)

            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            # return of '.data.max': two tensors (values and indices)
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            # 更新统计
            train_loss.update(total_correct.item())
            train_acc.update(loss_sum.item())

            # 更新进度条
            pbar.set_postfix({
                'Val_Loss': f'{train_acc.avg:.4f}',
                'Val_Acc': f'{train_acc.avg * 100:.2f}%'
            })
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # save the results every epoch if the training data is very large, otherwise can be saved every 5 epoches
        if epoch % 1 == 0:
            logger.info('Save models...')
            savepath = str(checkpoints_dir) + '/models.pth'
            log_string('Saving at %s' % savepath)
            state = {'epoch': epoch, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}
            torch.save(state, savepath)
            log_string('Saving models....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)] # 'deno' --> denominator
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data # target.shape: [16, 4096]

                # for my loss function, strcture-oriented loss (SOL)
                points_raw = points.float().cuda() # output.shape: [16, 4096, 9]
                target_SOL = target.long().cuda() # output.shape: [16, 4096]

                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1) # output.shape: [16, 9, 4096]
                classifier = classifier.eval()
                # seg_pred --> segmentation prediction, shape: [BATCH_SIZE, NUM_POINT, NUM_CLASSES] ([16, 4096, NUM_CLASSES])
                seg_pred, trans_feat = classifier(points)

                # for my loss function, strcture-oriented loss (SOL)
                seg_pred_SOL = seg_pred # output.shape: [16, 4096, NUM_CLASSES]

                pred_val = seg_pred.contiguous().cpu().data.numpy() # pred_val.shape: [16, 4096, NUM_CLASSES]
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) # output.shape: [16 * 4096, NUM_CLASSES]
                batch_label = target.cpu().data.numpy() # batch_label.shape: [16, 4096]
                target = target.view(-1, 1)[: , 0] # output.shape: [16 * 4096]

                # SOL
                loss = criterion(pred=seg_pred_SOL, target=target_SOL, points=points_raw, pred_previous=seg_pred, target_previous=target)

                loss_sum += loss
                pred_val = np.argmax(pred_val, 2) # output.shape: [16, 4096]
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l)) # '&': AND
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l))) # '|': OR; 'deno' --> denominator
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            # mIoU: mean Intersection over Union
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                # seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])): justify each line
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save models...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {'epoch': epoch, 'class_avg_iou': mIoU, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}
                torch.save(state, savepath)
                log_string('Saving models....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    args = parse_args()
    main(args)