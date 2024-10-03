import os
import datetime
import numpy as np
import torch
import torch.utils.data
import logging
from pathlib import Path
from tqdm import tqdm

# Import your custom modules here
# import provider
# from datasets import LWBridgeDataset
# import models.pointnet2_sem_seg_msg as MODEL

# Constants
ROOT_DIR = '.'  # Adjust this to your project root directory
NUM_CLASSES = 5  # Adjust based on your dataset
NUM_POINT = 4096
BATCH_SIZE = 16


# Configuration
class Config:
    model = 'pointnet2_sem_seg_msg'
    batch_size = 16
    epoch = 128
    learning_rate = 0.001
    gpu = '0'
    optimizer = 'Adam'
    log_dir = None
    decay_rate = 1e-4
    npoint = 4096
    step_size = 10
    lr_decay = 0.7


config = Config()


# Setup logging
def setup_logging(log_dir):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{config.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# Create directories
def create_directories():
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    experiment_dir = Path(ROOT_DIR) / 'log' / 'sem_seg' / (config.log_dir or timestr)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    return experiment_dir, checkpoints_dir, log_dir


# Load datasets
def load_datasets(root):
    print("Loading training data...")
    TRAIN_DATASET = LWBridgeDataset(split='train', data_root=root, num_point=NUM_POINT,
                                    block_size=1.0, sample_rate=1.0, num_class=NUM_CLASSES)
    print("Loading test data...")
    TEST_DATASET = LWBridgeDataset(split='test', data_root=root, num_point=NUM_POINT,
                                   block_size=1.0, sample_rate=1.0, num_class=NUM_CLASSES)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    return TRAIN_DATASET, TEST_DATASET, trainDataLoader, testDataLoader


# Initialize models
def initialize_model(num_classes, experiment_dir):
    classifier = MODEL.get_model(num_classes).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir / 'checkpoints/best_model.pth'))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded pretrained models')
    except:
        print('No existing models, starting training from scratch...')
        start_epoch = 0
        classifier.apply(weights_init)

    return classifier, criterion, start_epoch


# Training function
def train(classifier, criterion, optimizer, trainDataLoader, logger):
    classifier.train()
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
        points, target = points.float().cuda(), target.long().cuda()
        points = points.transpose(2, 1)

        optimizer.zero_grad()
        seg_pred, trans_feat = classifier(points)
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
        target = target.view(-1, 1)[:, 0]

        loss = criterion(seg_pred, target, trans_feat)
        loss.backward()
        optimizer.step()

        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        correct = np.sum(pred_choice == target.cpu().data.numpy())
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss.item()

    return loss_sum / len(trainDataLoader), total_correct / float(total_seen)


# Evaluation function
def evaluate(classifier, criterion, testDataLoader, num_classes):
    classifier.eval()
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]
    total_iou_deno_class = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]

            loss = criterion(seg_pred, target, trans_feat)
            loss_sum += loss.item()
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)

            for l in range(num_classes):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
    return loss_sum / len(testDataLoader), total_correct / float(total_seen), mIoU


# Main training loop
def main():
    experiment_dir, checkpoints_dir, log_dir = create_directories()
    logger = setup_logging(log_dir)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # Load datasets
    root = os.path.join(ROOT_DIR, 'data/bridges_5cls_0.05_partition_10m_95ol_norm')
    TRAIN_DATASET, TEST_DATASET, trainDataLoader, testDataLoader = load_datasets(root)

    logger.info(f"Number of training data: {len(TRAIN_DATASET)}")
    logger.info(f"Number of test data: {len(TEST_DATASET)}")

    # Initialize models
    classifier, criterion, start_epoch = initialize_model(NUM_CLASSES, experiment_dir)

    # Optimizer
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=config.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=config.learning_rate, momentum=0.9)

    best_iou = 0

    # Training loop
    for epoch in range(start_epoch, config.epoch):
        logger.info(f'Epoch {epoch + 1}/{config.epoch}')

        lr = max(config.learning_rate * (config.lr_decay ** (epoch // config.step_size)), 1e-5)
        logger.info(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, train_acc = train(classifier, criterion, optimizer, trainDataLoader, logger)
        logger.info(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(checkpoints_dir / 'models.pth'))

        eval_loss, eval_acc, mIoU = evaluate(classifier, criterion, testDataLoader, NUM_CLASSES)
        logger.info(f'Eval - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, mIoU: {mIoU:.4f}')

        if mIoU >= best_iou:
            best_iou = mIoU
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mIoU': mIoU,
            }, str(checkpoints_dir / 'best_model.pth'))

        logger.info(f'Best mIoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()
