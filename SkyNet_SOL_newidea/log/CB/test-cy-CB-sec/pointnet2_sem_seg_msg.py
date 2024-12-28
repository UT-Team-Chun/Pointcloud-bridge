import  torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet_util import PointNetSetAbstractionMsg, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    # ours: structure-oriented loss (SOL) function
    def forward(self, pred, target, points, pred_previous, target_previous):
        '''
        pred: prediction results (probabilities of all classes), shape --> [16, 4096, NUM_CLASSES], need to 'argmax' before using
        target: labels of points,  shape --> [16, 4096]
        points: acquired by trainDataLoader, shape --> [16, 4096, 9], [: , : , : 3] are xyz coordinates
        pred_previous: to calculate CEL, [16 * 4096, NUM_CLASSES]
        target_previous: to calculate CEL, [16 * 4096]
        '''
        
        batch_size = points.shape[0]
        num_classes = pred.shape[-1]

        a = 200

        raw_xyz = points[: , : , : 3] # output.shape: [16, 4096, 3]
        raw_xyzl = torch.cat([raw_xyz, target.view(16, -1, 1)], dim=-1) # output.shape: [16, 4096, 4]

        pred_label = torch.argmax(pred, dim=-1, keepdim=True) # output.shape: [16, 4096, 1]
        pred_xyzl = torch.cat([raw_xyz, pred_label], dim=-1) # output.shape: [16, 4096, 4]

        total_loss = 0

        for i in range(batch_size):
            weights = torch.ones(num_classes).cuda() # an array

            batch_raw_xyzl = raw_xyzl[i] # output.shape: [4096, 4]
            batch_pred_xyzl = pred_xyzl[i] # output.shape: [4096, 4]

            GT_abutment = False
            GT_girder = False
            GT_deck = False
            GT_parapet = False

            Pr_abutment = False
            Pr_girder = False
            Pr_deck = False
            Pr_parapet = False

            # 0: 'abutment'
            index_abutment_raw = torch.where(batch_raw_xyzl[: , -1] == 0)[0]
            raw_xyzl_abutment = batch_raw_xyzl[index_abutment_raw, : ]
            if raw_xyzl_abutment.shape[0] != 0:
                GT_abutment = True
                center_abutment_z_raw = torch.mean(raw_xyzl_abutment[: , 2])
            index_abutment_pred = torch.where(batch_pred_xyzl[: , -1] == 0)[0]
            pred_xyzl_abutment = batch_pred_xyzl[index_abutment_pred, : ]
            if pred_xyzl_abutment.shape[0] != 0:
                Pr_abutment = True
                center_abutment_z_pred = torch.mean(pred_xyzl_abutment[: , 2])
            # 1: 'girder'
            index_girder_raw = torch.where(batch_raw_xyzl[: , -1] == 1)[0]
            raw_xyzl_girder = batch_raw_xyzl[index_girder_raw, : ]
            if raw_xyzl_girder.shape[0] != 0:
                GT_girder = True
                center_girder_z_raw = torch.mean(raw_xyzl_girder[: , 2])
            index_girder_pred = torch.where(batch_pred_xyzl[: , -1] == 1)[0]
            pred_xyzl_girder = batch_pred_xyzl[index_girder_pred, : ]
            if pred_xyzl_girder.shape[0] != 0:
                Pr_girder = True
                center_girder_z_pred = torch.mean(pred_xyzl_girder[: , 2])
            # 2: 'deck'
            index_deck_raw = torch.where(batch_raw_xyzl[: , -1] == 2)[0]
            raw_xyzl_deck = batch_raw_xyzl[index_deck_raw, : ]
            if raw_xyzl_deck.shape[0] != 0:
                GT_deck = True
                center_deck_z_raw = torch.mean(raw_xyzl_deck[: , 2])
            index_deck_pred = torch.where(batch_pred_xyzl[: , -1] == 2)[0]
            pred_xyzl_deck = batch_pred_xyzl[index_deck_pred, : ]
            if pred_xyzl_deck.shape[0] != 0:
                Pr_deck = True
                center_deck_z_pred = torch.mean(pred_xyzl_deck[: , 2])
            # 3: 'parapet'
            index_parapet_raw = torch.where(batch_raw_xyzl[: , -1] == 3)[0]
            raw_xyzl_parapet = batch_raw_xyzl[index_parapet_raw, : ]
            if raw_xyzl_parapet.shape[0] != 0:
                GT_parapet = True
                center_parapet_z_raw = torch.mean(raw_xyzl_parapet[: , 2])
            index_parapet_pred = torch.where(batch_pred_xyzl[: , -1] == 3)[0]
            pred_xyzl_parapet = batch_pred_xyzl[index_parapet_pred, : ]
            if pred_xyzl_parapet.shape[0] != 0:
                Pr_parapet = True
                center_parapet_z_pred = torch.mean(pred_xyzl_parapet[: , 2])

            # 'No' in GT but 'Yes' in prediction
            # (1). 0: 'abutment'
            if ((not GT_abutment) and (Pr_abutment)):
                weights[0] += a
            # (2). 1: 'girder'
            if ((not GT_girder) and (Pr_girder)):
                weights[1] += a
            # (3). 2: 'deck'
            if ((not GT_deck) and (Pr_deck)):
                weights[2] += a
            # (4). 3: 'parapet'
            if ((not GT_parapet) and (Pr_parapet)):
                weights[3] += a

            # 1st level 
            # (1). 0: 'abutment' and 1: 'girder'
            if GT_abutment and Pr_abutment and GT_girder and Pr_girder and ((center_abutment_z_raw < center_girder_z_raw) and (center_abutment_z_pred >= center_girder_z_pred)):
                weights[0] += a
                weights[1] += a
            # (2). 1: 'girder' and 2: 'deck'
            if GT_girder and Pr_girder and GT_deck and Pr_deck and ((center_girder_z_raw < center_deck_z_raw) and (center_girder_z_pred >= center_deck_z_pred)):
                weights[1] += a
                weights[2] += a
            # (3). 2: 'deck' and 3: 'parapet'
            if GT_deck and Pr_deck and GT_parapet and Pr_parapet and ((center_deck_z_raw < center_parapet_z_raw) and (center_deck_z_pred >= center_parapet_z_pred)):
                weights[2] += a
                weights[3] += a
            
            # 2nd level
            # (1). 0: 'abutment' and 2: 'deck'
            if GT_abutment and Pr_abutment and GT_deck and Pr_deck and ((center_abutment_z_raw < center_deck_z_raw) and (center_abutment_z_pred >= center_deck_z_pred)):
                weights[0] += 2 * a
                weights[2] += 2 * a
            # (2). 1: 'girder' and 3: 'parapet'
            if GT_girder and Pr_girder and GT_parapet and Pr_parapet and ((center_girder_z_raw < center_parapet_z_raw) and (center_girder_z_pred >= center_parapet_z_pred)):
                weights[1] += 2 * a
                weights[3] += 2 * a
            
            # 3rd level
            # (1). 0: 'abutment' and 3: 'parapet'
            if GT_abutment and Pr_abutment and GT_parapet and Pr_parapet and ((center_abutment_z_raw < center_parapet_z_raw) and (center_abutment_z_pred >= center_parapet_z_pred)):
                weights[0] += 3 * a
                weights[3] += 3 * a

            loss = F.cross_entropy(pred_previous, target_previous, weight=weights)

            total_loss += loss
        
        return total_loss




if __name__ == '__main__':

    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))