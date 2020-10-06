import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np
from parse_config import ConfigParser

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
def mix_features(low_level, high_level):
    '''
    low_level features有更高的分辨率和细节信息 [128,28,28]
    high_level features 有较低的分辨了和语义信息[256,14,14]
    '''
    low_channel = low_level.shape[1]
    low_size = low_level.shape[2]
    high_channel = high_level.shape[1]
    high_size = high_level.shape[2]

    high_level_resize = F.interpolate(high_level, scale_factor = low_size/high_size, mode = "bilinear")

    mix_feature = torch.cat([low_level, high_level_resize], dim=1)
    return mix_feature

def nms(scores_np, proposalN, iou_threshs, coordinates):
    '''
    scores_np: (241, 1)
    proposal_N: int, [2, 3, 2]其中之一
    iou_threshs: int ,0.25
    coordinates: 对应ratio的滑动窗口们的坐标（241个滑动窗口）
    '''
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0] # 窗口数目
    # 将分数和坐标
    indices_coordinates = np.concatenate((scores_np, coordinates), 1) # coordinates(241, 4)， scores_np(241,1)--> (241,5)

    indices = np.argsort(indices_coordinates[:, 0]) # 从小到大排序的索引
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices] 
    # 将分数-坐标-index拼接起来
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1] # 从最后一个开始，因为最后一个
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1] # 取出除最后一个之前的

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        # 去掉和indice_coordinates iou大于阈值的box
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self, ratio, stride, input_size):
        super(APPM, self).__init__()
        # self.concats = concats
        self.ratio = ratio # 滑动窗口的size，例如[[14,14],[7,7]]
        # self.proposeN = proposeN # 注意力窗口的数目
        self.stride = stride
        self.input_size = input_size
        self.avgpools = [nn.Avgpool2d(self.ratio[i], 1) for i in range(len(self.ratio))] # size of propose window
        # self.window_nums = compute_window_nums
    
    def forward(self, x, proposeN, N_list, iou_thres, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(self.ratio))]
        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(self.ratio))]
        # add channels to activity map
        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(self.ratio))], dim=1)
        # concat diffierent slide windows activity map
        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(self.ratio))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy() # (10, 591, 1)
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1) # torch.Size([10, 591])

        #nms
        window_nums = self.compute_window_nums(self.stride, self.input_size, self.ratio)
        coordinates = self.concats(window_nums, self.stride, self.input_size, self.ratio)
        window_nums_sum = [0, sum(window_nums[:1])]

        proposeN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                coordinates = coordinates[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
                # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse
        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)
        # ipdb.set_trace()
        return proposalN_indices, proposalN_windows_scores, window_scores, coordinates

    def concats(self, window_nums, stride, input_size, ratios):
        """
        coordinates_cat, 
        """

        indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
        # 
        coordinates = [self.indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)]
        coordinates_cat = np.concatenate(coordinates, 0)

        return coordinates_cat
    def compute_window_nums(self,stride,input_size,ratios):
        # 滑动窗口的数目
        size = input_size / stride
        window_nums = []

        for _, ratio in enumerate(ratios):
            window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))
        return window_nums

    def indices2coordinates(self, indices, stride, image_size, ratio):
        # indics to coordinates
        batch, _ = indices.shape
        coordinates = []

        for j, indice in enumerate(indices):
            coordinates.append(self.computeCoordinate(image_size, stride, indice, ratio))

        coordinates = np.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
        return coordinates

    def computeCoordinate(self, image_size, stride, indice, ratio):
        size = int(image_size / stride) # 768 / 32
        # ipdb.set_trace()
        column_window_num = (size - ratio[1]) + 1
        x_indice = indice // column_window_num
        y_indice = indice % column_window_num
        x_lefttop = x_indice * stride - 1
        y_lefttop = y_indice * stride - 1
        x_rightlow = x_lefttop + ratio[0] * stride
        y_rightlow = y_lefttop + ratio[1] * stride
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
            x_rightlow +=1
        if y_lefttop < 0:
            y_lefttop = 0
            y_rightlow +=1
        coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)

        return coordinate

class MainNet(BaseModel):
    def __init__(self, proposeN, num_classes, base_model, ratio, stride, input_size, config):
        super(MainNet, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.ratio = ratio
        self.batch_size = config['batch_size']
        self.proposeN = proposeN
        self.pretrained_model = base_model # 可以选择冻结参数的部分
        self.rawclas_net = nn.Linear(channels, self.num_classes)
        self.finecls_net = nn.Linear(channels, 100)
        self.stride = stride
        self.input_size = input_size
        self.APPM = APPM(ratio = self.ratio, self.stride, self.input_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, status='test', DEVICE='cuda'):
        """
        status: test推理阶段
                raw 粗分类训练阶段
                fine 细分类训练阶段
        """
        # if status is "raw":
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048
        raw_logits = self.rawcls_net(embedding)
        if status is "raw":
            return raw_logits
        fm = mix_features(conv5_b, fm)
        # 
        proposalN_indices, proposalN_windows_scores, window_scores, coordinates = self.APPM(fm, self.proposeN, self.config["N_list"], self.config["iou_thres"], DEVICE='cuda')
        window_imgs = torch.zeros([self.batch_size, self.proposalN, 3, 448, 448]).to(DEVICE)  # [N, 4, 3, 224, 224]
        for i in range(batch_size):
            for j in range(proposalN):
                [x0, y0, x1, y1] = coordinates[proposalN_indices2[i, j]]
                # ipdb.set_trace()
                window_imgs[i:i+1, j] = x[i:i + 1, :, x0:(x1), y0:(y1)]
        window_imgs = window_imgs.reshape(batch_size * proposalN2, 3, 448, 448)
        local_fm, local_embeddings, _ = self.pretrained_model(window_imgs.detach())
        local_class = self.finecls_net(local_embeddings)
        if status is "fine": # 细粒度分类根据注意力区域进行加权
            






    
if __name__ == "__main__":
    
    model = MainNet()
    