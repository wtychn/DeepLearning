import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision import models
import json
from SE_RestNext import se_resnext_50


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        net = models.resnet50(pretrained=False)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.6 * heatmap + 0.4 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    path_img = '../DBtest/img/test/9/2018-03-26_18_44_09.jpg'
    json_path = './cam/labels.json'
    output_dir = '../DBtest/img/output'

    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}

    # 只取标签名
    # classes = list(classes.get(key) for key in range(1000))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)
    img_input = img_preprocess(img)

    # 加载预训练模型
    net = ResNet()
    path_checkpoint = '../DBtest/checkPoint/ResNet_checkPoint/ResNet_best_19.pth'   # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    net.eval()  # 8
    print(net)

    # 注册hook
    net.features.layer4[2].register_forward_hook(farward_hook)  # 9
    net.features.layer4[2].register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(idx))

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir)
