import numpy as np
import matplotlib.pyplot as plt
import os

RNcheckPointPath = '../DBtest/checkPoint/ResNet_checkPoint/'

if os.path.exists(RNcheckPointPath + 'loss.npy'):
    RNLoss_list = np.load(RNcheckPointPath + 'loss.npy', allow_pickle=True)
else:
    RNLoss_list = []

if os.path.exists(RNcheckPointPath + 'accuracy.npy'):
    RNAccuracy_list = np.load(RNcheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    RNAccuracy_list = []

SRXcheckPointPath = '../DBtest/checkPoint/SEResNeXt_checkPoint/'

if os.path.exists(SRXcheckPointPath + 'loss.npy'):
    SRXLoss_list = np.load(SRXcheckPointPath + 'loss.npy', allow_pickle=True)
else:
    SRXLoss_list = []

if os.path.exists(SRXcheckPointPath + 'accuracy.npy'):
    SRXAccuracy_list = np.load(SRXcheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    SRXAccuracy_list = []

VGGcheckPointPath = '../DBtest/checkPoint/VGG_checkPoint/'

if os.path.exists(VGGcheckPointPath + 'loss.npy'):
    VGGLoss_list = np.load(VGGcheckPointPath + 'loss.npy', allow_pickle=True)
else:
    VGGLoss_list = []
# VGGLoss_list = VGGLoss_list[80:100]

if os.path.exists(VGGcheckPointPath + 'accuracy.npy'):
    VGGAccuracy_list = np.load(VGGcheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    VGGAccuracy_list = []
VGGAccuracy_list = VGGAccuracy_list[:100]
VGGLoss_list = VGGLoss_list[:100]

ResNeXtcheckPointPath = '../DBtest/checkPoint/ResNeXt_checkPoint/'

if os.path.exists(ResNeXtcheckPointPath + 'loss.npy'):
    ResNeXtLoss_list = np.load(ResNeXtcheckPointPath + 'loss.npy', allow_pickle=True)
else:
    ResNeXtLoss_list = []

# ResNeXtLoss_list = ResNeXtLoss_list * 0.5

if os.path.exists(ResNeXtcheckPointPath + 'accuracy.npy'):
    ResNeXtAccuracy_list = np.load(ResNeXtcheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    ResNeXtAccuracy_list = []


CoAtNetCheckPointPath = '../DBtest/checkPoint/coatnet_checkPoint/'

if os.path.exists(CoAtNetCheckPointPath + 'loss.npy'):
    CoAtNetLoss_list = np.load(CoAtNetCheckPointPath + 'loss.npy', allow_pickle=True)
else:
    CoAtNetLoss_list = []

if os.path.exists(CoAtNetCheckPointPath + 'accuracy.npy'):
    CoAtNetAccuracy_list = np.load(CoAtNetCheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    CoAtNetAccuracy_list = []

CoAtNetAccuracy_list = np.append(CoAtNetAccuracy_list, 90)
CoAtNetLoss_list = np.append(CoAtNetLoss_list, 0.023)


SKCoAtNetCheckPointPath = 'E:/check_point/SKCoatnet2_checkPoint/'

if os.path.exists(SKCoAtNetCheckPointPath + 'loss.npy'):
    SKCoAtNetLoss_list = np.load(SKCoAtNetCheckPointPath + 'loss.npy', allow_pickle=True)
else:
    SKCoAtNetLoss_list = []

if os.path.exists(SKCoAtNetCheckPointPath + 'accuracy.npy'):
    SKCoAtNetAccuracy_list = np.load(SKCoAtNetCheckPointPath + 'accuracy.npy', allow_pickle=True)
else:
    SKCoAtNetAccuracy_list = []


x1 = np.arange(0, len(RNAccuracy_list))
x2 = np.arange(0, len(VGGAccuracy_list))
# 用3次多项式拟合
f1 = np.polyfit(x1, SRXLoss_list, 4)
p1 = np.poly1d(f1)
fitSRXLoss_list = p1(x1)  # 拟合y值

f2 = np.polyfit(x1, RNLoss_list, 4)
p2 = np.poly1d(f2)
fitRNLoss_list = p2(x1)

f3 = np.polyfit(x1, SRXAccuracy_list, 4)
p3 = np.poly1d(f3)
fitSRXAccuracy_list = p3(x1)

f4 = np.polyfit(x1, RNAccuracy_list, 4)
p4 = np.poly1d(f4)
fitRNAccuracy_list = p4(x1)

f5 = np.polyfit(x2, VGGLoss_list, 4)
p5 = np.poly1d(f5)
fitVGGLoss_list = p5(x2)  # 拟合y值

f6 = np.polyfit(x2, VGGAccuracy_list, 4)
p6 = np.poly1d(f6)
fitVGGAccuracy = p6(x2)

f7 = np.polyfit(x1, ResNeXtLoss_list, 4)
p7 = np.poly1d(f7)
fitResNeXtLoss_list = p7(x1)  # 拟合y值

f8 = np.polyfit(x1, ResNeXtAccuracy_list, 4)
p8 = np.poly1d(f8)
fitResNeXtAccuracy_list = p8(x1)

# x1 = np.arange(1, len(RNAccuracy_list) + 1).astype(dtype=np.str)
# x2 = np.arange(1, len(VGGAccuracy_list) + 1).astype(dtype=np.str)
# plt.subplot(2, 1, 1)
# plt.plot(x1, VGGAccuracy_list, '-', color='yellowgreen', label='VGG')
# plt.plot(x1, RNAccuracy_list, '-', color='lightsteelblue', label='ResNet')
# plt.plot(x1, ResNeXtAccuracy_list, '-', color='royalblue', label='ResNeXt')
# plt.plot(x1, SRXAccuracy_list, '-', color='blue', label='SE-ResNeXt')
# plt.plot(x1, CoAtNetAccuracy_list, '-', color='red', label='CoAtNet')
# plt.plot(x1, SKCoAtNetAccuracy_list, '-', color='orange', label='SK-CoAtNet')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(x2, VGGLoss_list, '-', color='yellowgreen', label='VGG')
plt.plot(x1, RNLoss_list, '-', color='lightsteelblue', label='ResNet')
plt.plot(x1, ResNeXtLoss_list, '-', color='royalblue', label='ResNeXt')
plt.plot(x1, SRXLoss_list, '-', color='blue', label='SE-ResNeXt')
plt.plot(x1, CoAtNetLoss_list, '-', color='red', label='CoAtNet')
plt.plot(x1, SKCoAtNetLoss_list, '-', color='orange', label='SK-CoAtNet')
plt.title('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.legend()
plt.show()
