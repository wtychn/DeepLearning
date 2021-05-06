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
# VGGAccuracy_list = VGGAccuracy_list[80:100]

x1 = range(0, len(RNAccuracy_list))
x2 = range(0, len(VGGAccuracy_list))
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

# plt.subplot(2, 1, 1)
# plt.plot(x1, fitVGGAccuracy_list, '.-', color='yellowgreen', label='VGG')
plt.plot(x1, fitRNAccuracy_list, '-', color='lightsteelblue', label='ResNet')
plt.plot(x1, fitSRXAccuracy_list, '-', color='royalblue', label='SE-ResNeXt')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(x2, fitVGGLoss_list, '-', color='yellowgreen', label='VGG')
# plt.plot(x1, fitRNLoss_list, '-', color='lightsteelblue', label='ResNet')
# plt.plot(x1, fitSRXLoss_list, '-', color='royalblue', label='SE-ResNeXt')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
plt.legend()
plt.show()
