from DBtest.ImgTreatment import ImgTreatment
from DBtest.Move import Move

# it = ImgTreatment()
# for i in range(10):
#     srcDir = 'DBtest/img/data2/' + str(i)
#     targetDir = 'DBtest/img/train/' + str(i)
#     it.treatment(srcDir, targetDir, isDehaze=False, isRepair=False, isSave=True)


m = Move()
# m.copy('D:\\OneDrive - csu.edu.cn\\研究课题\\铁水图像视频\\part1', 'DBtest\\img\\data2')
for i in range(10):
    srcDir = 'DBtest/img/test/' + str(i) + '/'
    targetDir = 'DBtest/img/val/' + str(i) + '/'
    m.cut(srcDir, targetDir, proportion=0.2)
