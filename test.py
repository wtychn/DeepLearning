from PIL import Image
from DBtest.ImgTreatment import ImgTreatment

for i in range(0, 10):
    print('Treating No.' + str(i) + ' image directory')
    imgTreatment = ImgTreatment()
    imgTreatment.treatment('DBtest\\img\\data\\' + str(i), 'DBtest\\img\\train\\' + str(i))


