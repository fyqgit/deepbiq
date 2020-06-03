import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import sys
import os

from PIL import Image
from sklearn.externals import joblib
from comm_model import *
from torch.autograd import Variable

#usage: python test_dataset.py 图片文件夹路径 输出保存到的文本路径


svr_save_path = './svr_mode.pkl'
feature_mode_path = '../trained_models/model_best.pth.tar'


sys.stdout = open(sys.argv[2], mode = 'w',encoding='utf-8')


def main():
    #load feature_model
    model = FeatureMode(feature_mode_path)

    #load img
    normalize = get_imagenet_normalize()
    trans1 = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    for filepath, dirnames, filenames in os.walk(sys.argv[1]):
        for filename in filenames:
            # 判断图片格式
            if filename[-4:] == '.png' or filename[-4:] == '.jpg' or filename[-4:] == '.bmp' or filename[-4:] == '.PNG' or filename[-4:] == '.JPG' or filename[-4:] == '.BMP':
                file = os.path.join(filepath, filename)
                print(file + '\t')

                img = Image.open(file)
                input = trans1(img)  # var input is a tensor
                # input = np.expand_dims(input, 0)
                input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)

                # switch to evaluate mode
                #end = time.time()
                output = model.extract_feature(input)
                #test_time = time.time() - end
                #print('Time : %.3f sec\n' % test_time)

                # load SVR model
                clf = joblib.load(svr_save_path)
                pred_y = clf.predict(output)
                print(pred_y)


if __name__ == "__main__":
    main()

