import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import sys
import cv2

from PIL import Image
from sklearn.externals import joblib
from comm_model import *
from torch.autograd import Variable

svr_save_path = './svr_mode.pkl'
feature_mode_path = '../trained_models/model_best.pth.tar'

def main():

    #load feature_model, comm_model.py
    model = FeatureMode(feature_mode_path)

    #conv img to tensor
    #normalize = get_imagenet_normalize()
    trans1 = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            #normalize,
        ])

    img = Image.open(sys.argv[1])
    #img = img.resize((640,480))

    # var input is a tensor
    input = trans1(img)
    #print(input)
    input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)

    #switch to evaluate mode，extract_feature in comm_model.py FeatureMode
    end = time.time()
    output = model.extract_feature(input)
    test_time = time.time() - end
    print('Time : %.3f sec\n' % test_time)

    # load SVR model
    clf = joblib.load(svr_save_path)
    pred_y = clf.predict(output)
    print(pred_y)
    print(torch.__version__)


if __name__ == "__main__":
    main()


