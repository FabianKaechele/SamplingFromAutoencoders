import numpy as np
import ot
import math
from scipy import linalg
from tqdm.notebook import tqdm
import torch
from torch import nn
import torchvision.models as models
import torchvision.datasets as dset
import torch.nn.functional as F
import torchvision.transforms as transforms


def compute_score(real, fake, k=1, sigma=1, sqrt=True):
    """
    Compute wasserstein, MMD, 1-Nearest Neighbor classifier
    """
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    Mxx = distance(real, real, device, False)
    Mxy = distance(real, fake, device, False)
    Myy = distance(fake, fake, device, False)

#    s = Score()
#    s.emd = wasserstein(Mxy, sqrt)
#    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
#    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)
    s = np.zeros(3)
    s[0] = wasserstein(Mxy, sqrt)
    s[1] = mmd(Mxx, Mxy, Myy, sigma)
    s[2] = knn(Mxx, Mxy, Myy, k, False).acc

    return s

def distance(X, Y, device, sqrt=False):
    #device = "cpu"
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1).to(device)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1).to(device)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    #mmd = Mxx.mean() + Myy.mean() - 2 * Mxy.mean()
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd

def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd
    
def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))) #[1,1,1....,0,0,0...]
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False) 
    #topk: returns indices of the k smallest elements along dim 0

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    # count[i] = label[idx[0][i]]
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()
    # if count > 0.5, then 1; otherwise 0

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_t = s.tp / (s.tp + s.fn)
    s.acc_f = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean() #if label == pred, then 1
    s.k = k

    return s


class Score:
    emd = 0
    mmd = 0
    knn = None
    def printScore(self):
        print("-------------------------")
        print("EMD:",self.emd)
        print("MMD:",self.mmd)
        print("1NN Precision:",float(self.knn.precision))
        print("1NN Recall:",float(self.knn.recall))
        print("1NN Acc_t:",float(self.knn.acc_t))
        print("1NN Acc_f:",float(self.knn.acc_f))
        print("-------------------------")


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0
    
def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2*m_w.dot(m) + np.trace(C + C_w - 2*C_C_w_sqrt)
    return np.sqrt(score)

eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score

def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score

class ConvNetFeatureSaver(object):
    def __init__(self, workers=2, batch_size=64):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.batch_size = batch_size
        self.workers = workers
        #resnet = getattr(models, model)(pretrained=True)
        resnet = torch.load('./resnet34.pth')
        self.resnet = resnet
        self.resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                       resnet.relu,
                                       resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3,
                                       resnet.layer4).to(self.device).eval()
        
    def extract(self, img):
        """
        Input:  img
        Output: features of the input img extrated by ResNet34
        """
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []
        with torch.no_grad():
            input_ = img.to(self.device)
            fconv = self.resnet_feature(input_).mean(3).mean(2).squeeze()
            flogit = self.resnet.fc(fconv)
            fsmax = F.softmax(flogit)
            
            feature_pixl.append(img)
            feature_conv.append(fconv.data.cpu())
            feature_logit.append(flogit.data.cpu())
            feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0)
        feature_conv = torch.cat(feature_conv, 0)
        feature_logit = torch.cat(feature_logit, 0)
        feature_smax = torch.cat(feature_smax, 0)

        return feature_pixl, feature_conv, feature_logit, feature_smax
