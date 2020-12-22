# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import gc
import random
import argparse
import time
import scipy.sparse as sp
from ToolScripts.utils import loadData
from ToolScripts.utils import load
from ToolScripts.utils import sparse_mx_to_torch_sparse_tensor
from ToolScripts.utils import mkdir
from ToolScripts.utils import normalize_adj

from MyGCN import MODEL
from BPRData import BPRData
import torch.utils.data as dataloader
import evaluate


device_gpu = t.device("cuda")
modelUTCStr = str(int(time.time()))[4:]

isLoadModel = False
LOAD_MODEL_PATH = ""

class Model():

    def __init__(self, args, isLoad=False):
        self.args = args
        self.datasetDir = os.path.join(os.getcwd(), "dataset", args.dataset)

        trainMat, testData, validData = self.getData(args)
        self.userNum, self.itemNum = trainMat.shape
        self.trainMat = (trainMat != 0)*1
        
        a = sp.csr_matrix((self.userNum, self.userNum))
        b = sp.csr_matrix((self.itemNum, self.itemNum))
        uv_adj = sp.vstack([sp.hstack([a, self.trainMat]), sp.hstack([self.trainMat.T, b])])
        uv_adj = (uv_adj != 0) * 1
        uv_adj = (uv_adj + sp.eye(uv_adj.shape[0])) *1 
        self.uv_adj = normalize_adj(uv_adj) 
        self.adj_sp_tensor = sparse_mx_to_torch_sparse_tensor(self.uv_adj).cuda()

        #train data
        train_u, train_v = self.trainMat.nonzero()
        assert np.sum(self.trainMat.data ==0) == 0
        log("train data size = %d"%(train_u.size))
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, self.args.num_ng, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0)
        #valid data
        valid_dataset = BPRData(validData, self.itemNum, self.trainMat, 0, False)
        self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)
        
        #test_data
        test_dataset = BPRData(testData, self.itemNum, self.trainMat, 0, False)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)

        self.lr = self.args.lr #0.001
        self.curEpoch = 0
        self.isLoadModel = isLoad
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg  = []
        gc.collect()
        log("gc.collect()")

    def setRandomSeed(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
    
    def getData(self, args):
        data = loadData(args.dataset)
        trainMat, testData, validData = data
        return trainMat, testData, validData

    #初始化参数
    def prepareModel(self):
        self.modelName = self.getModelName() 
        self.setRandomSeed()
        self.layer = eval(self.args.layer)
        self.hide_dim = args.hide_dim
        self.out_dim = sum(self.layer) + self.hide_dim
        self.model = MODEL(self.args, self.userNum, self.itemNum, self.hide_dim, self.layer).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay=0)

    def adjust_learning_rate(self, opt, epoch):
        for param_group in opt.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
            # log("cur lr = %.6f"%(param_group['lr']))
    
    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u,i), dim=1)
        pred_j = t.sum(t.mul(u,j), dim=1)
        return pred_i, pred_j

    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        if self.isLoadModel == True:
            self.loadModel(LOAD_MODEL_PATH)
            HR, NDCG = self.validModel(self.test_loader)
            log("HR = %.4f, NDCG = %.4f"%(np.mean(HR), np.mean(NDCG)))
        cvWait = 0
        best_HR = 0.1
        for e in range(self.curEpoch, self.args.epochs+1):
            #记录当前epoch,用于保存Model
            self.curEpoch = e
            log("**************************************************************")
            #训练
            epoch_loss = self.trainModel()
            self.train_loss.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"% (e, self.args.epochs, epoch_loss))
            
            HR, NDCG = self.validModel(self.valid_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)
            log("epoch %d/%d, valid HR = %.4f, valid NDCG = %.4f"%(e, self.args.epochs, HR, NDCG))
            # if e%10 == 0 and e != 0:
            # log(self.getModelName())
            # HR, NDCG = self.test()

            self.adjust_learning_rate(self.opt, e)
            if HR > best_HR:
                best_HR = HR
                cvWait = 0
                best_epoch = self.curEpoch
                self.saveModel()
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            self.saveHistory()

            if cvWait == self.args.patience:
                log('Early stopping! best epoch = %d'%(best_epoch))
                self.loadModel(self.modelName)
                HR, NDCG = self.validModel(self.test_loader)
                log("epoch %d/%d, test HR = %.4f, test NDCG = %.4f"%(e, self.args.epochs, HR, NDCG))
                break
        
        
    def test(self):
        #load test dataset
        HR, NDCG = self.validModel(self.test_loader)
        log("test HR = %.4f, test NDCG = %.4f"%(HR, NDCG))
        log("model name : %s"%(self.modelName))
    

    def trainModel(self):
        train_loader = self.train_loader
        log("start negative sample...")
        train_loader.dataset.ng_sample()
        log("finish negative sample...")
        epoch_loss = 0
        for user, item_i, item_j in train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()
            user_embed, item_embed = self.model(self.adj_sp_tensor)
            
            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]
            
            pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5*(bprloss + self.args.reg * regLoss)/self.args.batch
            epoch_loss += bprloss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('setp %d/%d, step_loss = %f'%(i, loss.item()), save=False, oneline=True)
        log("finish train")
        return epoch_loss

        
    def validModel(self, data_loader, save=False):
        HR, NDCG = [], []
        user_embed, item_embed = self.model(self.adj_sp_tensor)
        for user, item_i in data_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            
            userEmbed = user_embed[user]
            testItemEmbed = item_embed[item_i]
            pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)

            batch = int(user.cpu().numpy().size/101)
            assert user.cpu().numpy().size % 101 ==0
            for i in range(batch):
                batch_scores = pred_i[i*101: (i+1)*101].view(-1)
                _, indices = t.topk(batch_scores, self.args.top_k)
                tmp_item_i = item_i[i*101: (i+1)*101]
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                gt_item = tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
        return np.mean(HR), np.mean(NDCG)


    def getModelName(self):
        title = "NGCF_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr + \
        "_reg_" + str(self.args.reg)+ \
        "_batch_" + str(self.args.batch) + \
        "_lr_" + str(self.args.lr) + \
        "_decay_" + str(self.args.decay) + \
        "_hide_" + str(self.args.hide_dim) + \
        "_Layer_" + self.args.layer +\
        "_slope_" + str(self.args.slope) +\
        "_top_" + str(self.args.top_k)
        return ModelName


    def saveHistory(self):
        #保存历史数据，用于画图
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        # ModelName = self.getModelName()
        ModelName = self.modelName
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + self.args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'lr': self.lr,
            'model': self.model,
            'reg':self.args.reg,
            'history':history,
            }
        t.save(params, savePath)


    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + args.dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.lr = checkpoint['lr']
        self.model = checkpoint['model']
        self.args.reg = checkpoint['reg']
        #恢复history
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NGCF main.py')
    #dataset params
    parser.add_argument('--dataset', type=str, default="Yelp", help="Yelp")
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--hide_dim', type=int, default=4)
    parser.add_argument('--layer', type=str, default="[4]")
    parser.add_argument('--slope', type=float, default=0.2)

    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=180)
    #early stop params
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_ng', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)

    args = parser.parse_args()
    print(args)
    args.dataset = args.dataset
    args.time = 0
    args.time_step = 0
    mkdir(args.dataset)
    hope = Model(args, isLoadModel)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()
    hope.test()

