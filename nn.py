
import os
#os.environ['GLOG_minloglevel'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
from caffe import layers,params
#import lmdb
from caffe.io import datum_to_array, array_to_datum
from caffe.proto import caffe_pb2

'''
#Architecture [(conv(3x3)->relu>pool(2x2))] x3 -> (fc1 (1280)) -> (fc2 (2048)) -> (fc3 (5))
def cnn_structure(lmdb,batch_size):
    n = caffe.NetSpec()
    n.data, n.label = layers.Data(batch_size=batch_size, backend=params.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = layers.Convolution(n.data, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu1 = layers.ReLU(n.conv1, in_place = True)
    n.pool1 = layers. Pooling(n.relu1, kernel_size=2, stride=2, pool=params.Pooling.MAX)

    n.conv2 = layers.Convolution(n.pool1, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu2 = layers.ReLU(n.conv2, in_place = True)
    n.pool2 = layers. Pooling(n.relu2, kernel_size=2, stride=2, pool=params.Pooling.MAX)

    n.conv3 = layers.Convolution(n.pool2, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu3 = layers.ReLU(n.conv3, in_place = True)
    n.pool3 = layers.Pooling(n.relu3,kernel_size=2,stride=2,pool=params.Pooling.MAX)

    n.fc1 = layers.InnerProduct(n.pool3, num_output=1280, weight_filler=dict(type='xavier'))
    n.relu4 =  layers.ReLU(n.fc1, in_place = True)

    n.fc2 = layers.InnerProduct(n.relu4,num_output=2048,weight_filler=dict(type='xavier'))
    n.relu5 = layers.ReLU(n.fc2,in_place=True)

    n.score = layers.InnerProduct(n.relu5, num_output=5, weight_filler=dict(type='xavier'))
    n.loss =  layers.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()
'''
'''
#architecture [(conv(3x3)->relu) x2 -> pool(2x2)  ] -> [(conv(3x3)->relu) x1 -> pool(2x2)  ]   -> (fc1 (5120)) -> (fc2 (5120)) -> (fc3 (5)) "
def cnn_structure2(lmdb,batch_size):
    n = caffe.NetSpec()
    n.data, n.label = layers.Data(batch_size=batch_size, backend=params.Data.LMDB, source=lmdb,
                              ntop=2)

    n.conv1 = layers.Convolution(n.data, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu1 = layers.ReLU(n.conv1, in_place = True)
    n.conv2 = layers.Convolution(n.relu1, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu2 = layers.ReLU(n.conv2, in_place = True)
    n.pool1 = layers. Pooling(n.relu2, kernel_size=2, stride=2, pool=params.Pooling.MAX)

    n.conv3 = layers.Convolution(n.pool1, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.relu3 = layers.ReLU(n.conv3, in_place = True)
    n.pool2 = layers.Pooling(n.relu3,kernel_size=2,stride=2,pool=params.Pooling.MAX)

    n.fc1 = layers.InnerProduct(n.pool2, num_output=5120, weight_filler=dict(type='xavier'))
    n.relu4 =  layers.ReLU(n.fc1, in_place = True)

    n.fc2 = layers.InnerProduct(n.relu4,num_output=5120,weight_filler=dict(type='xavier'))
    n.relu5 = layers.ReLU(n.fc2,in_place=True)

    n.score = layers.InnerProduct(n.relu5, num_output=5, weight_filler=dict(type='xavier'))
    n.loss =  layers.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()
'''
#architecture same as 2 with only 10 filters per layer
def cnn_structure3(lmdb,batch_size):
    n = caffe.NetSpec()
    n.data, n.label = layers.Data(batch_size=batch_size, backend=params.Data.LMDB, source=lmdb,
                              ntop=2)

    n.conv1 = layers.Convolution(n.data, kernel_size=3, num_output=10, weight_filler=dict(type='xavier'))
    n.relu1 = layers.ReLU(n.conv1, in_place = True)
    n.conv2 = layers.Convolution(n.relu1, kernel_size=3, num_output=10, weight_filler=dict(type='xavier'))
    n.relu2 = layers.ReLU(n.conv2, in_place = True)
    n.pool1 = layers. Pooling(n.relu2, kernel_size=2, stride=2, pool=params.Pooling.MAX)

    n.conv3 = layers.Convolution(n.pool1, kernel_size=3, num_output=10, weight_filler=dict(type='xavier'))
    n.relu3 = layers.ReLU(n.conv3, in_place = True)
    n.pool2 = layers.Pooling(n.relu3,kernel_size=2,stride=2,pool=params.Pooling.MAX)

    n.fc1 = layers.InnerProduct(n.pool2, num_output=2560, weight_filler=dict(type='xavier'))
    n.relu4 =  layers.ReLU(n.fc1, in_place = True)

    n.fc2 = layers.InnerProduct(n.relu4,num_output=2560,weight_filler=dict(type='xavier'))
    n.relu5 = layers.ReLU(n.fc2,in_place=True)

    n.score = layers.InnerProduct(n.relu5, num_output=5, weight_filler=dict(type='xavier'))
    n.loss =  layers.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


def solver(train_net_path,test_net_path):
    solv = caffe_pb2.SolverParameter()
    #solv.random_seed = 0xCAFFE
    solv.train_net = train_net_path
    solv.test_net.append(test_net_path)
    solv.test_interval = 100
    solv.test_iter.append(28)
    solv.max_iter = 1001
    solv.type = "SGD"
    solv.base_lr = 0.001
    solv.momentum  = 0.9
    solv.weight_decay = 5e-4  #regularization
    solv.lr_policy = 'inv'
    solv.gamma = 0.0001
    solv.power = 0.75
    #solv.display = 10000
    solv.snapshot = 500
    solv.snapshot_prefix = "/output/"
    solv.solver_mode = caffe_pb2.SolverParameter.GPU
    return solv


def main():

    train_net_path = '/output/train_val.prototxt'
    test_net_path = '/output/test.prototxt'
    solver_path = '/output/solver.prototxt'

    with open(train_net_path, 'w') as f:
        f.write(str(cnn_structure3('/dataset/train_data/', 256)))

    with open(test_net_path, 'w') as f:
        f.write(str(cnn_structure3('/dataset/validation_data/', 64)))

    with open(solver_path, 'w') as f:
        f.write(str(solver(train_net_path,test_net_path)))
    print '---------- prototext generated ----------'

    solvr = caffe.get_solver(solver_path)
    print '---------- solver acquired ----------'
    train(solvr)


def train(solver):
    print '---------- training network ---------- '
    niter = 1001
    test_interval = 100
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / float(test_interval))))
    train_acc=np.zeros(niter)
    print '---------- training network started ----------'

    for iteration  in range(niter):
        correct_per_iteration=0


        solver.step(1)  # SGD by Caffe

        train_loss[iteration] = solver.net.blobs['loss'].data

        correct_per_iteration = sum(solver.net.blobs['score'].data.argmax(1) == solver.net.blobs['label'].data)
        train_acc[iteration] = correct_per_iteration / float(256)

        print 'iteration %d started loss %f, correct per iteration %0.2f %%'%(iteration,train_loss[iteration],train_acc[iteration]*100)

        # full test
        if iteration % test_interval == 0:
            print 'Iteration', iteration, 'testing...'
            correct = 0
            for test_iteration in range(28):
                #print test_iteration
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
            test_acc[iteration // test_interval] = correct / float(1744)
            print 'test accuracy %0.2f'%(test_acc[iteration // test_interval]*100)

    print '---------- writing output of training completed----------'
    np.savetxt('/output/train_accuracy.csv',train_acc,delimiter=',')
    np.savetxt('/output/train_loss.csv',train_loss,delimiter=',')
    np.savetxt('/output/test_accuracy.csv',test_acc,delimiter=',')
    experiment = "dataset size : 11744\nclasses : 2,3,4,5,9\nsize: 64x64\nnormalized to range [0,1]\n1000 iter\n \
                  architecture [(conv(3x3)->relu) x2 -> pool(2x2)  ] -> [(conv(3x3)->relu) x1 -> pool(2x2)  ]   -> (fc1 (5120)) -> (fc2 (5120)) -> (fc3 (5)) "
    np.savetxt('/output/description.txt',[experiment],fmt='%s')

    print '---------- training & testing completed ----------'


if __name__ == '__main__':
    main()
    #floyd run --gpu --data LbPKmjJp86tef2rGBkTUAV:dataset --env caffe:py2 "python nn.py"
    # floyd run --mode jupyter --env caffe:py2 --data TwBVEYwEdYxaA3u4RgepbH:train_output --data LbPKmjJp86tef2rGBkTUAV:dataset
