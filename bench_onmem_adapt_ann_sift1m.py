# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import numpy as np
import csv
import os
import threading
import psutil
import queue
from datasets import evaluate
import faiss
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

current_file_path = os.path.realpath(__file__)
current_directory = os.path.dirname(current_file_path)
print("data path:" + current_directory)

sift1M_base_dataPath = current_directory + "/sift/sift_base.fvecs"
sift1M_groundtruth_dataPath = current_directory + "/sift/sift_groundtruth.ivecs"
sift1M_learn_dataPath = current_directory + "/sift/sift_learn.fvecs"
sift1M_query_dataPath = current_directory + "/sift/sift_query.fvecs"

csv_log_path = current_directory + "/eva_logs/local_sift1M_cpu_mem.csv"

dataset = "sift1M"
processor = "cpu"
# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
# index_type = "IVF4096,Flat"
# index_type = "IVF512,PQ32"
# index_type = "IVF1024,PQ32"
index_type = "IVF2048,PQ32"
# index_type = "IVF4096,PQ32"
csv_log_title = ["dataset", "row_select", "nprobe", "index_type", "processor", "search_latency", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]



#################################################################
# Small I/O functions
#################################################################

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

class CPU_Monitor(threading.Thread):
    def __init__(self, q, ret_q):
        super(CPU_Monitor, self).__init__()
        self.q = q
        self.ret_q = ret_q
        self.cpu_usage = 0
        
    def run(self):
        while self.q.get() == 0:
            self.cpu_usage = psutil.cpu_percent(None)
        
        self.ret_q.put(self.cpu_usage)
        
#new glocal var
if not os.path.exists(csv_log_path):
    fd = open(csv_log_path, 'w')
    fd.close()

#二维数组
mapping_batch_rows = 10
mapping_nprobe_cols = 10
# mapping = np.zeros([mapping_batch_rows+1, mapping_nprobe_cols+1])

#总query数
total_query = 10000
query_stride = total_query//mapping_batch_rows

#总聚类数
total_nprobe = 30
nprobe_stride = total_nprobe//mapping_nprobe_cols

#保守时延偏移
latency_offset = 0.0000
# #对照组的nprobe
# baseline_nprobe = 1024


# train the index
xt = fvecs_read(sift1M_learn_dataPath)
index = faiss.index_factory(xt.shape[1], index_type)
index.train(xt)

xb = fvecs_read(sift1M_base_dataPath)
index.add(xb)

# load query vectors and ground-truth
xq = fvecs_read(sift1M_query_dataPath)
gt = ivecs_read(sift1M_groundtruth_dataPath)

acceptable_latency_list = []
A_latency_list = []
H_latency_list = []
L_latency_list = []
rA1_list = []
rA10_list = []
rA100_list = []
rH1_list = []
rH10_list = []
rH100_list = []
rL1_list = []
rL10_list = []
rL100_list = []
nums = []
nprobe_H = 28
nprobe_L = 3

#################################################################
#  Main program
#################################################################

def init():
    csv_log_file = open(csv_log_path, 'w')
    csv_log_writer = csv.writer(csv_log_file)
    csv_log_writer.writerow(csv_log_title)
    
    for rows_select in range(0, total_query + 1, query_stride):
        if rows_select == 0:
            continue
    # for rows_select in rows_selects:
        # print("rows_select:", rows_select)
        xq_select = xq[:rows_select]
        gt_select = gt[:rows_select]
        
        # for lnprobe in range(10):
        for lnprobe in range(0, total_nprobe + 1, nprobe_stride):
            if lnprobe == 0:
                continue
            nprobe = lnprobe
            
            if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
                index.nprobe = nprobe
            
            q = queue.Queue()
            ret_q = queue.Queue()
            cpu_monitor = CPU_Monitor(q, ret_q)
            q.put(0)
            cpu_monitor.start()
            total = 0
            for i in range(10):
                t, r = evaluate(index, xq_select, gt_select, 100)
                total = total + t
            t = total/10
            
            q.put(1)
            cpu_monitor.join()
            cpu_usage = ret_q.get()
            
            csv_log_data = [dataset, rows_select, nprobe, index_type, processor, t, 1.0/(t/1000.0), r[1], r[10], r[100], cpu_usage]
            # mapping[rows_select//query_stride][nprobe//nprobe_stride] = t
             
            csv_log_writer.writerow(csv_log_data)

            # print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, t, r[1], r[10], r[100]))
            # print("cpu usage:", cpu_usage)
    
    # csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', '', ''])

    csv_log_file.close()

def query(batch_size, latency):
    # print("main")
    df = pd.read_csv(csv_log_path)[['row_select', 'nprobe', 'search_latency']]
    
    start_position = ((batch_size-1)//query_stride) * mapping_nprobe_cols
    satisfied_nprobe = 0
    
    for i in range(mapping_nprobe_cols):
        if df.iloc[start_position + i]['search_latency'] > latency - latency_offset:
            # if i > 0:
            #     satisfied_nprobe = df.iloc[start_position + i - 1]['nprobe']
            break
        else:
            satisfied_nprobe = df.iloc[start_position + i]['nprobe']
    
    # print("selected nprobe:",satisfied_nprobe)
    
    if satisfied_nprobe == 0:
        print("batch_size = %s | acceptable latency = %.4f | No satisfied nprobe" % (batch_size, latency))
        return
    
    xq_select = xq[:batch_size]
    gt_select = gt[:batch_size]
    
    if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
        index.nprobe = int(satisfied_nprobe)
            
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()
        total = 0
        for i in range(100):
            t, r = evaluate(index, xq_select, gt_select, 100)
            total = total + t
        t = total/100
        q.put(1)
        cpu_monitor.join()  
        cpu_usage = ret_q.get()
        print("batch_size = %s | acceptable latency = %.4f | selected nprobe = %4d | latency = %.4f ms | recalls(top 1/10/100)= %.4f %.4f %.4f" % (batch_size, latency, satisfied_nprobe, t, r[1], r[10], r[100]))
        return t,r[1],r[10],r[100]

def baseline_query(batch_size, nprobe):
    index.nprobe = int(nprobe)
    xq_select = xq[:batch_size]
    gt_select = gt[:batch_size]
    q = queue.Queue()
    ret_q = queue.Queue()
    cpu_monitor = CPU_Monitor(q, ret_q)
    q.put(0)
    cpu_monitor.start()
    total = 0
    for i in range(100):
        t, r = evaluate(index, xq_select, gt_select, 100)
        total = total + t
    t = total/100
    q.put(1)
    cpu_monitor.join()  
    cpu_usage = ret_q.get()
    print("batch_size = %s | selected nprobe = %4d | latency = %.4f ms | recalls(top 1/10/100)= %.4f %.4f %.4f" % (batch_size, nprobe, t, r[1], r[10], r[100]))
    return t,r[1],r[10],r[100]

def adapt_test():
    for batch_size in range(500, 10000, 1000):
        for latency in range(4, 15, 2):
            query(batch_size, latency/1000)

def baseline(nprobe):
    print("baseline nprobe = %d" % nprobe)
    for batch_size in range(500, 10000, 1000):
            xq_select = xq[:batch_size]
            gt_select = gt[:batch_size]
            
            if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
                #复用latency参数，实际传递的是对照组的nprobe
                index.nprobe = bprobe     
                q = queue.Queue()
                ret_q = queue.Queue()
                cpu_monitor = CPU_Monitor(q, ret_q)
                q.put(0)
                cpu_monitor.start()
                t, r = evaluate(index, xq_select, gt_select, 100)
                q.put(1)
                cpu_monitor.join()  
                cpu_usage = ret_q.get()
                
                print("batch_size = %s  | latency = %.4f ms | recalls(top 1/10/100)= %.4f %.4f %.4f" % (batch_size, t, r[1], r[10], r[100]))


# acceptable_latency_list = []
# A_latency_list = []
# H_latency_list = []
# L_latency_list = []
# rA1_list = []
# rA10_list = []
# rA100_list = []
# rH1_list = []
# rH10_list = []
# rH100_list = []
# rL1_list = []
# rL10_list = []
# rL100_list = []
# nprobe_H = 20
# nprobe_L = 5
def plot_latency(x):
    plt.figure()
    plt.plot(x, A_latency_list, color='blue', label='A')
    plt.plot(x, H_latency_list, color='red', label='H')
    plt.plot(x, L_latency_list, color='green', label='L')
    plt.plot(x, acceptable_latency_list, color='black', label='acceptable_latency')
    plt.xlabel('acceptable_latency')
    plt.ylabel('latency')
    plt.title('latency of different ANN')
    # plt.show()
    plt.legend(loc='center left',fontsize=5)
    plt.savefig('latency.png')
    
def plot_r1(x):
    plt.figure()
    plt.plot(x, rA1_list, color='blue', label='A')
    plt.plot(x, rH1_list, color='red', label='H')
    plt.plot(x, rL1_list, color='green', label='L')
    plt.xlabel('acceptable_latency')
    plt.ylabel('recalls_top1')
    plt.title('recalls_top1 of different ANN')
    # plt.show()
    plt.legend(loc='center left',fontsize=5)
    plt.savefig('recalls_top1.png')
def plot_r10(x):
    plt.figure()
    plt.plot(x, rA10_list, color='blue', label='A')
    plt.plot(x, rH10_list, color='red', label='H')
    plt.plot(x, rL10_list, color='green', label='L')
    plt.xlabel('acceptable_latency')
    plt.ylabel('recalls_top10')
    plt.title('recalls_top10 of different ANN')
    # plt.show()
    plt.legend(loc='center left',fontsize=5)
    plt.savefig('recalls_top10.png')
def plot_r100(x):
    plt.figure()
    plt.plot(x, rA100_list, color='blue', label='A')
    plt.plot(x, rH100_list, color='red', label='H')
    plt.plot(x, rL100_list, color='green', label='L')
    plt.xlabel('acceptable_latency')
    plt.ylabel('recalls_top100')
    plt.title('recalls_top100 of different ANN')
    # plt.show()
    plt.legend(loc='center left',fontsize=5)
    plt.savefig('recalls_top100.png')


def latency_recalls_test(batch_size,nprobe_H,nprobe_L):
    #case1
    # for latency in range(4, 30, 2):
    #     acceptable_latency_list.append(latency/1000)
    #     #adapt-ANN
    #     adapt_ann_latency,rA1,rA10,rA100 = query(batch_size, latency/1000)
    #case2
    num = 0.004
    for latency in range(40, 300, 5):
        acceptable_latency_list.append(latency/10000)
        #adapt-ANN
        nums.append(num)
        num = num + 0.0005
        adapt_ann_latency,rA1,rA10,rA100 = query(batch_size, latency/10000)
        A_latency_list.append(adapt_ann_latency)
        rA1_list.append(rA1)
        rA10_list.append(rA10)
        rA100_list.append(rA100)
        #high-fixed-nprobe-ANN
        high_latency,rH1,rH10,rH100 = baseline_query(batch_size, nprobe_H)
        H_latency_list.append(high_latency)
        rH1_list.append(rH1)
        rH10_list.append(rH10)
        rH100_list.append(rH100)
        #low-fixed-nprobe-ANN
        low_latency,rL1,rL10,rL100 = baseline_query(batch_size, nprobe_L)
        L_latency_list.append(low_latency)
        rL1_list.append(rL1)
        rL10_list.append(rL10)
        rL100_list.append(rL100)
    #case3
    # for latency in range(300, 40, -5):
    #     acceptable_latency_list.append(latency/10000)
    #     #adapt-ANN
    #     nums.append(num)
    #     num = num + 0.0005
    #     adapt_ann_latency,rA1,rA10,rA100 = query(batch_size, latency/10000)
    #     A_latency_list.append(adapt_ann_latency)
    #     rA1_list.append(rA1)
    #     rA10_list.append(rA10)
    #     rA100_list.append(rA100)
    #     #high-fixed-nprobe-ANN
    #     high_latency,rH1,rH10,rH100 = baseline_query(batch_size, nprobe_H)
    #     H_latency_list.append(high_latency)
    #     rH1_list.append(rH1)
    #     rH10_list.append(rH10)
    #     rH100_list.append(rH100)
    #     #low-fixed-nprobe-ANN
    #     low_latency,rL1,rL10,rL100 = baseline_query(batch_size, nprobe_L)
    #     L_latency_list.append(low_latency)
    #     rL1_list.append(rL1)
    #     rL10_list.append(rL10)
    #     rL100_list.append(rL100)
    plot_latency(acceptable_latency_list)
    plot_r1(acceptable_latency_list)
    plot_r10(acceptable_latency_list)
    plot_r100(acceptable_latency_list)
    # plot_latency(nums)
    # plot_r1(nums)
    # plot_r10(nums)
    # plot_r100(nums)
        
        
        
        
# def recalls_test2(batch_size,nprobe_H,nprobe_L):
#     for latency in range(4, 15, 2):
#         query(batch_size, latency/1000)
#         baseline_query(batch_size, nprobe_H)
#         baseline_query(batch_size, nprobe_L)

if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    latency = float(sys.argv[2])
    # print('batch_size: ', batch_size)
    # print('max latency: ', latency)
    # init()
    if(batch_size == 0):
        adapt_test()
    if(batch_size == -1):
        for bprobe in range(0, total_nprobe + 1, nprobe_stride):
            if bprobe == 0:
                continue
            baseline(bprobe)
    else:
        #将adapt-ANN与对照组的固定nprobe传递给query函数,获取时延与recalls变化图像
        # latency_recalls_test(batch_size,nprobe_H,nprobe_L)
    # else:
        query(batch_size, latency)