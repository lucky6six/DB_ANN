# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import csv
import os
import threading
import psutil
import queue
# import mkl
# mkl.get_max_threads()
# sys.path.append('/home/wwj/Vector_DB_Acceleration/faiss/python')
from datasets import evaluate
import faiss

current_file_path = os.path.realpath(__file__)
current_directory = os.path.dirname(current_file_path)
print("data path:" + current_directory)

sift1M_base_dataPath = current_directory + "/sift/sift_base.fvecs"
sift1M_groundtruth_dataPath = current_directory + "/sift/sift_groundtruth.ivecs"
sift1M_learn_dataPath = current_directory + "/sift/sift_learn.fvecs"
sift1M_query_dataPath = current_directory + "/sift/sift_query.fvecs"

csv_log_path = current_directory + "/eva_logs/local_sift1M_cpu_mem.csv"

# sift1M_base_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/sift_base.fvecs"
# sift1M_groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/sift_groundtruth.ivecs"
# sift1M_learn_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/sift_learn.fvecs"
# sift1M_query_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/sift_query.fvecs"

# csv_log_path = "/home/wwj/Vector_DB_Acceleration/faiss/python/wwj_test_codes/eva_logs/local_sift1M_cpu_mem.csv"

# sift1M_base_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_base.fvecs"
# sift1M_groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_groundtruth.ivecs"
# sift1M_learn_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_learn.fvecs"
# sift1M_query_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_query.fvecs"

# csv_log_path = "/home/wwj/Vector_DB_Acceleration/faiss/python/wwj_test_codes/eva_logs/remote_sift1M_cpu_mem.csv"

dataset = "sift1M"
processor = "cpu"
# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
# index_type = "IVF4096,Flat"
# index_type = "IVF512,PQ32"
# index_type = "IVF1024,PQ32"
# index_type = "IVF2048,PQ32"
index_type = "IVF4096,PQ32"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

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


#################################################################
#  Main program
#################################################################

tmpdir = '/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/trained_index/'

if 1:
    if not os.path.exists(csv_log_path):
        fd = open(csv_log_path, 'w')
        fd.close()
        print("create new csv log file")

    print("open csv log")
    csv_log_file = open(csv_log_path, 'a')
    csv_log_writer = csv.writer(csv_log_file)
    csv_log_writer.writerow(csv_log_title)
    
    # train the index
    xt = fvecs_read(sift1M_learn_dataPath)
    index = faiss.index_factory(xt.shape[1], index_type)
    print("training index")
    index.train(xt)
    
    xb = fvecs_read(sift1M_base_dataPath)
    index.add(xb)

    # load query vectors and ground-truth
    xq = fvecs_read(sift1M_query_dataPath)
    gt = ivecs_read(sift1M_groundtruth_dataPath)

    # print the size of xq
    print("xq size:", xq.shape)
    
    for lnprobe in range(10):
        nprobe = 1 << lnprobe
        
        if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
            index.nprobe = nprobe
        
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()

        t, r = evaluate(index, xq, gt, 100)
        
        q.put(1)
        cpu_monitor.join()
        cpu_usage = ret_q.get()
        
        csv_log_data = [dataset, nprobe, index_type, processor, t, 1.0/(t/1000.0), r[1], r[10], r[100], cpu_usage]
        csv_log_writer.writerow(csv_log_data)

        print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, t, r[1], r[10], r[100]))
        print("cpu usage:", cpu_usage)
    
    csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', ''])

    csv_log_file.close()
