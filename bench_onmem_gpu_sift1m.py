# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# !/usr/bin/env python2

from __future__ import print_function
from collections.abc import Callable, Iterable, Mapping
import os
import time
from typing import Any
import numpy as np
import pdb
import sys
import csv
import threading
import pynvml
import queue
import mkl
import pynvml
mkl.get_max_threads()
# sys.path.append('/home/wwj/Vector_DB_Acceleration/faiss/python')

current_file_path = os.path.realpath(__file__)
current_directory = os.path.dirname(current_file_path)
print("data path:" + current_directory)

import faiss
from datasets import load_sift1M, evaluate

sift1M_base_dataPath = current_directory + "/sift/sift_base.fvecs"
sift1M_groundtruth_dataPath = current_directory + "/sift/sift_groundtruth.ivecs"
sift1M_learn_dataPath = current_directory + "/sift/sift_learn.fvecs"
sift1M_query_dataPath = current_directory + "/sift/sift_query.fvecs"

# sift1M_base_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_base.fvecs"
# sift1M_groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_groundtruth.ivecs"
# sift1M_learn_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_learn.fvecs"
# sift1M_query_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1M/sift_query.fvecs"

csv_log_path = "/home/wwj/Vector_DB_Acceleration/faiss/python/wwj_test_codes/eva_logs/remote_sift1M_gpu_mem.csv"

dataset = "sift1M"
processor = "gpu_a100"
# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
# index_type = "IVF4096,Flat"
# index_type = "IVF512,PQ32"
# index_type = "IVF1024,PQ32"
# index_type = "IVF2048,PQ32"
index_type = "IVF4096,PQ32"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "gpu_usage/%"]

class GPU_Monitor(threading.Thread):
    def __init__(self, q, ret_q):
        super(GPU_Monitor, self).__init__()
        self.q = q
        self.ret_q = ret_q
        self.gpu_util = 0
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    def run(self):
        while self.q.get() == 0:
            self.gpu_util += pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        
        pynvml.nvmlShutdown()
        self.ret_q.put(self.gpu_util)

print("load data")

xb, xq, xt, gt = load_sift1M(sift1M_learn_dataPath, sift1M_base_dataPath, sift1M_query_dataPath, sift1M_groundtruth_dataPath)
nq, d = xq.shape

pass

# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()


#################################################################
#  Exact search experiment
#################################################################

print("============ Exact search")

flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0

index = faiss.GpuIndexFlatL2(res, d, flat_config)

print("add vectors to index")

index.add(xb)

print("warmup")

index.search(xq, 123)

print("benchmark")

for lk in range(11):
    k = 1 << lk
    t, r = evaluate(index, xq, gt, k)

    # the recall should be 1 at all times
    print("k=%d %.3f ms, R@1 %.4f" % (k, t, r[1]))


#################################################################
#  Approximate search experiment
#################################################################

print("============ Approximate search")

# index = faiss.index_factory(d, "IVF4096,PQ64")
index = faiss.index_factory(d, index_type)

# faster, uses more memory
# index = faiss.index_factory(d, "IVF16384,Flat")

# co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
# co.useFloat16 = True

# index = faiss.index_cpu_to_gpu(res, 0, index, co)
index = faiss.index_cpu_to_gpu(res, 0, index)

print("train")

index.train(xt)

print("add vectors to index")

index.add(xb)

print("warmup")

index.search(xq, 123)

print("benchmark")

if not os.path.exists(csv_log_path):
    fd = open(csv_log_path, 'w')
    fd.close()
    print("create new csv log file")

print("open csv log")
csv_log_file = open(csv_log_path, 'a')
csv_log_writer = csv.writer(csv_log_file)
csv_log_writer.writerow(csv_log_title)

print("starting search benchmark......")

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
        index.nprobe
        index.nprobe = nprobe
    
    q = queue.Queue()
    ret_q = queue.Queue()
    gpu_monitor = GPU_Monitor(q, ret_q)
    q.put(0)
    gpu_monitor.start()
        
    t, r = evaluate(index, xq, gt, 100)
    
    q.put(1)
    gpu_monitor.join()
    gpu_usage = ret_q.get()
        
    # write perf data to log file
    csv_log_data = [dataset, nprobe, index_type, processor, t, 1.0/(t/1000.0), r[1], r[10], r[100], gpu_usage]
    csv_log_writer.writerow(csv_log_data)

    print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, t, r[1], r[10], r[100]))
    print("gpu usage:", gpu_usage)

csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', ''])

csv_log_file.close()
