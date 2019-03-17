import numpy as np
import h5py
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-probe", default='./result/probe.h5', help="Path to probe")
ap.add_argument("-gallery", default='./result/gallery.h5', help="Path to gallery")
ap.add_argument("-result", default='./result/search_result.csv', help="file to save the result")
args = vars(ap.parse_args())

# read in indexed images feature vectors and corresponding image names
h5f = h5py.File(args["gallery"], 'r')
gallery_feats = h5f['hash48'][:]
gallery_fig_name = h5f['fig_name'][:]

h5f = h5py.File(args["probe"], 'r')
probe_feats = h5f['hash48'][:]
probe_fig_name = h5f['fig_name'][:]
h5f.close()

print("--------------------------------------------------")
print("               searching starts                   ")
print("--------------------------------------------------")


maxres = 200  # the top maxres images
imlist_all = []
for i in range(len(probe_fig_name)):
    print("query image No. %d , %d images in total" % ((i + 1), len(probe_fig_name)))

    # 异或法计算汉明距离
    queryVec = np.tile(probe_feats[i], (len(gallery_feats), 1))
    xor = np.logical_xor(queryVec, gallery_feats)
    scores = np.sum(xor == 1, 1)
    rank_ID = np.argsort(scores) # 从小到大排序，汉明距离越小编码越相近

    # # 点乘法衡量编码距离(本质同汉明距离)
    # queryVec = probe_feats[i]
    # scores = np.dot(queryVec, gallery_feats.T)
    # rank_ID = np.argsort(scores)[::-1]
    # # rank_score = scores[rank_ID]

    # number of top retrieved images to show
    img_name = str(probe_fig_name[i], encoding='utf-8')
    imlist = [str(gallery_fig_name[index], encoding='utf-8') for i, index in enumerate(rank_ID[0:maxres])]
    imlist.insert(0, img_name)
    imlist_all.append(imlist)
    # print("top %d images in order are: " %maxres, imlist)

# 从列表写入csv文件
csvFile2 = open(args["result"], 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile2)
for i in range(len(imlist_all)):
    writer.writerow(imlist_all[i])
csvFile2.close()
