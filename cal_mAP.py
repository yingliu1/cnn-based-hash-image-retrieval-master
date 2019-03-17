# import numpy as np
import csv


def map_rank_quick_eval(query_info, val_info):

    QUERY_NUM = len(query_info)
    mAP = 0.0

    for q_index, query_list in enumerate(query_info):

        cnt = 0
        ap = 0.0

        for t_index in range(1, len(query_list)):
            if query_list[t_index] in val_info[q_index]:
                cnt += 1
                precise = cnt / t_index
                ap += precise

            if cnt == len(val_info[q_index])-1:
                break

        if cnt != 0:
            mAP += ap / cnt

    mAP = mAP / QUERY_NUM

    return mAP


if __name__ == '__main__':

    # dataset = ['search_test_a', 'search_val']
    query_file = csv.reader(open('./result/search_result.csv', 'r'))
    query_sorted = sorted(query_file, key=lambda x: (x[0]))

    val_file = csv.reader(open('./dataset/search_a/search_val/val.csv', 'r'))
    val_sorted = sorted(val_file, key=lambda x: (x[0]))

    mAP = map_rank_quick_eval(query_sorted, val_sorted)
    print('mAP:\t%f' % mAP)