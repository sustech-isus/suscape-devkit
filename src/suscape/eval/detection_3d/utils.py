# coding:utf-8

'''
Name  : utils.py
Author: Tim Liu
Desc:
'''

from collections import defaultdict



def group_by_key(data_info,key):
    '''
    group the data by givec key
    :param data_info: dic
    :param key: given key
    :return:defaultdict(list)
    '''
    groups = defaultdict(list)
    for box in data_info:
        if key == 'bbox':
            groups[box[key].name].append(box)
        else:
            groups[box[key]].append(box)
    return groups



if __name__ == "__main__":

    class_names = ['car']
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    gt = \
    [
        {
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [974.2811881299899, 1714.6815014457964,-23.689857123368846],
        'size': [1.796, 4.488, 1.664],
        'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
        'name': 'car'
        },
        {
            'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [874.2811881299899, 1614.6815014457964, -22.689857123368846],
            'size': [1.596, 3.488, 1.264],
            'rotation': [0.04882026466054782, 0, 0, 0.8888642620837121],
            'name': 'car'
        },
        {
            'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [774.2811881299899, 1514.6815014457964, -21.689857123368846],
            'size': [1.496, 3.488, 1.064],
            'rotation': [0.10882026466054782, 0, 0, 0.8888642620837121],
            'name': 'car'
        },
        {
            'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [174.2811881299899, 914.6815014457964, -19.689857123368846],
            'size': [1.096, 3.488, 0.664],
            'rotation': [0.15882026466054782, 0, 0, 0.8888642620837121],
            'name': 'car'
        },
        {
            'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [274.2811881299899, 814.6815014457964, -18.689857123368846],
            'size': [1.786, 2.488, 0.564],
            'rotation': [0.11882026466054782, 0, 0, 0.7888642620837121],
            'name': 'car'
        },
        {
            'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [374.2811881299899, 714.6815014457964, -17.689857123368846],
            'size': [1.746, 1.488, 0.664],
            'rotation': [0.09882026466054782, 0, 0, 0.5888642620837121],
            'name': 'car'
        },
        {
            'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
            'translation': [474.2811881299899, 614.6815014457964, -14.689857123368846],
            'size': [1.706, 2.488, 0.564],
            'rotation': [0.04882026466054782, 0, 0, 0.2888642620837121],
            'name': 'car'
        }
    ]

    predictions = \
        [
            {
                'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [874.2811881299899, 1614.6815014457964, -22.689857123368846],
                'size': [1.696, 4.388, 1.564],
                'rotation': [0.04882026466054782, 0, 0, 0.8888642620837121],
                'name': 'car',
                'score': 0.98
            },
            {
                'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [674.2811881299899, 1814.6815014457964, -28.689857123368846],
                'size': [2.596, 4.488, 5.264],
                'rotation': [0.14882026466054782, 0, 0, 1.8888642620837121],
                'name': 'car',
                'score': 0.97
            },
            {
                'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [770.2811881299899, 1510.6815014457964, -20.689857123368846],
                'size': [0.496, 6.488, 8.064],
                'rotation': [0.18882026466054782, 0, 0, 0.7888642620837121],
                'name': 'car',
                'score': 0.95
            },
            {
                'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [170.2811881299899, 90.6815014457964, -20.689857123368846],
                'size': [1.096, 5.488, 3.664],
                'rotation': [0.35882026466054782, 0, 0, 1.8888642620837121],
                'name': 'car',
                'score': 0.88
            },
            {
                'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [264.2811881299899, 874.6815014457964, -98.689857123368846],
                'size': [11.786, 21.488, 6.564],
                'rotation': [6.11882026466054782, 0, 0, 3.7888642620837121],
                'name': 'car',
                'score': 0.87
            },
            {
                'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [1374.2811881299899, 1714.6815014457964, -117.689857123368846],
                'size': [11.746, 11.488, 10.664],
                'rotation': [10.09882026466054782, 0, 0, 10.5888642620837121],
                'name': 'car',
                'score': 0.85
            },
            {
                'sample_token': '0abcdefghijklmnopqrst55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': [4174.2811881299899, 6114.6815014457964, -114.689857123368846],
                'size': [11.706, 21.488, 10.564],
                'rotation': [10.04882026466054782, 0, 0, 10.2888642620837121],
                'name': 'car',
                'score': 0.75
            }
        ]

    gt_by_class_name = group_by_key(gt, 'name')
    pred_by_class_name = group_by_key(predictions, 'name')




    print('aaa')