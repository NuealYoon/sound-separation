# sound-separation


- dc_train.py의 DC는 Deep-learning Clustering 약자로 1ch 데이터를 입력 받아 2ch 데이터로 speration 해 준다.

입력 데이터:
1ch 데이터[1, 250, 257]

출력 데이터:
16ch 데이터[1, 64250, 16]

라벨:
2ch 데이터 [1, 250, 257]
flatened_ys: [1, 64250, 2]


라벨은 2가지 종류가 있다.
gt_mask, phase(duet kmeans inference)

gt_mask는 source에서 발생하는 소리를 2ch 마이크에 수음시, 2ch 마이크에서 동일한 주기중 개수가 많은 주기를 가지는 마이크가 source와 가까운 거리에 있다. 라고 가정한다.

gt_mask와 phase로 source가 어떤 마이크 channel에서 source가 수음 되는지, 구분 하는 loss로 쓸만 할까?
or 2개 다 사용하면 좋을 거 같은데 각 loss의 특성은 무엇일까?
