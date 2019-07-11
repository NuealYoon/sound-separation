# sound-separation


- dc_train.py의 DC는 Deep-learning Clustering 약자로 1ch 데이터를 입력 받아 2ch 데이터로 speration 해 준다.

입력 데이터:
1ch 데이터[1, 250, 257]

출력 데이터:
16ch 데이터[1, 64250, 16]

라벨 데이터:
2ch 데이터 [1, 250, 257]
flatened_ys: [1, 64250, 2]




