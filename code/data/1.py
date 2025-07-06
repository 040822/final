import json

input_file = r'/root/final/code/data/translation2019zh_train.json'
output_file = r'/root/final/code/data/translation2019zh_train1.json'
max_lines = 500000

with open(input_file, 'r', encoding='utf-8') as fin, \
    open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    for line in fin:
       if count >= max_lines:
          break
       fout.write(line)
       count += 1