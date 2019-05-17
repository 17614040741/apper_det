#coding=utf-8

import csv

def merge_shape_glass_beard_features():
    file_list = ['./result-1.csv','./result-2.csv','./result-3.csv']
    headers = ['folder','face_0','face_1','face_2','face_3','face_4','glass','beard']

    with open('./result-all.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for file_path in file_list:
            with open(file_path, 'r') as input_file:
                reader = csv.reader(input_file)
                for item in reader:
                    if reader.line_num == 1:
                        continue
                    writer.writerow(item)
    print('done')

def merge_face_encoding_and_other_features():
    #读入两个文件的数据用于比较
    others = []
    encodings = []
    with open('./result-all.csv','r') as other:
        other_reader = csv.reader(other)
        for other_item in other_reader:
            if other_reader.line_num ==1:
                continue
            others.append(other_item)
    with open('./test-all-embedding.csv','r') as encoding_file:
        encoding_reader = csv.reader(encoding_file)
        for encoding_item in encoding_reader:
            if encoding_reader.line_num == 1:
                continue
            encodings.append(encoding_item)

    headers = ['folder','face_0','face_1','face_2','face_3','face_4','glass','beard','face_encoding']
    with open('./facial_features.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        counter = 0
        for other_item in others:
            folder_name = other_item[0]
            for encoding_item in encodings:
                if folder_name == encoding_item[0]:
                    print(folder_name, ', ', encoding_item[0])
                    counter+=1
                    line = [item for item in other_item]
                    for i in range(1,len(encoding_item)):
                        line.append(encoding_item[i])
                    print(len(line))
                    writer.writerow(line)
            print(counter)

if __name__=='__main__':
    merge_face_encoding_and_other_features()