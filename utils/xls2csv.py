#coding: utf-8
import os
import pandas as pd

def get_excel_list(path):
    # 获取Excel文件列表
    excel_file_list = []
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('xlsx') or file_name.endswith('xls'):
            excel_file_list.append(file_name)
    return excel_file_list

excel_path_arr = ['../pointdata4/traindata/', '../pointdata4/testdata/']

for excel_path in excel_path_arr:
    for excel_file in get_excel_list(excel_path):
        data = pd.read_excel(excel_path + excel_file, header=None,index_col=None)
        for key in data:
            if key == 3 or key == '3':
                data.pop(3)
        # data.to_csv(excel_path + '/output.csv', mode='a', encoding='utf-8')
        if os.access(excel_path + 'csv/', os.F_OK) == False:
            os.makedirs(excel_path + 'csv/')
        data.to_csv(excel_path + 'csv/' + excel_file[:-5] + '.csv', header=0,index=0,encoding='utf-8')
