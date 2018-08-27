import xlwt
import pandas as pd

exl_path = r'E:\testFolder\company\test.xlsx'
tb = pd.read_excel(exl_path)
ent_list = tb.iloc[:, 1].values
ent_set = set(ent_list)

fault_list = check_ent(ent_set)


def check_ent(list):
    for a in list:
        tb2 = tb[tb['企业名'].isin([a])]
