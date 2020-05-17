# coding=utf-8

# Change by CZ.
import  csv
from csv import  DictWriter
import pandas  as pd



def dict_of_list_to_csv(list, out_file, attributes):

    """
    :param list:   [{'title': '真心半解', 'rate': '8.0', 'id': 33420285}, {'title': '利刃出鞘', 'rate': '8.2', 'id': 30318116}]
    :param out_file: file name
    :return: csv file
    """
    out = []
    out_id = set()
    for i in list:
        item = []
        for j in attributes:
            item.append(i[j])
        out.append(item)
        out_id.add(i['id'])
    res = pd.DataFrame(out)
    out_id = pd.DataFrame(out_id)
    res.to_csv('../data/'+out_file, encoding = 'utf-8',index = False,header = attributes)
    out_id.to_csv('../data/movie_id.csv', encoding='utf-8', index=False, header=['id'])

def sec_dict_of_list_to_csv(list, out_file, attributes):
    """
    :param list:
    # data = [
    #     {'key_1': {'calc1': 42, 'calc2': 3.142}},
    #     {'key_2': {'calc1': 123.4, 'calc2': 1.414}},
    #     {'key_3': {'calc1': 2.718, 'calc2': 0.577}}
    # ]
    :return:
    """
    try:
        with open('../data/'+out_file, 'wb') as f:
            writer = DictWriter(f, attributes)
            writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
            for i in data:
                key, values = i.items()[0]
                writer.writerow(dict(key=key, **values))
    except:
        f.close()

def list_to_csv(list, out_file):
    """
    :param list:  ['aa', 'bbb']
    :param out_file: file_name
    :return: csv file
    """
    res = pd.DataFrame(list)
    res.to_csv('../data/'+out_file, encoding = 'utf-8', mode='a', index = False,header = ['comments'])
    # w = csv.writer(open('../data/'+out_file, 'wb'), quoting=csv.QUOTE_ALL)
    # for word in list:
    #     w.writerow([word])
