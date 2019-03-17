import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(ip='222.19.197.228', port=5555)  # ip address of the GPU machine

bc.encode(["Ì«¿É¶ñ"])

# data = pd.read_csv("./test_encode&labels/word_m.csv", sep=",", header=None, usecols=[0]).values
# data = list(map(lambda item: item[0], data))
# print(data)
# print(len(data))
# encode_list = bc.encode(data)
# print(len(encode_list))
# p_data = pd.DataFrame(encode_list)
# p_data.to_csv('./test_encode&labels/word_m_encode.csv', header=None, index=None)

