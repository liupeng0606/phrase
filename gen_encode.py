import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(ip='222.19.197.228', port=5555, check_version=False)  # ip address of the GPU machine

# x1 = bc.encode(["很漂亮"])
# x2 = bc.encode(["很漂亮","真的"])
# print(x2.shape)


data = pd.read_csv("./test_encode&labels/word_no_label.csv", sep=",", header=None, usecols=[0], ).values

data = data.tolist()

data = list(map(lambda x: x[0], data))
print(data)

encode_list = bc.encode(data)
print(len(encode_list))
p_data = pd.DataFrame(encode_list)
p_data.to_csv('./test_encode&labels/word_encode.csv', header=None, index=None)

