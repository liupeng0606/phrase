import pandas as pd
#
# data = pd.read_csv("./test_data&labels/DSAP_TestW_Input.csv", sep=",", header=0)
#
# Phrase = data["Word"]
#
# Valence_Mean = data["Valence_Mean"]
#
# Arousal_Mean = data["Arousal_Mean"]
#
# p_data = list(zip(Phrase, Valence_Mean, Arousal_Mean))
#
# p_data = pd.DataFrame(p_data)
#
# print(p_data)
#
# p_data.to_csv('./simple_chinese_train/word.csv', sep=',', header=None, index=None)


# data = pd.read_csv("./test_data&labels/DSAP_TestW_Gold.txt", sep=",", header=None, usecols=[1, 2]).astype('float32')
# data.to_csv('./test_encode&labels/word_labels.csv', sep=',', header=None, index=None)


#
# def gen_labels():
#     data = pd.read_csv("./simple_chinese_train/word.csv", sep=",", header=None, usecols=[1, 2], encoding="GBK")
#     data.to_csv('./train_labels/word.csv', sep=',', header=None, index=None)

#
data = pd.read_csv("./test_data&labels/DSAP_TestW_Gold.txt", sep=",", header=None, usecols=[1,2], encoding="GBK")



print(data)

# data = list(map(lambda x: x.strip(), data))

p_data = pd.DataFrame(data)
p_data.to_csv('./test_encode&labels/word_label.csv', sep=',', header=None, index=None)

