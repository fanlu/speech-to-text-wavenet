# -*- coding:utf-8 -*-

_data_path = "asset/data/"

wordMap = {}
inverseWordMap = {}

phoneMap = {}
inversePhoneMap = {}


def createWordMapFromFile():
  f = file(_data_path + "6855map.txt")
  lines = f.readlines()
  index = 0
  for item in lines:
    item = item.replace("\n", "")
    item = item.replace("\r", "")
    item.strip()
    split = item.split(" ")
    wordMap[split[1]] = split[0]
    inverseWordMap[int(split[0])] = split[1]
    index = 1
  return


def createPhone():
  f = file(_data_path + "thchs30/data_thchs30/lm_phone/lexicon.txt")
  lines = f.readlines()
  for i, item in enumerate(lines):
    phoneMap[item.strip().split(" ")[0]] = i
    inversePhoneMap[i] = item.strip().split(" ")[0]
  return


def str2index(str_):
  str_ = str_.strip().decode('utf-8')
  l = []
  for ch in str_:
    if ch != u' ':
      l.append(wordMap[ch.encode('utf-8')])
  return l
  # p = re.compile(ur'[\u4e00-\u9fa5]')
  # return p.split(str_)


def index2str(index_list):
  str_ = ''
  for ch in index_list:
    if ch > 0:
      str_ += inverseWordMap[ch]
    elif ch == 0:  # <EOS>
      break
  return str_


def str2phoneindex(str_):
  str_ = str_.strip().split(" ")
  l = []
  for ch in str_:
    if ch != u' ':
      l.append(phoneMap[ch])
  return l
  # p = re.compile(ur'[\u4e00-\u9fa5]')
  # return p.split(str_)


def phoneindex2str(index_list):
  str_ = ''
  for ch in index_list:
    if ch > 0:
      str_ += inversePhoneMap[ch]
    elif ch == 0:  # <EOS>
      break
  return str_

def print_phoneindex(indices):
  for index_list in indices:
    print(phoneindex2str(index_list))

# print list of index list
def print_index(indices):
  for index_list in indices:
    print(index2str(index_list))


createWordMapFromFile()
createPhone()
#voca_size = len(wordMap)
voca_size = len(phoneMap)