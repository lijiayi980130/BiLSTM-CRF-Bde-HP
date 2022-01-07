import os
import re
from sklearn.model_selection import train_test_split
from xml.dom.minidom import parse
import xml.dom.minidom
import nltk

def remove_space(temp_split):
    words_split = []
    for word in temp_split:
        if(word!='' and word!=' '):
            words_split.append(word)

    return words_split

def data_preprocess():
    """
    change the form of train_data for HPO
    """
    train_vocab = ""
    sentences = []
    with open("D:\DeepLearning\DeepLearning_Code\PGR\corpora/11_03_2019_corpus/train.tsv",encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            data = line.strip("\n").split("\t")

            data[1] = re.sub(r'[\u2000-\u20ff]',' ',data[1])

            data[1] = re.sub(r'\xa0', ' ', data[1])

            sentences.append(data[1])



            rule = re.compile(r'(\W|_)')
            words_split = rule.split(data[1].strip())
            gene_split = rule.split(data[2].strip())
            phenotype_split = rule.split(data[3].strip())
            words_split = remove_space(words_split)
            gene_split = remove_space(gene_split)
            phenotype_split = remove_space(phenotype_split)

            words_label = ["O"]*len(words_split)

            for id,word in enumerate(words_split):
               flag = True
               if(word == gene_split[0]):
                   for i in range(1,len(gene_split)):
                       if(words_split[id+i] != gene_split[i]):
                           flag = False

                   if(flag):
                       words_label[id] = "B-Gene"
                       for k in range(1,len(gene_split)):
                           words_label[id+k] = "I-Gene"

               elif (word == phenotype_split[0]):
                   for i in range(1, len(phenotype_split)):
                       if (words_split[id + i] != phenotype_split[i]):
                           flag = False

                   if (flag):
                       words_label[id] = "B-Phenotype"
                       for k in range(1, len(phenotype_split)):
                           words_label[id + k] = "I-Phenotype"

            for i in range(len(words_split)):
                train_vocab = train_vocab + words_split[i]+"\t"+words_label[i]+"\n"

            train_vocab = train_vocab+"\n"


        with open("D:/DeepLearning/Human_Phenotype/train_2019.txt","w",encoding="utf-8") as f:
            f.write(train_vocab)


def data_split():
    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/all_data.tsv",encoding="utf-8") as f:
        data = f.read()

    sen = data.split("\n\n")
    print(len(sen))

    train_set,test_set = train_test_split(sen,test_size=0.2,random_state=42)

    print(len(train_set))
    print(len(test_set))


    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/train_1.tsv", "w", encoding="utf-8") as f:
        vocab = "\n\n".join(train_set)
        f.write(vocab)

    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/test_1.tsv", "w", encoding="utf-8") as f:
        vocab = "\n\n".join(test_set)
        f.write(vocab)

def change_label():
    str = ""
    with open("D:/DeepLearning/Human_Phenotype/HPO_IOB/gsc_test.tsv", "r", encoding="utf-8") as f:
        data = f.read()

        train_data = data.split("\n\n")
        train_data = [token.split("\n") for token in train_data]
        train_data = [[j.split() for j in i] for i in train_data]

        train_data.pop()

        for sen_word in train_data:
            for pos in range(len(sen_word)):  # 遍历序列中的每个位置
                curr_entity = sen_word[pos][1]
                if (curr_entity[-4:] == "Gene"):
                    sen_word[pos][1] = "O"
                elif pos == len(sen_word) - 1:  # 序列的最后一个位置
                    if curr_entity.startswith("B-"):
                        sen_word[pos][1] = curr_entity.replace("B-", "S-")
                    elif curr_entity.startswith("I-"):
                        sen_word[pos][1] = curr_entity.replace("I-", "E-")
                else:
                    print(sen_word[pos+1])
                    next_entity = sen_word[pos + 1][1]
                    if curr_entity.startswith("B-"):
                        if next_entity.startswith("O") or next_entity.startswith("B-"):
                            sen_word[pos][1] = curr_entity.replace("B-", "S-")
                    elif curr_entity.startswith("I-"):
                        if next_entity.startswith("O") or next_entity.startswith("B-"):
                            sen_word[pos][1] = curr_entity.replace("I-", "E-")
    # print(str)
    vocab = ""
    for sen_word in train_data:
        for word in sen_word:
            vocab = vocab + word[0] + "\t" + word[1] + "\n"
        vocab = vocab + "\n"

    with open("D:/DeepLearning/Human_Phenotype/HPO_IOBES/gsc_test.tsv", "w", encoding="utf-8") as f:
        f.write(vocab)

def form_text():
    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/train.tsv", encoding="utf-8") as f:
        data = f.read()

        train_data = data.split("\n\n")
        train_data = [token.split("\n") for token in train_data]
        train_data = [[j.split() for j in i] for i in train_data]
        count = 0
        vocab = ""
        for sen_word in train_data:
            count = count + 1
            for word in sen_word:
                if(word[1]==''):
                    print(word)
                vocab = vocab + word[1]+" "
            vocab = vocab + "\n"

    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/train.txt", "w",encoding="utf-8") as f:
        f.write(vocab)


def xml_txt(files,path,path1):
    vocab = ""
    # 使用minidom解析器打开 XML 文档
    for file in files:
        xml_path = path + file
        DOMTree = xml.dom.minidom.parse(xml_path)
        #collection = DOMTree.documentElement

        document = DOMTree.getElementsByTagName('document')
        #vocab = vocab+document[0].getAttribute("id") +"\t"

        sentences = document[0].getElementsByTagName('sentence')

        #entitys = document[0].getElementsByTagName('entity')

        for sentence in sentences:
           vocab = vocab + document[0].getAttribute("id") + "\t"
           vocab = vocab + sentence.getAttribute("text") + "\t"


           if(len(sentence.childNodes)!=0):
                 entitys = sentence.getElementsByTagName('entity')
                 for entity in entitys:
                     vocab = vocab + entity.getAttribute("text") + "\t"
                     vocab = vocab + entity.getAttribute("ontology_id") + "\t"

                 vocab = vocab +"\n"

        #print(vocab)



    print(vocab)

    with open(path1, "w",encoding="utf-8") as f:
        f.write(vocab)

    '''
    # 打印每部电影的详细信息
    for movie in movies:
        print
        "*****Movie*****"
        if movie.hasAttribute("title"):
            print
            "Title: %s" % movie.getAttribute("title")

        type = movie.getElementsByTagName('type')[0]
        print
        "Type: %s" % type.childNodes[0].data
        format = movie.getElementsByTagName('format')[0]
        print
        "Format: %s" % format.childNodes[0].data
        rating = movie.getElementsByTagName('rating')[0]
        print
        "Rating: %s" % rating.childNodes[0].data
        description = movie.getElementsByTagName('description')[0]
        print
        "Description: %s" % description.childNodes[0].data

    '''
def gsc_preprocess():

    test_vocab = ""

    with open("D:\DeepLearning\corpus\corpus\GSC/GSCplus_test_gold.tsv","r",encoding="utf-8") as f:
        data = f.read()

        train_data = data.split("\n\n")
        train_data = [token.split("\n") for token in train_data]
        train_data = [[j.split("\t") for j in i] for i in train_data]

        train_data.pop()

        #print(train_data)
        for abs in train_data:

            pheno_list = []
            sen_list = []
            senlabel_list = []
            pos =-1

            split_line = abs[1][0].replace('-', ' - ').replace('/', ' / ')
            sentences = nltk.sent_tokenize(split_line)
            sentences = [nltk.word_tokenize(sent) for sent in sentences]

            for i in range(2,len(abs)):
                if(int(abs[i][0])>int(pos)):
                    abs[i][2] = abs[i][2].replace('-', ' - ').replace('/', ' / ')
                    phenotype_split = nltk.word_tokenize(abs[i][2].strip())
                    pheno_list.append(phenotype_split)
                    pos = abs[i][1]

            for sentence in sentences:

                words_label = ["O"] * len(sentence)
                senlabel_list.append(words_label)
                sen_list.append(sentence)


            for phenotype_split in pheno_list:

                for sentence,words_label in zip(sen_list,senlabel_list):

                   for id, word in enumerate(sentence):
                       flag = True
                       if (word == phenotype_split[0]):
                          for i in range(1, len(phenotype_split)):
                             if(id+i < len(sentence)):
                                 if (sentence[id + i] != phenotype_split[i]):
                                     flag = False
                                     break
                             else:
                                 flag = False

                          if (flag):
                             words_label[id] = "B-Phenotype"
                             for k in range(1, len(phenotype_split)):
                               words_label[id + k] = "I-Phenotype"
                             break

            for i in range(len(sen_list)):
                for j in range(len(sen_list[i])):
                    test_vocab = test_vocab + sen_list[i][j] + "\t" + senlabel_list[i][j] + "\n"

                test_vocab = test_vocab + "\n"

        with open("D:\DeepLearning\Human_Phenotype\HPO_IOB/gsc_test.tsv","w",encoding="utf-8") as f:
            f.write(test_vocab)

import torch.nn as nn
import torch

if __name__ == "__main__":
    #data_preprocess()
    #data_split()
    #change_label()
    #form_text()
    '''
    # 这里放着你要操作的文件夹名称
    path = 'D:/DeepLearning/DeepLearning_Code/PGR/corpora/10_12_2018_corpus/pgr_test/pgr_go/'
    path1 = 'D:/DeepLearning/DeepLearning_Code/PGR/corpora/10_12_2018_corpus/test_ner_go.tsv'

    # 把e:\get_key\目录下的文件名全部获取保存在files中
    files = os.listdir(path)
    #files = ['30845641.xml']
    #xml_txt(files, path, path1)
    
    for file in files:
        xml_path = path + file
        fin = open(xml_path,"r",encoding="utf-8")

        data = fin.read()
        data = data.strip().split("\n")
        for id,split in enumerate(data):
            if(split.startswith("\t<sentence")):
                temp = split[2:-1]
                temp = temp.replace("<","")
                temp = temp.replace(">","")
                temp = "\t<"+temp +">"
                data[id] = temp

        data ="\n".join(data)
        fout = open(xml_path, "w", encoding="utf-8")
        print(data)
        fout.write(data)
    
    xml_txt(files, path, path1)

    '''


    #xml_txt(files,path,path1)
    gsc_preprocess()
    #change_label()
    #form_text()


