import sys
import json
import io
from ssplit_tokenzier import ssplit_token_pos_lemma
import nltk

class Trie(object):
    class Node(object):
        def __init__(self):
            self.term = None
            self.next = {}

    def __init__(self, terms=[]):
        self.root = Trie.Node()
        for term in terms:
            self.add(term)

    def add(self, term):
        node = self.root
        for char in term:
            if not char in node.next:
                node.next[char] = Trie.Node()
            node = node.next[char]
        node.term = term

    def match(self, query):
        results = []
        for i in range(len(query)):
            node = self.root
            for j in range(i, len(query)):
                node = node.next.get(query[j])
                if not node:
                    break
                if node.term:
                    results.append((i, len(node.term)))
        return results

    def __repr__(self):
        output = []

        def _debug(output, char, node, depth=0):
            output.append('%s[%s][%s]' % (' ' * depth, char, node.term))
            for (key, n) in node.next.items():
                _debug(output, key, n, depth + 1)

        _debug(output, '', self.root)
        return '\n'.join(output)


class dic_ont():

    def __init__(self, ont_files):

        dicin = open(ont_files['dic_file'], 'r', encoding='utf-8')
        win_size = 50000
        Dic = []
        print("loading dict!")
        for line in dicin:
            line = line.strip()
            if len(line.split()) <= win_size:
                words = line.split()
                for i in range(len(words)):
                    if len(words[i]) > 3 and (not words[i].isupper()):
                        words[i] = words[i].lower()
                line = ' '.join(words[0:])
                Dic.append(line.strip())
        print("Dic_len:", len(Dic))
        dicin.close()

        self.dic_trie = Trie(Dic)
        print("load dic done!")

        # load word hpo mapping
        fin_map = open(ont_files['word_hpo_file'], 'r', encoding='utf-8')
        self.word_hpo = json.load(fin_map)
        fin_map.close()

        # load hpo word mapping
        fin_map = open(ont_files['hpo_word_file'], 'r', encoding='utf-8')
        self.hpo_word = json.load(fin_map)
        fin_map.close()

    def matching(self, source):

        fin = io.StringIO(source)
        fout = io.StringIO()

        sent_list = []
        sent = []
        sent_ori_list = []
        sent_ori = []

        for line in fin:
            line = line.strip()
            if line == "":
                sent_list.append(sent)
                sent_ori_list.append(sent_ori)
                sent = []
                sent_ori = []
            else:
                words = line.split('\t')
                words[1] = words[1].lower()
                sent.append(words[1])  # word lemma
                sent_ori.append(words[0])
        sent = []
        fin.close()

        for k in range(len(sent_list)):
            sent = sent_list[k]
            sentence = ' '.join(sent[0:]) + " "
            sentence_ori = ' '.join(sent_ori_list[k])
            #print('sentence:',sentence)
            result = self.dic_trie.match(sentence)
            #print('result:',result)
            new_result = []
            for i in range(0, len(result)):
                if result[i][0] == 0 and sentence[result[i][1]] == " ":
                    new_result.append([result[i][0], result[i][0] + result[i][1]])
                elif result[i][0] > 0 and sentence[result[i][0] - 1] == ' ' and sentence[
                    result[i][0] + result[i][1]] == ' ':
                    new_result.append([result[i][0], result[i][0] + result[i][1]])
            #print('new result:',new_result)

            if len(new_result) == 0:
                fout.write(sentence_ori + '\n\n')

            else:
                fout.write(sentence_ori + '\n')
                for ele in new_result:
                    entity_text = sentence[ele[0]:ele[1]]
                    if entity_text in self.word_hpo.keys():
                        hpoid = self.word_hpo[entity_text]
                    else:
                        print('no id:', entity_text)
                        hpoid = ['None']
                    if ele[0] == 0:
                        sid = "0"
                    else:
                        temp_sent = sentence[0:ele[0]]
                        sid = str(len(temp_sent.rstrip().split(' ')))
                    temp_sent = sentence[0:ele[1]]
                    eid = str(len(temp_sent.rstrip().split(' ')) - 1)
                    #                    print(sid,eid,entity_text,hpoid[0])
                    fout.write(sid + '\t' + eid + '\t' + entity_text + '\t' + ";".join(hpoid) + '\t1.00\n')
                fout.write('\n')

        return fout.getvalue()

def get_oripos(data_split):
    original_pos = []
    for data_ in data_split:
        sen_pos = []
        sen_str = ""
        split = data_.strip().split("\t")
        split_line = split[1].replace('-', ' - ').replace('/', ' / ')
        sentences = nltk.sent_tokenize(split_line)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        for sent in sentences:
            sen_str = sen_str + " ".join(sent)+" "
            sen_str = sen_str + " "

        for i in range(2,len(split)-1):
            if(split[i+1].startswith("HP")):
                hp_split = nltk.word_tokenize(split[i])
                hp_str = " ".join(hp_split)

                s_ind = sen_str.find(hp_str)
                if(s_ind!=-1):
                    e_ind = s_ind+len(hp_str)

                    if s_ind == 0:
                       sid = 0
                    else:
                       temp_sent = sen_str[0:s_ind]
                       sid = len(temp_sent.rstrip().split(' '))
                    temp_sent = sen_str[0:e_ind]
                    eid = len(temp_sent.rstrip().split(' ')) - 1
                    sen_pos.append((sid,eid))

        original_pos.append(sen_pos)

    return original_pos

def change_label():
    str = ""
    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOB/train_2019.tsv", "r", encoding="utf-8") as f:
        data = f.read()

        train_data = data.split("\n\n")
        train_data = [token.split("\n") for token in train_data]
        train_data = [[j.split() for j in i] for i in train_data]
        train_data.pop()

        for sen_word in train_data:
            for pos in range(len(sen_word)):  # 遍历序列中的每个位置
                curr_entity = sen_word[pos][-1]
                if (curr_entity[-4:] == "Gene"):
                    sen_word[pos][-1] = "O"
                elif pos == len(sen_word) - 1:  # 序列的最后一个位置
                    if curr_entity.startswith("B-"):
                        sen_word[pos][-1] = curr_entity.replace("B-", "S-")
                    elif curr_entity.startswith("I-"):
                        sen_word[pos][-1] = curr_entity.replace("I-", "E-")
                else:
                    print(sen_word[pos+1])
                    next_entity = sen_word[pos + 1][-1]
                    if curr_entity.startswith("B-"):
                        if next_entity.startswith("O") or next_entity.startswith("B-"):
                            sen_word[pos][-1] = curr_entity.replace("B-", "S-")
                    elif curr_entity.startswith("I-"):
                        if next_entity.startswith("O") or next_entity.startswith("B-"):
                            sen_word[pos][-1] = curr_entity.replace("I-", "E-")
    # print(str)
    vocab = ""
    for sen_word in train_data:
        for word in sen_word:
            vocab = vocab + word[0] + "\t" + word[-1] + "\n"
        vocab = vocab + "\n"

    with open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOBES/train_2019.tsv", "w", encoding="utf-8") as f:
        f.write(vocab)


if __name__ == '__main__':
    ssplit_tokens = []
    dic_results = []
    ssen_label = []
    new_pos = []
    ontfiles = {'dic_file': '../dict/hpo_noabb_lemma.dic',
                'word_hpo_file': '../dict/word_hpoid_map.json',
                'hpo_word_file': '../dict/hpoid_word_map.json'}
    biotag_dic = dic_ont(ontfiles)
    #text = 'Nevoid 11.5 basal cell carcinoma syndrome (NBCCS) is a hereditary condition transmitted as an autosomal dominant trait with complete penetrance and variable expressivity. The syndrome is characterised by numerous basal cell carcinomas (BCCs), odontogenic keratocysts of the jaws, palmar and/or plantar pits, skeletal abnormalities and intracranial calcifications. In this paper, the clinical features of 37 Italian patients are reviewed. Jaw cysts and calcification of falx cerebri were the most frequently observed anomalies, followed by BCCs and palmar/plantar pits. Similar to the case of African Americans, the relatively low frequency of BCCs in the Italian population is probably due to protective skin pigmentation. A future search based on mutation screening might establish a possible genotype phenotype correlation in Italian patients.'
    fin = open("D:\DeepLearning\DeepLearning_Code\PGR\corpora/11_03_2019_corpus/train_ner.tsv", 'r', encoding='utf-8')
    data = fin.read()
    data_split = data.split("\n")
    data_split.pop()
    sentences = [data_.strip().split("\t")[1] for data_ in data_split]

    original_pos = get_oripos(data_split)

    print(len(original_pos))

    for sentence,sen_pos in zip(sentences,original_pos):
        #print(sen_pos)
        dic_pos = []

        ssplit_token = ssplit_token_pos_lemma(sentence)
        ssplit_tokens.append(ssplit_token)
        ssplit = ssplit_token.rstrip().split("\n")
        #print(ssplit[sen_pos[0][0]],ssplit[sen_pos[0][1]])

        dic_result = biotag_dic.matching(ssplit_token)
        dic_ressplit = dic_result.split("\n")
        #print(dic_ressplit)
        pre = 0
        for i in range(1,len(dic_ressplit)-2):
            if(dic_ressplit[i]==''):
                pre = pre + len(dic_ressplit[i-1].split(" ")) + 1
                continue
            elif(dic_ressplit[i].endswith("1.00")):
                entity_inf = dic_ressplit[i].split("\t")
                #print(i,entity_inf)
                dic_pos.append((int(entity_inf[0])+pre,int(entity_inf[1])+pre))

        sen_pos.extend(dic_pos)
        sen_new_pos = sorted(sen_pos,key=lambda pos:pos[0])
        new_pos.append(sen_new_pos)
        dic_results.append(dic_result)
        #print(dic_results)

    for ssplit_token,sen_pos in zip(ssplit_tokens,new_pos):
        ssplit = ssplit_token.rstrip().split("\n")
        sen_label = ["O"]*len(ssplit)
        pre_eind = -1
        for word_pos in sen_pos:
            print(word_pos)
            if (word_pos[0]>pre_eind):
                sen_label[word_pos[0]] = "B-Phenotype"
                for i in range(word_pos[0]+1,word_pos[1]+1):
                    sen_label[i] = "I-Phenotype"
                pre_eind = word_pos[1]

        ssen_label.append(sen_label)
        #print(sen_label)

    vocab = ""
    for ssplit_token,sen_label in zip(ssplit_tokens,ssen_label):
        ssplit = ssplit_token.rstrip().split("\n")
        for word_split,word_label in zip(ssplit,sen_label):
            if(word_split==''):
                vocab = vocab +"\n"
                continue
            word_split_label = word_split+"\t"+word_label
            vocab = vocab + word_split_label +"\n"
        vocab = vocab + "\n"
    #print(vocab)

    fout = open("D:/DeepLearning/Human_Phenotype/NEW_HPO_IOB/train_2019.tsv", 'w', encoding='utf-8')
    fout.write(vocab)
    change_label()
    #print(ssen_label)




