"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)
 
.. moduleauthor:: Liyuan Liu, Frank Xu
"""


import torch
import numpy as np
import itertools
from torch.autograd import Variable

import models.utils as utils
from models.crf import CRFDecode_vb

class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy 

    args: 
        packer: provide method to convert target into original space [TODO: need to improve] 
        l_map: dictionary for labels    
    """
   

    def __init__(self, packer, l_map):
        self.packer = packer
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        """
        update statics for f1 score

        args: 
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        str1=""

        #print('decode_data: ', decoded_data.size(), ' target_data: ', target_data.size())
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            best_path = decoded[:length]
            #print('gold:', gold.size(), ' best_path: ', best_path.size(), ' length: ', length)
            for i in range(len(best_path.numpy())):
                str1 = str1 + str(best_path[i]) + "\n"
            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.numpy(), gold.numpy())
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i


    def calc_acc_batch(self, decoded_data, target_data):
        """
        update statics for accuracy

        args: 
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)   # micro-F1
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):
        """
        update statics for one instance

        args: 
            best_path (seq_len): predicted
            gold (seq_len): ground-truth
        """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = utils.iobes_to_spans(gold, self.r_l_map)
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes_to_spans(best_path, self.r_l_map)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """
   
    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)
        
        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()
                
        for feature, tg, mask in itertools.chain.from_iterable(dataset_loader):
            fea_v, _, mask_v = self.packer.repack_vb(feature, tg, mask)
            scores, _ = ner_model(fea_v)
            decoded = self.decoder.decode(scores.data, mask_v.data)

            self.eval_b(decoded, tg)

        return self.calc_s()

class eval_wc(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """
   
    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)
        
        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])
        self.l_map = l_map
        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, file_no):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v,s_t,mask1_v,e_t,mask2_v,bert_f in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v,s_t,mask1_v,e_t,mask2_v,bert_f = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v,s_t,mask1_v,e_t,mask2_v,bert_f)
            scores,binclass_scores,all_attention = ner_model(f_f, f_p, b_f, b_p, w_f, bert_f,file_no)
            # print("size scores:", scores.data.size(), " size mask:", mask_v.data.size())
            decoded = self.decoder.decode(scores.data, mask_v.data)

            #decodedPN = self.decoder.decodePN(binclass_scores,all_attention,mask1_v,self.l_map)
            '''
            decoded = decoded.permute(1,0).numpy().tolist()
            decodedPN = decodedPN.permute(1,0).numpy().tolist()
            new_decode = []
            for batch_de,batch_dePN in zip(decoded,decodedPN):
                length = utils.find_length_from_labels(batch_de, self.l_map)
                seq_len = len(batch_de)
                batch_de = batch_de[:length]
                batch_dePN = batch_dePN[:length]
                newbatch_de = []
                startIdx = -1
                flag = False
                for token,token_PN in zip(enumerate(batch_de),batch_dePN):
                    if(token[1]!=token_PN):
                        if(token[1]==self.l_map['O'] and token_PN==self.l_map['S-Phenotype']):
                            newbatch_de.append(token_PN)
                        elif(token[1]==self.l_map['E-Phenotype'] and token_PN==self.l_map['I-Phenotype']):
                            newbatch_de.append(token[1])
                            flag = True
                        else:
                            newbatch_de.append(token[1])
                    else:
                        if(token[1]==self.l_map['B-Phenotype'] and token_PN==self.l_map['B-Phenotype']):
                            startIdx = token[0]
                            newbatch_de.append(token[1])
                        elif(flag):
                            if(token[1]==self.l_map['E-Phenotype'] and token_PN==self.l_map['E-Phenotype'] and startIdx!=-1):
                                newbatch_de = newbatch_de[:startIdx]

                                for t in range(startIdx,token[0]+1):
                                    newbatch_de.append(batch_dePN[t])
                                startIdx=-1
                                flag = False
                            else:
                                newbatch_de.append(token[1])
                        else:
                            newbatch_de.append(token[1])

                assert len(newbatch_de)==len(batch_de)

                for i in range(len(newbatch_de),seq_len):
                    newbatch_de.append(self.l_map["<pad>"])

                new_decode.append(newbatch_de)

            new_decode = torch.Tensor(new_decode).permute(1,0) 
            #self.eval_b(decoded, tg)
            '''
            self.eval_b(decoded , tg)

        return self.calc_s()
