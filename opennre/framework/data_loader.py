import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from opennre.dataset.utils import is_normal, is_multi_label, is_over_lapping

NA = ["NA", "Other", "None"]


# class SentenceREDataset(data.Dataset):
#     """
#     Sentence-level relation extraction dataset
#     """

#     def __init__(self, path, rel2id, tokenizer, kwargs, sort=False, sort_reverse=False):
#         """
#         Args:
#             path: path of the input file
#             rel2id: dictionary of relation->id mapping
#             tokenizer: function of tokenizing
#         """
#         super().__init__()
#         self.path = path
#         self.tokenizer = tokenizer
#         self.rel2id = rel2id
#         self.kwargs = kwargs
#         self.sort = sort

#         # Load the file
#         self.data = []
#         self.NA_id = -1
#         for r in rel2id:
#             if r in NA:
#                 self.NA_id = rel2id[r]
#         self.load(sort, sort_reverse)

#     def load(self, sort, sort_reverse):
#         f = open(self.path)
#         sentense_idx = [{} for _ in range(1000)]

#         for i,line in enumerate(f.readlines()):
#             line = line.rstrip()
#             if len(line) > 0:
#                 print(line)
#                 _d = json.loads(line)
#                 if "token" not in _d:
#                     _d["token"] = _d["text"].split(" ")
#                 if len(_d["token"]) > 512:
#                     continue
#                 if " ".join(_d["token"]) in sentense_idx[len(_d["token"])]:
#                     idx = sentense_idx[len(_d["token"])][" ".join(_d["token"])]
#                     entity_ids = [-1, -1]
#                     for i, k in enumerate(["h", "t"]):
#                         try:
#                             entity_ids[i] = self.data[idx]["entity_list"].index(_d[k])
#                         except ValueError:
#                             self.data[idx]["entity_list"].append(_d[k])
#                             entity_ids[i] = len(self.data[idx]["entity_list"]) - 1
#                     assert min(entity_ids) > -1
#                     self.data[idx]["entity_pair_list"].append([entity_ids[0], entity_ids[1], self.rel2id[_d['relation']]])
#                 else:
#                     _d["entity_list"] = [_d["h"], _d["t"]]
#                     _d["entity_pair_list"] = [[0, 1, self.rel2id[_d['relation']]]]
#                     del _d["h"]
#                     del _d["t"]
#                     indexed_tokens, att_mask, new_index = self.tokenizer(_d)
#                     _d["indexed_tokens"] = indexed_tokens
#                     _d["att_mask"] = att_mask
#                     _d["new_index"] = new_index
#                     _d["seq_len"] = indexed_tokens.shape[-1]
#                     if indexed_tokens.shape[-1] <= 512:
#                         self.data.append(_d)
#                         sentense_idx[len(_d["token"])][" ".join(_d["token"])] = len(self.data) - 1
#         if sort:
#             self.data.sort(key=lambda x: x["seq_len"], reverse=sort_reverse)
#         else:
#             random.shuffle(self.data)
#         f.close()

#     def set_max_words(self, w_max=100):
#         self.data = [d for d in self.data if len(d["token"]) <= w_max]

#     def remove_na(self):
#         new_list = []
#         for d in self.data:
#             d["entity_pair_list"] = [tuple(ep) for ep in d["entity_pair_list"] if ep[2] != self.NA_id]

#             if len(d["entity_pair_list"]) > 0:
#                 new_list.append(d)
#         self.data = new_list

#     def remove_repeat(self):
#         for d in self.data:
#             epl = d["entity_pair_list"]
#             epl = list(set([tuple(ep) for ep in epl]))
#             epl = [list(ep) for ep in epl]
#             d["entity_pair_list"] = epl

#     def char_idx_to_word_idx(self):
#         for d in self.data:
#             idx = []
#             word_i = 0

#             for i, c in enumerate(d["text"]):
#                 idx.append(word_i)
#                 if c == " ":
#                     word_i += 1
#             idx.append(word_i)
#             for e in d["entity_list"]:
#                 e["pos"][0] = idx[e["pos"][0]]
#                 e["pos"][1] = idx[e["pos"][1]] + 1

#     def split(self):
#         # pass
#         new_data = []
#         for d in self.data:
#             if len(d["entity_pair_list"]) == 1:
#                 new_data.append(d)
#             else:
#                 for i in range(len(d["entity_pair_list"])):
#                     d1 = d.copy()
#                     d1["entity_pair_list"] = [d["entity_pair_list"][i]]
#                     new_data.append(d1)

#         self.data = new_data
#         random.shuffle(self.data)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]

#         seq_len = item["indexed_tokens"].shape[-1]
#         predicate_one_hot_labels = torch.zeros((1, len(self.rel2id), seq_len, seq_len))
#         # item["entity_list"][0]["test"] = "test"

#         for epl in item["entity_pair_list"]:

#             predicate_one_hot_labels = self.merge_ont_hot_labels(predicate_one_hot_labels, item["entity_list"][epl[0]],
#                                                                  item["entity_list"][epl[1]], epl[2], item["new_index"],
#                                                                  seq_len)
#         if predicate_one_hot_labels.max().item() > 1:
#             predicate_one_hot_labels = (predicate_one_hot_labels > 0).float()

#         em = SentenceREDataset.get_entity_mask(item["entity_list"][epl[0]], item["entity_list"][epl[1]],
#                                                item["new_index"], seq_len)
#         token_labels = em.sum(dim=0) + em.sum(dim=1)
#         token_labels = (token_labels > 0).long()

#         return item["indexed_tokens"], item["att_mask"], predicate_one_hot_labels, token_labels

#     def merge_ont_hot_labels(self, predicate_one_hot_labels, pos_head, pos_tail, rel_id, new_index, seq_len):
#         em = SentenceREDataset.get_entity_mask(pos_head, pos_tail, new_index, seq_len)

#         predicate_one_hot_labels[0][rel_id] += em
#         return predicate_one_hot_labels

#     @staticmethod
#     def get_entity_mask(h, t, new_index, seq_len):
#         pos_head = h['pos']
#         pos_tail = t['pos']
#         pos_head = [new_index.index(i) for i in pos_head]
#         pos_tail = [new_index.index(i) for i in pos_tail]

#         entity_1 = torch.zeros((seq_len))
#         entity_1[pos_head[0]:pos_head[1]] = 1
#         entity_2 = torch.zeros((seq_len))
#         entity_2[pos_tail[0]:pos_tail[1]] = 1

#         res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
#         res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
#         res = (res == 2).float()
#         return res

#     @staticmethod
#     def collate_fn(data):
#         data = list(zip(*data))

#         subject_labels = data[-1]
#         batch_subject_labels = pad_sequence(subject_labels, batch_first=True, padding_value=0)  # (B)

#         labels = list(data[-2])
#         seq_len = batch_subject_labels.shape[-1]
#         for i in range(len(labels)):
#             concat_shape = list(labels[i].shape)
#             concat_shape[-1] = seq_len - concat_shape[-1]

#             labels[i] = torch.cat([labels[i], torch.zeros(concat_shape)], dim=-1)

#             concat_shape = list(labels[i].shape)
#             concat_shape[-2] = seq_len - concat_shape[-2]

#             labels[i] = torch.cat([labels[i], torch.zeros(concat_shape)], dim=-2)
#         batch_labels = torch.cat(labels, 0)  # (B)

#         seqs = data[0:2]
#         batch_seqs = []
#         for seq in seqs:
#             seq = list(seq)
#             for i in range(len(seq)):
#                 seq[i] = torch.cat([seq[i], torch.zeros((1, seq_len - seq[i].shape[-1]), dtype=seq[i].dtype)], dim=-1)
#             batch_seqs.append(torch.cat(seq, dim=0))
#             # batch_seqs.append(torch.cat(seq, 0))  # (B, L)
#         return batch_seqs + [batch_labels, batch_subject_labels]

#     def information(self):
#         normal_count = 0
#         multi_label_count = 0
#         over_lapping_count = 0

#         triples_num_count = [0] * 4
#         triples_count = 0
#         NA_count = 0

#         for d in self.data:
#             epl = d["entity_pair_list"].copy()
#             NA_count += len(epl)
#             epl = [ep for ep in epl if ep[2] != self.NA_id]
#             NA_count -= len(epl)
#             triples_count += len(epl)
#             normal_count += 1 if is_normal(epl) and len(epl) > 0 else 0
#             multi_label_count += 1 if is_multi_label(epl) else 0
#             over_lapping_count += 1 if is_over_lapping(epl) else 0
#             count = len(epl) if len(epl) < 3 else 3
#             triples_num_count[count] += 1
#         print("data: %s\nnormal_count: %d\nmulti_label_count: %d\nover_lapping_count: %d" % (self.path, normal_count, multi_label_count, over_lapping_count))
#         print("triples_count :")
#         for i, tc in enumerate(triples_num_count):
#             print("%d : %d" % (i, tc))
#         print("NA_count : %d" % NA_count)
#         print("data len %d " % len(self.data))
#         print("triples count %d " % triples_count)

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.rel2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        return [self.rel2id[item['relation']]] + seq # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        goldens = []
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
                goldens.append(golden)
            else:
                golden = self.rel2id[self.data[i]['relation']]
                goldens.append(golden)
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)

        labels = list(set(goldens))
        #Micro
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        # Macro
        try:
            macro_p = float(correct_positive) / float(pred_positive)
        except:
            macro_p = 0
        try:
            macro_r = float(correct_positive) / float(gold_positive)
        except:
            macro_r = 0
        try:
            macro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            macro_f1 = 0
        #result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1}
        result = metrics.classification_report(goldens, pred_result, labels=labels)
        logging.info('Evaluation result: \n {}.'.format(result))
        return result
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """
    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass
  
    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag
            
        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag)] + seqs
  
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
            seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L)
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

  
    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc}

def BagRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, entpair_as_bag=False, bag_size=0, num_workers=8, 
        collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
