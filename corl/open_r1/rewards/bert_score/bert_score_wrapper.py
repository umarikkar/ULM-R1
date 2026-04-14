import os
from transformers import AutoTokenizer, AutoConfig, MPNetModel, BertModel
import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import util


class BertScoreWrapper:
    def __init__(
            self, model_ckpt="sentence-transformers/all-mpnet-base-v2", device="cuda",
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        # self.model = AutoModel.from_pretrained(model_ckpt)
        config = AutoConfig.from_pretrained(model_ckpt)
        self.model = MPNetModel(config)

        checkpoint = torch.load(os.path.join(model_ckpt, 'pytorch_model.bin'), map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(device)
        self.model = self.model.eval()

    def get_embeddings(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)  #, output_hidden_states=True
            embeddings = outputs.last_hidden_state  # [bs, n_seq, 768]
        return embeddings, inputs['attention_mask']

    @staticmethod
    def compute_similarity_matrix(pred_embeddings, ref_embeddings, pred_mask, ref_mask):
        batch_size = pred_embeddings.size(0)
        similarities = []
        for i in range(batch_size):
            pred_valid_len = pred_mask[i].sum().item()
            ref_valid_len = ref_mask[i].sum().item()
            pred_emb = pred_embeddings[i, :pred_valid_len, :]  # [pred_len, hidden_size]
            ref_emb = ref_embeddings[i, :ref_valid_len, :]
            pred_norm = F.normalize(pred_emb, p=2, dim=1)
            ref_norm = F.normalize(ref_emb, p=2, dim=1)
            # [pred_len, ref_len]
            sim_matrix = torch.mm(pred_norm, ref_norm.transpose(0, 1))
            similarities.append(sim_matrix)
        return similarities

    def compute_f1(self, predictions, references):
        pred_embeddings, pred_mask = self.get_embeddings(predictions)
        ref_embeddings, ref_mask = self.get_embeddings(references)

        similarity_matrices = self.compute_similarity_matrix(
            pred_embeddings, ref_embeddings, pred_mask, ref_mask
        )

        # precision_list = []
        # recall_list = []
        f1_list = []
        for sim_matrix in similarity_matrices:
            # Precision: 对于预测句子中的每个词，找到参考句子中最相似的词
            precision_scores = sim_matrix.max(dim=1)[0]  # [pred_len]
            precision = precision_scores.mean().item()

            # Recall: 对于参考句子中的每个词，找到预测句子中最相似的词
            recall_scores = sim_matrix.max(dim=0)[0]  # [ref_len]
            recall = recall_scores.mean().item()

            # F1 Score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # precision_list.append(precision)
            # recall_list.append(recall)
            f1_list.append(f1)

        return f1_list


class BertSimCSEWrapper:
    def __init__(self, model_ckpt="princeton-nlp/sup-simcse-bert-base-uncased", device="cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        config = AutoConfig.from_pretrained(model_ckpt)
        self.model = BertModel(config)

        checkpoint = torch.load(os.path.join(model_ckpt, 'pytorch_model.bin'), map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(device)
        self.model = self.model.eval()

    def get_embeddings(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)  #, output_hidden_states=True
            embeddings = outputs.last_hidden_state  # [bs, n_seq, 768]
        return embeddings, inputs['attention_mask']

    @staticmethod
    def mean_pooling(embeddings, attention_mask):
        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            embeddings.size()).float()
        sentence_embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)  # [bs, 768]
        return sentence_embeddings

    def compute_simcse(self, predictions, references):
        pred_embeddings, pred_mask = self.get_embeddings(predictions)
        ref_embeddings, ref_mask = self.get_embeddings(references)

        sent_pred_embeddings = self.mean_pooling(pred_embeddings, pred_mask)
        sent_ref_embeddings = self.mean_pooling(ref_embeddings, ref_mask)

        sims = [util.cos_sim(p, r).item() for p, r in zip(sent_pred_embeddings, sent_ref_embeddings)]
        return sims
