# -*- coding: utf-8 -*-
# ==========================================================
# Assignment - Sanskrit -> English Neural Machine Translation
# ==========================================================

# ========================Installation and Execution Steps==================================
# Step 1: Open Google Colab,Create a new notebook, Upload this .ipynb notebook file.
# Step 2: Enable GPU Runtime (Edit -> Notebook Settings -> Hardware Accelerator -> GPU)
# Step 3: Install Required Libraries
# Step 4: Upload the dataset files (train_sa_10000.csv, train_en_10000.csv, dev_sa_1000.csv, dev_en_1000.csv, test_sa_1000.csv, test_en_1000.csv) to the Colab environment.
# Step 5: Run the notebook cells sequentially to train the model and generate translations.
# Step 6: The final output will be saved as "submission.csv" containing the translated English sentences.
# =============================================================================================


# Sanskrit -> English Neural Machine Translation
# ==========================================================

!pip install bert-score nltk -q

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import corpus_bleu
from bert_score import score

import random
import time

##################################################
#GPU
###################################################

import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
#############################################

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", DEVICE)

# ==========================================================
# LOAD DATA
# ==========================================================

train_sa = pd.read_csv("train_sa_10000.csv")
train_en = pd.read_csv("train_en_10000.csv")

dev_sa = pd.read_csv("dev_sa_1000.csv")
dev_en = pd.read_csv("dev_en_1000.csv")

test_sa = pd.read_csv("test_sa_1000.csv")
test_en = pd.read_csv("test_en_1000.csv")

train_df = pd.DataFrame({
    "sa": train_sa["Sentence_sa"],
    "en": train_en["Sentence_en"]
})

dev_df = pd.DataFrame({
    "sa": dev_sa["Sentence_sa"],
    "en": dev_en["Sentence_en"]
})

test_df = pd.DataFrame({
    "sa": test_sa["Sentence_sa"],
    "en": test_en["Sentence_en"]
})

print("Training Samples:", len(train_df))
print("Validation Samples:", len(dev_df))
print("Test Samples:", len(test_df))

# ==========================================================
# VOCABULARY
# ==========================================================

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_vocab(texts):

    counter = Counter()

    for sentence in texts:
        counter.update(str(sentence).split())

    vocab = SPECIALS + list(counter.keys())

    stoi = {token:i for i,token in enumerate(vocab)}
    itos = {i:token for i,token in enumerate(vocab)}

    return stoi, itos


src_stoi, src_itos = build_vocab(train_df["sa"])
tgt_stoi, tgt_itos = build_vocab(train_df["en"])

PAD_IDX = src_stoi["<pad>"]
SOS_IDX = tgt_stoi["<sos>"]
EOS_IDX = tgt_stoi["<eos>"]
UNK_IDX = tgt_stoi["<unk>"]

# ==========================================================
# DATASET
# ==========================================================

class TranslationDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        source = str(self.df.iloc[idx]["sa"]).split()

        target = str(self.df.iloc[idx]["en"]).split()

        source_ids = [
            src_stoi.get(word, UNK_IDX)
            for word in source
        ]

        target_ids = (
            [SOS_IDX]
            + [tgt_stoi.get(word, UNK_IDX) for word in target]
            + [EOS_IDX]
        )

        return (
            torch.tensor(source_ids),
            torch.tensor(target_ids)
        )


def collate_fn(batch):

    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return src_batch, tgt_batch


train_loader = DataLoader(
    TranslationDataset(train_df),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    TranslationDataset(dev_df),
    batch_size=128,
    shuffle=False,
    collate_fn=collate_fn
)

# ==========================================================
# Encoder converts the input Sanskrit sentence into numerical
# representations that capture the meaning of the sentence.
# A bidirectional GRU is used so that information from both
# left and right context can be learned.
# ==========================================================

class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            emb_dim,
            padding_idx=PAD_IDX
        )

        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            hid_dim * 2,
            hid_dim
        )

    def forward(self, src):

        embedded = self.embedding(src)

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(
            self.fc(
                torch.cat(
                    (hidden[-2], hidden[-1]),
                    dim=1
                )
            )
        )

        return outputs, hidden

# ==========================================================
# Attention helps the decoder focus on the most relevant
# source words while generating each English word.
# Instead of relying only on the final encoder state,
# the decoder can look back at important parts of the
# Sanskrit sentence whenever needed.
# ==========================================================

class Attention(nn.Module):

    def __init__(self, hid_dim):

        super().__init__()

        self.attn = nn.Linear(
            hid_dim * 3,
            hid_dim
        )

        self.v = nn.Linear(
            hid_dim,
            1,
            bias=False
        )

    def forward(self, hidden, encoder_outputs):

        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(
            1,
            src_len,
            1
        )

        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (
                        hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )

        attention = self.v(
            energy
        ).squeeze(2)

        return torch.softmax(
            attention,
            dim=1
        )

# ==========================================================
# Decoder generates the English translation one word at a time.
# At every step it receives the previous predicted word,
# the current hidden state and the attention context
# coming from the encoder outputs.
# ==========================================================

class Decoder(nn.Module):

    def __init__(
            self,
            output_dim,
            emb_dim,
            hid_dim,
            attention):

        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            emb_dim
        )

        self.attention = attention

        self.rnn = nn.GRU(
            emb_dim + hid_dim*2,
            hid_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hid_dim*3 + emb_dim,
            output_dim
        )

    def forward(
            self,
            input_token,
            hidden,
            encoder_outputs):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(
            input_token
        )

        a = self.attention(
            hidden,
            encoder_outputs
        )

        a = a.unsqueeze(1)

        context = torch.bmm(
            a,
            encoder_outputs
        )

        rnn_input = torch.cat(
            (embedded, context),
            dim=2
        )

        output, hidden = self.rnn(
            rnn_input,
            hidden.unsqueeze(0)
        )

        prediction = self.fc(
            torch.cat(
                (
                    output.squeeze(1),
                    context.squeeze(1),
                    embedded.squeeze(1)
                ),
                dim=1
            )
        )

        return prediction, hidden.squeeze(0)

# ==========================================================
# This class combines the encoder and decoder into a single
# translation model. The encoder processes the Sanskrit input
# sentence and the decoder uses that information to generate
# the corresponding English sentence.
# ==========================================================

class Seq2Seq(nn.Module):

    def __init__(
            self,
            encoder,
            decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            src,
            trg,
            teacher_forcing_ratio=0.5):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            output_dim
        ).to(DEVICE)

        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[:,0]

        for t in range(1, trg_len):

            output, hidden = self.decoder(
                input_token,
                hidden,
                encoder_outputs
            )

            outputs[:,t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input_token = (
                trg[:,t]
                if teacher_force
                else top1
            )

        return outputs

# ==========================================================
# MODEL
# ==========================================================

INPUT_DIM = len(src_stoi)
OUTPUT_DIM = len(tgt_stoi)

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256

attention = Attention(HID_DIM)

encoder = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    HID_DIM
)
decoder = Decoder(
    OUTPUT_DIM,
    DEC_EMB_DIM,
    HID_DIM,
    attention
)

model = Seq2Seq(
    encoder,
    decoder
).to(DEVICE)

# ==========================================================
# PARAMETER COUNT
# ==========================================================

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable Parameters:", params)

# ==========================================================
# Training is performed using teacher forcing.
# During training, the correct previous word is used
# some of the time to help the decoder learn faster.
# Cross-entropy loss is used to compare predicted words
# with the actual English translations.
# ==========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)

EPOCHS = 15


for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for src, tgt in train_loader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        output = model(src, tgt)

        output_dim = output.shape[-1]

        output = output[:,1:].reshape(
            -1,
            output_dim
        )

        tgt = tgt[:,1:].reshape(-1)

        loss = criterion(
            output,
            tgt
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} Loss={epoch_loss:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "nmt_model.pth"
)

print("Model Saved")

# ==========================================================
# BEAM SEARCH DECODING
# ==========================================================
# Decoding Strategy:
# This function generates an English translation using Beam Search Decoding.
#
# During inference, instead of selecting only the highest probability word
# at each step, multiple candidate translations are maintained.
#
# Beam width = 3 means the decoder keeps track of the 3 most promising
# translation paths while generating the output sentence.
#
# At every decoding step:
#   - The decoder predicts probabilities for the next word.
#   - The top candidate words are selected.
#   - New translation paths are generated.
#   - Paths are scored using cumulative log probabilities.
#
# The highest-scoring sequence is selected as the final translation.
#
# This generally produces more fluent translations than simple
# greedy decoding.
# ============================================================================

def beam_search_translate(
        sentence,
        beam_width=3,
        max_len=60):

    model.eval()

    tokens = str(sentence).split()

    src_ids = [
        src_stoi.get(word, UNK_IDX)
        for word in tokens
    ]

    src_tensor = torch.LongTensor(
        src_ids
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        encoder_outputs, hidden = model.encoder(
            src_tensor
        )

    beams = [
        (
            [SOS_IDX],
            hidden,
            0.0
        )
    ]

    for _ in range(max_len):

        new_beams = []

        for seq, hidden_state, score_value in beams:

            last_token = seq[-1]

            if last_token == EOS_IDX:

                new_beams.append(
                    (
                        seq,
                        hidden_state,
                        score_value
                    )
                )

                continue

            current_token = torch.LongTensor(
                [last_token]
            ).to(DEVICE)

            with torch.no_grad():

                prediction, new_hidden = model.decoder(
                    current_token,
                    hidden_state,
                    encoder_outputs
                )

            log_probs = torch.log_softmax(
                prediction,
                dim=1
            )

            top_probs, top_tokens = torch.topk(
                log_probs,
                beam_width
            )

            for k in range(beam_width):

                token = top_tokens[0][k].item()

                candidate_score = (
                    score_value +
                    top_probs[0][k].item()
                )

                candidate_sequence = (
                    seq + [token]
                )

                new_beams.append(
                    (
                        candidate_sequence,
                        new_hidden,
                        candidate_score
                    )
                )

        beams = sorted(
            new_beams,
            key=lambda x: x[2],
            reverse=True
        )[:beam_width]

        if all(
            beam[0][-1] == EOS_IDX
            for beam in beams
        ):
            break

    best_sequence = beams[0][0]

    translated_words = []

    for idx in best_sequence[1:]:

        if idx == EOS_IDX:
            break

        word = tgt_itos.get(
            idx,
            "<unk>"
        )

        if word not in [
            "<sos>",
            "<eos>",
            "<pad>"
        ]:
            translated_words.append(
                word
            )

    return " ".join(translated_words)

# ==========================================================
# EVALUATION
# ==========================================================

start = time.time()

predictions = []

for sent in test_df["sa"]:

    predictions.append(
        beam_search_translate(
            sent,
            beam_width=3
        )
    )

end = time.time()

print(
    "Inference Time:",
    end-start,
    "seconds"
)

# ==========================================================
# BLEU
# BLEU score is used to compare generated translations
# with the reference English sentences. A higher BLEU
# score generally indicates better translation quality.
# ==========================================================

references = test_df["en"].tolist()

bleu = corpus_bleu(
    [[r.split()] for r in references],
    [p.split() for p in predictions]
)

print("BLEU:", bleu)

# ==========================================================
# BERTSCORE
# BERTScore measures semantic similarity between the
# generated translation and the reference sentence.
# Unlike BLEU, it can capture meaning even when the
# exact wording is different.
# ==========================================================

P,R,F1 = score(
    predictions,
    references,
    lang="en",
    rescale_with_baseline=True
)

print(
    "BERTScore F1:",
    F1.mean().item()
)

# ==========================================================
# SUBMISSION
# ==========================================================

submission = pd.DataFrame(
    {
        "Source_id":
        test_sa["Source_id"],

        "Sentence_en":
        predictions
    }
)

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv created")

