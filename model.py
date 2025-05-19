import torch
import torch.nn as nn
import os 
import logging
from datetime import datetime

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'model_logs/{timestamp}.txt'

# Set up logging with the timestamped filename
logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers=1, cell_type="LSTM"):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        rnn_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_class(embed_size, hidden_size, num_layers, batch_first=True)
        self.cell_type = cell_type

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_layers=1, cell_type="LSTM"):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        rnn_class = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_class(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.cell_type = cell_type

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        predictions = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type="LSTM"):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size, target_len = target.size()
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)
        hidden = self.encoder(source)

        input = target[:, 0].unsqueeze(1)

        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs
    
def build_vocab(filepaths):
    chars = set()
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                native, roman, _ = line.strip().split("\t")
                chars.update(native)
                chars.update(roman)
    return chars

def make_char2idx(char_set):
    char_list = ["<pad>", "<sos>", "<eos>", "<unk>"] + sorted(list(char_set))
    return {char: idx for idx, char in enumerate(char_list)}, char_list

train_path = "/mnt/e_disk/ch24s016/da6401_assignment3/dataset/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
dev_path = "/mnt/e_disk/ch24s016/da6401_assignment3/dataset/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"

char_set = build_vocab([train_path, dev_path])
roman2idx, idx2roman = make_char2idx(set(c for c in char_set if c.isascii()))
devanagari2idx, idx2devanagari = make_char2idx(set(c for c in char_set if not c.isascii()))

from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=roman2idx["<pad>"], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=devanagari2idx["<pad>"], batch_first=True)
    return src_batch, tgt_batch


import torch
from torch.utils.data import Dataset, DataLoader

class TransliterationDataset(Dataset):
    def __init__(self, tsv_path, src_char2idx, tgt_char2idx, max_len=32):
        self.pairs = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                native, roman, _ = line.strip().split('\t')
                self.pairs.append((roman, native))

        self.src_c2i = src_char2idx
        self.tgt_c2i = tgt_char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        roman, native = self.pairs[i]

        # map chars → indices, add <sos> / <eos> tokens as needed
        src_idxs = [self.src_c2i.get(c, self.src_c2i["<unk>"]) 
                    for c in roman][: self.max_len]
        tgt_idxs = [self.tgt_c2i["<sos>"]] + \
                   [self.tgt_c2i.get(c, self.tgt_c2i["<unk>"]) 
                    for c in native][: (self.max_len-1)] + \
                   [self.tgt_c2i["<eos>"]]

        return torch.tensor(src_idxs), torch.tensor(tgt_idxs)


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)

    src_max_len = max(seq.size(0) for seq in src_seqs)
    tgt_max_len = max(seq.size(0) for seq in tgt_seqs)


    src_padded = torch.stack([
        torch.cat([seq, torch.full((src_max_len - len(seq),), roman2idx["<pad>"], dtype=torch.long)])
        for seq in src_seqs
    ])

    tgt_padded = torch.stack([
        torch.cat([seq, torch.full((tgt_max_len - len(seq),), devanagari2idx["<pad>"], dtype=torch.long)])
        for seq in tgt_seqs
    ])

    return src_padded, tgt_padded

train_ds = TransliterationDataset(
    "/mnt/e_disk/ch24s016/da6401_assignment3/dataset/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv",
    src_char2idx=roman2idx,
    tgt_char2idx=devanagari2idx,
    max_len=32
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)


train_dataset = TransliterationDataset(train_path, roman2idx, devanagari2idx, max_len=32)
dev_dataset = TransliterationDataset(dev_path, roman2idx, devanagari2idx, max_len=32)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)


import wandb

sweep_config = {
    'method': 'bayes',
    'name': 'Seq2Seq Transliteration Sweep',
    'metric': {'name': "val_accuracy", 'goal': 'maximize'},
    'parameters': {
        'embed_size': {'values': [32, 64, 128]},
        'hidden_size': {'values': [64, 128, 256]},
        'num_layers': {'values': [1]},
        'cell_type': {'values': ['RNN', 'GRU', 'LSTM']},
        'optimizer': {'values': ['adam', 'adamw','sgd']},
        'lr': {'values': [0.01, 0.001, 0.0001]},
        'batch_size': {'values': [32, 64]},
        'epochs': {'values': [10 , 15]}
    },
}

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            pred = output.argmax(dim=2)
            for i in range(tgt.size(0)):
                for j in range(1, tgt.size(1)):
                    if tgt[i, j].item() == devanagari2idx["<pad>"]:
                        break
                    if pred[i, j].item() == tgt[i, j].item():
                        correct += 1
                    total += 1
    return correct / total if total > 0 else 0.0

def train_sweep():
    wandb.init()
    config = wandb.config
    logging.info(wandb.config)

    # Update hyperparameters from sweep config
    embed_size = config.embed_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    cell_type = config.cell_type
    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.lr

    # Update data loader if batch_size changes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model setup
    encoder = Encoder(len(roman2idx), embed_size, hidden_size, num_layers, cell_type).to(device)
    decoder = Decoder(len(devanagari2idx), embed_size, hidden_size, num_layers, cell_type).to(device)
    model = Seq2Seq(encoder, decoder, cell_type).to(device)

    # Optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # this two loss functions are not giving good accuracy comparing others
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # elif config.optimizer == 'rmsprop':
    #     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    # Loss function
    loss_function = nn.CrossEntropyLoss(ignore_index=devanagari2idx["<pad>"])

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = loss_function(output, tgt_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_acc = evaluate_accuracy(model, train_loader)
        val_acc = evaluate_accuracy(model, dev_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss": total_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })
        
    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)

    # Unique file name using wandb run name or ID
    run_id = wandb.run.name  # or wandb.run.id
    model_path = os.path.join(model_dir, f"model_{run_id}.pt")
    torch.save(model.state_dict(), model_path)

    logging.info(f"Model saved to {model_path}")
    wandb.finish()
    
sweep_id = wandb.sweep(sweep_config, project="Seq2SeqAssignment3")
wandb.agent(sweep_id, function=train_sweep, count=30)


