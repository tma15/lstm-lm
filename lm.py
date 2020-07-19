import argparse
import collections 
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F


class Vocabulary:
    def __init__(self):
        self.index2item = []
        self.item2index = {}

    def __len__(self):
        return len(self.item2index)

    def __contains__(self, item):
        return item in self.item2index.keys()

    def add_item(self, item):
        index = len(self.item2index)
        self.index2item.append(item)
        self.item2index[item] = index

    def get_item(self, index):
        return self.index2item[index]

    def get_index(self, item):
        return self.item2index[item]

    def save(self, vocab_file):
        with open(vocab_file, 'w') as f:
            for word in self.item2index:
                print(word, file=f)

    @classmethod
    def load(cls, vocab_file):
        vocab = cls()
        with open(vocab_file) as f:
            for line in f:
                word = line.strip()
                vocab.item2index[word] = len(vocab.item2index)
                vocab.index2item.append(word)
        return vocab


class LanguageModel(torch.nn.Module):
    def __init__(
            self,
            vocab,
            dim_emb=128,
            dim_hid=256):
        super().__init__()

        self.vocab = vocab
        self.embed = torch.nn.Embedding(len(vocab), dim_emb)
        self.rnn = torch.nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.out = torch.nn.Linear(dim_hid, len(vocab))

    def forward(self, x, state=None):
        x = self.embed(x)
        x, (h, c) = self.rnn(x, state)
        x = self.out(x)
        return x, (h, c)

    def generate(self, start=None, max_len=100):
        if start is None:
            start = random.choice(self.vocab.index2item)

        idx = self.embed.weight.new_full(
            (1, 1),
            self.vocab.get_index(start),
            dtype=torch.long)
        decoded = [start]
        state = None
        unk = self.vocab.get_index('<unk>')
        for i in range(max_len):
            x, state = self.forward(idx, state)
            x[:, :, unk] = -float('inf')
            idx = torch.argmax(x, dim=-1)

            word = self.vocab.get_item(idx.item())
            decoded.append(word)
        return ' '.join(decoded)


class Preprocessor:
    def __init__(
            self,
            data_file,
            bin_dir='data-bin',
            vocab_file = 'vocab',
            n_max_word=10000,
            num_token_per_file=1000000,
            force_preprocess=False):

        self.data_file = data_file
        self.vocab_file = vocab_file
        self.n_max_word = n_max_word
        self.num_token_per_file = num_token_per_file
        self.bin_dir = bin_dir
        self.force_preprocess = force_preprocess

    def _build_vocab(self):
        counter = collections.Counter()
        with open(self.data_file) as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    counter[word] += 1

        vocab = Vocabulary()
        vocab.add_item('<pad>')
        vocab.add_item('<unk>')
        for word, _ in counter.most_common(self.n_max_word - 2):
            vocab.add_item(word)
        vocab.save(self.vocab_file)
        return vocab

    def _binarize_text(self, vocab):
        data = []
        unk = vocab.get_index('<unk>')

        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

        num_file = 0
        num_token = 0
        with open(self.data_file) as f:
            lines = []
            for line in f:
                lines.append(line)

            random.shuffle(lines)

            for line in lines:
                words = line.strip().split()
                indices = [vocab.get_index(word) if word in vocab
                           else unk for word in words]
                data += indices

                num_token += len(indices)
                if len(data) >= self.num_token_per_file:
                    data = torch.tensor(data)
                    torch.save(data, f'{self.bin_dir}/{num_file}.pt')
                    num_file += 1
                    batch = []

        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

        data = torch.tensor(data)
        torch.save(data, f'{self.bin_dir}/{num_file}.pt')
        print(f'Data size: {num_token}')

    def run(self):
        if self.force_preprocess:
            vocab = self._build_vocab()
            shutil.rmtree(self.bin_dir)
            self._binarize_text(vocab)

        if not os.path.exists(self.vocab_file):
            vocab = self._build_vocab()
        else:
            vocab = Vocabulary.load('vocab')

        if not os.path.exists(self.bin_dir):
            self._binarize_text(vocab)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.filenames = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        tensor = torch.load(self.filenames[index])
        return tensor


def collate_fn(batch):
    return batch[0]


class LanguageModelTrainer:
    def __init__(
            self,
            model,
            optimizer,
            data_dir,
            device,
            max_epochs=10,
            batch_size=32,
            log_interval=200,
            sequence_length=250,
            clip=0.25,
            scheduled_sampling=False):

        self.model = model
        self.optimizer = optimizer
        self.model.to(device)
        self.device = device
        self.step = 0
        self.epoch = 0

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.clip = clip
        self.sequence_length = sequence_length
        self.scheduled_sampling = scheduled_sampling

        self.dataset = Dataset(data_dir)

    def forward(self, inputs, state):
        self.step += 1

        if self.scheduled_sampling:
            bsz = inputs.size(0)
            sequence_length = inputs.size(1)
            x = []
            x_t = None
            h = None
            c = None
            for t in range(sequence_length):
                if t > 0:

#                     p = inputs.new_full(
#                         (bsz, 1),
#                         1 / (1 + math.exp(self.step / 1)))
                    threshold = 1 / (1 + math.exp((self.epoch + 1) / 1))
                    coin = torch.randn(bsz, 1, device=inputs.device)

                    inputs_t = torch.where(
                        threshold < coin,
                        x_t.argmax(dim=-1),
                        inputs[:, t].unsqueeze(1))

                    inputs_t = x_t.argmax(dim=-1)
                else:
                    inputs_t = inputs[:, t].unsqueeze(1)

                x_t, (h_t, c_t) = self.model.forward(inputs[:, t].unsqueeze(1), state)
                x.append(x_t)
                state = (h_t, c_t)
                h = h_t
                c = c_t
            x = torch.cat(x, dim=1)
        else:
            x, (h, c) = self.model.forward(inputs, state)
        return x, (h, c)

    def train(self):
        print('Run trainer')

        pad = self.model.vocab.get_index('<pad>')

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2)
        start_at = time.time()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            loss_epoch = 0.
            num_token = 0
            step = 0
            for data in data_loader:
                # batchfy
                num_batch = data.size(0) // self.batch_size
                data = data.narrow(0, 0, num_batch * self.batch_size)
                batch = data.view(self.batch_size, -1)
                batch = batch[:, :10000]

                state = None
                print('Num batch:', batch.size(1) // self.sequence_length)
                for seen_batch, i in enumerate(range(0, batch.size(1), self.sequence_length), start=1):
                    e = min(i + self.sequence_length, batch.size(1))
                    batch_i = batch[:, i: e]
                    batch_i = batch_i.to(self.device)
                    step += 1

                    input_i = batch_i[:, :-1]
                    target_i = batch_i[:, 1:]

                    x, (h, c) = self.forward(input_i, state)

                    vocab_size = x.size(2)
                    num_token_i = (target_i != pad).sum().item()
                    loss = F.nll_loss(
                        F.log_softmax(x, dim=-1).contiguous().view(-1, vocab_size),
                        target_i.contiguous().view(-1),
                        reduction='sum',
                        ignore_index=pad)

                    self.optimizer.zero_grad()

                    loss.div(num_token_i).backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    num_token += num_token_i
                    loss_epoch += loss.item()
                    
                    h = h.clone().detach()
                    c = c.clone().detach()
                    state = (h, c)

                    if step % self.log_interval == 0:
                        elapsed = time.time() - start_at
                        avg_loss = loss_epoch / num_token
                        ppl = math.exp(avg_loss)
                        print(f'epoch:{epoch} step:{step}'
                              f' loss:{avg_loss:.2f}'
                              f' ppl:{ppl:.2f}'
                              f' elapsed:{elapsed:.2f}',
                              flush=True)

            loss_epoch /= num_token
            ppl = math.exp(loss_epoch)
            elapsed = time.time() - start_at
            print('-' * 50)
            print(f'epoch:{epoch} loss:{loss_epoch:.2f}'
                  f' ppl:{ppl:.2f} elapsed:{elapsed:.2f}')
            decoded = self.model.generate()
            print(f'Sampled: {decoded}')
            print('-' * 50)


def run_trainer(
        train_file,
        max_epochs=100,
        batch_size=128,
        bin_dir='data-bin',
        device='cuda:0',
        force_preprocess=False,
        log_interval=200,
        scheduled_sampling=False):

    preprocessor = Preprocessor(
        train_file,
        bin_dir=bin_dir,
        force_preprocess=force_preprocess)
    preprocessor.run()

    vocab = Vocabulary.load('vocab')
    print(f'Vocabulary size: {len(vocab)}')
    model = LanguageModel(vocab)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = LanguageModelTrainer(
        model,
        optimizer,
        bin_dir,
        device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        log_interval=log_interval,
        scheduled_sampling=scheduled_sampling)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--do-train', action='store_true', default=True)
    parser.add_argument('--do-eval', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--with-scheduled-sampling', default=False, action='store_true')

    args = parser.parse_args()

    if args.do_train:
        run_trainer(
            '/Users/takuya/data/ja.text8',
            device=args.device,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
            scheduled_sampling=args.with_scheduled_sampling,
        )
    elif args.do_eval:
        pass


if __name__ == '__main__':
    main()
