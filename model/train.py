import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group
from model_2b import build_model
from tokenizer import MathTokenizer  # wrap sentencepiece
from dataset import ScholarData
from torch.utils.data import DataLoader

def main():
    init_process_group(backend="nccl")

    model = build_model(vocab_size=32000).cuda()
    model = FSDP(model)

    dataset = MyDataset("dataset_clean.jsonl", seq_len=4096)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for step, batch in enumerate(loader):
        tokens = batch.cuda()
        logits = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step} loss {loss.item()}")

if __name__ == "__main__":
    main()
