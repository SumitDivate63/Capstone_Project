from datasets.dataloader import get_train_dataloader

loader = get_train_dataloader(batch_size=4)

batch = next(iter(loader))

print(batch["participant_id"])

print(len(batch["participant_id"]))

print(batch["labels"].keys())

print(batch["metadata"].keys())

print(batch["visual"].keys())

print(batch["audio"].keys())

print(batch["text"].keys())
