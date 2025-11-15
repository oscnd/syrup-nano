from nano import Nano
from datasets import load_dataset

nano = Nano()

# construct word special index
nano.construct_word_special("dataset/tokenizer/word_*.jsonl")
print(f"constructed word special with final token number: {nano.get_num()}")

# construct word compound index
nano.construct_word_root("dataset/tokenizer/root.jsonl")
print(f"constructed word root with final token number: {nano.get_num()}")

# construct code jsonl dataset
nano.construct_from_glob("download/code/*.jsonl")
print(f"code dataset with final token number: {nano.get_num()}")

# load tiny-webtext dataset
ds = load_dataset("nampdn-ai/tiny-webtext")
print(f"dataset: {ds}")

# process training set
for i in range(len(ds['train'])):
    content = ds['train'][i]['bot']
    filename = f"nampdn-tiny-webtext/train:{i}"
    nano.construct_content(filename, content)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i+1}/{len(ds['train'])} entries. Token count: {nano.get_num()}")

print(f"final token number: {nano.get_num()}")