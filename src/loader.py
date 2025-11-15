"""
loader class for handling multiple data sources with dynamic loading
"""

import json
import glob
from datasets import load_dataset
from typing import Tuple, Optional, List, Dict, Any


class Loader:
    def __init__(self):
        self.sources: List[Dict[str, Any]] = []
        self.current_source_idx = 0
        self.current_item_idx = 0
        self.current_source_data = None
        self.total_items = 0

    def add_jsonl(self, pattern: str):
        """add jsonl files from glob pattern"""
        jsonl_files = glob.glob(pattern)
        for file_path in jsonl_files:
            # count lines in file
            with open(file_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)

            self.sources.append({
                'type': 'jsonl',
                'path': file_path,
                'size': num_lines,
                'file_handle': None
            })
            self.total_items += num_lines
            print(f"added jsonl source: {file_path} ({num_lines} lines)")

    def add_huggingface(self, dataset_name: str, split: str = 'train', content_key: str = 'text'):
        """add huggingface dataset with dynamic loading"""
        print(f"loading dataset info: {dataset_name}")

        try:
            # load dataset info without downloading full dataset
            ds = load_dataset(dataset_name, split=split, streaming=False)
            num_rows = len(ds)

            self.sources.append({
                'type': 'huggingface',
                'dataset_name': dataset_name,
                'split': split,
                'content_key': content_key,
                'size': num_rows,
                'dataset': None  # will be loaded when needed
            })
            self.total_items += num_rows
            print(f"added HuggingFace dataset: {dataset_name}/{split} ({num_rows} rows)")
        except Exception as e:
            print(f"error loading dataset {dataset_name}: {e}")

    def _load_current_source(self):
        """load the current source data if not already loaded"""
        if self.current_source_idx >= len(self.sources):
            return False

        source = self.sources[self.current_source_idx]

        if source['type'] == 'jsonl':
            if source['file_handle'] is None:
                source['file_handle'] = open(source['path'], 'r', encoding='utf-8')
                source['lines'] = source['file_handle'].readlines()
                source['file_handle'].close()
                source['file_handle'] = None
            self.current_source_data = source

        elif source['type'] == 'huggingface':
            if source['dataset'] is None:
                print(f"Loading dataset: {source['dataset_name']}/{source['split']}")
                source['dataset'] = load_dataset(source['dataset_name'], split=source['split'])
            self.current_source_data = source

        return True

    def get(self) -> Optional[Tuple[str, str]]:
        """get next (metadata, content) tuple"""
        while self.current_source_idx < len(self.sources):
            # load current source if needed
            if self.current_source_data is None:
                if not self._load_current_source():
                    return None

            source = self.sources[self.current_source_idx]

            # check if we've exhausted current source
            if self.current_item_idx >= source['size']:
                # move to next source
                self.current_source_idx += 1
                self.current_item_idx = 0
                self.current_source_data = None
                continue

            # get item from current source
            try:
                if source['type'] == 'jsonl':
                    line = source['lines'][self.current_item_idx]
                    data = json.loads(line.strip())
                    content = data.get('content', '')
                    metadata = f"{source['path']}:{self.current_item_idx}"

                elif source['type'] == 'huggingface':
                    item = source['dataset'][self.current_item_idx]
                    content = item.get(source['content_key'], '')
                    metadata = f"{source['dataset_name']}/{source['split']}:{self.current_item_idx}"

                self.current_item_idx += 1

                if content:
                    return metadata, content
                else:
                    # skip empty content
                    continue

            except Exception as e:
                print(f"error reading item {self.current_item_idx} from source {self.current_source_idx}: {e}")
                self.current_item_idx += 1
                continue

        # no more data
        return None

    def reset(self):
        """reset iterator to beginning"""
        self.current_source_idx = 0
        self.current_item_idx = 0
        self.current_source_data = None

        # close any open file handles
        for source in self.sources:
            if source['type'] == 'jsonl' and source.get('file_handle'):
                source['file_handle'].close()
                source['file_handle'] = None
            if source['type'] == 'huggingface':
                source['dataset'] = None

    def __len__(self):
        return self.total_items

    def __del__(self):
        """cleanup when object is destroyed"""
        for source in self.sources:
            if source['type'] == 'jsonl' and source.get('file_handle'):
                try:
                    source['file_handle'].close()
                except:
                    pass


# initialize loader
def create_loader():
    loader = Loader()

    # add jsonl files
    loader.add_jsonl('download/code/*.jsonl')

    # add huggingface datasets
    loader.add_huggingface("nampdn-ai/tiny-webtext", split='train', content_key='bot')
    loader.add_huggingface("nampdn-ai/tiny-textbooks", split='train', content_key='textbook')

    print(f"\nTotal entries in loader: {len(loader):,}")
    return loader