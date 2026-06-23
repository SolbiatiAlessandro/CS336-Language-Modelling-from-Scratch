
import regex as re
from functools import reduce
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.first_merge_number = 255 + len(self.special_tokens) + 1
        self.inverse_vocab = {b: i for i, b in vocab.items()}
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}

    def encode(self, text, debug=False):
        if len(self.special_tokens) > 0:
              special_pattern = "|".join(
                  re.escape(token)
                  for token in sorted(self.special_tokens, key=len, reverse=True)
              )
            
              pattern = f"({special_pattern})"
              chunks = re.split(pattern, text)
        else:
            chunks = [text]
        res = []
        for input_string in chunks:
            if input_string in self.special_tokens:
                res.append([self.inverse_vocab[input_string.encode("utf-8")]])
            else:
                input_pretokenized = re.findall(PAT, input_string)
                input_pretokenized_bytes = [
                    [self.inverse_vocab[bytes([byte])] for byte in pretoken.encode("utf-8")] for pretoken in input_pretokenized]
                if debug: print(input_pretokenized_bytes)

                for k, pretoken in enumerate(input_pretokenized_bytes):
                    done_merging = False
                    while not done_merging:
                        done_merging = True
                        possible_merges = []
                        for j, token_j in enumerate(pretoken[:-1]):
                            token_jp1 = pretoken[j+1]
                            possible_merges.append((token_j, token_jp1))
                            done_merging = False
                        def sorted_key(merge_pair):
                            key = (self.vocab[merge_pair[0]],self.vocab[merge_pair[1]])
                            if key not in self.merge_rank.keys():
                                return 1e9
                            return self.merge_rank[key]
                        merge_pairs = sorted(possible_merges, key=sorted_key, reverse=False)
                        if len(merge_pairs) == 0:
                            break
                        merge_pair = merge_pairs[0]
                        if sorted_key(merge_pair) == 1e9:
                            break
                        
                        merge_value = self.inverse_vocab[self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]]
    
                        old_tokens = pretoken
                        i, new_tokens = 0, []
                        while i < len(old_tokens):
                            if (i+1) < len(old_tokens) and \
                            (old_tokens[i], old_tokens[i+1]) == merge_pair:
                                new_tokens.append(merge_value)
                                i += 2
                            else:
                                new_tokens.append(old_tokens[i])
                                i += 1
                        pretoken = new_tokens
                    input_pretokenized_bytes[k] = pretoken
                    
                    
                """ not optimized version, O(n^2) 
                for i, merge_pair in enumerate(self.merges):
                    merge_value = self.inverse_vocab[merge_pair[0] + merge_pair[1]] 
                    for j, old_tokens in enumerate(input_pretokenized_bytes):
                        i, new_tokens = 0, []
                        while i < len(old_tokens):
                            if (i+1) < len(old_tokens) and \
                            (self.vocab[old_tokens[i]], self.vocab[old_tokens[i+1]]) == merge_pair:
                                new_tokens.append(merge_value)
                                i += 2
                            else:
                                new_tokens.append(old_tokens[i])
                                i += 1
                        input_pretokenized_bytes[j] = new_tokens
                """

                        
                res.append(reduce(lambda x,y: x+y, input_pretokenized_bytes, []))
        return reduce(lambda x,y: x+y, res, [])

    def decode(self, tokens):
        return reduce(lambda x,y: x+y, [self.vocab[i] for i in tokens], b"").decode('utf-8', errors="replace")

    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from self.encode(chunk)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with (vocab_filepath).open(encoding="utf-8") as f:
          vocab = {
              int(token_id): bytes.fromhex(value)
              for token_id, value in json.load(f).items()
          }
        
        with (merges_filepath).open(encoding="utf-8") as f:
          merges = [
              (bytes.fromhex(left), bytes.fromhex(right))
              for left, right in json.load(f)
          ]
    
        return cls(
          vocab=vocab,
          merges=merges,
          special_tokens=special_tokens,
        ) 


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from itertools import islice
    from pathlib import Path
    from time import perf_counter

    import numpy as np
    from tqdm import tqdm

    project_dir = Path(__file__).resolve().parent.parent
    default_input = project_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    trained_runs_dir = project_dir / "artifacts" / "tinystories_bpe_runs"

    parser = argparse.ArgumentParser(description="Encode text into uint16 BPE token IDs.")
    parser.add_argument("filename", nargs="?", default=default_input, type=Path)
    parser.add_argument("--file-limit", type=int, default=None, help="Maximum number of input lines.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Directory containing vocab.json and merges.json. Defaults to the latest trained run.",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="Token IDs written per batch.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the encoding progress bar.")
    args = parser.parse_args()

    if args.tokenizer_dir is None:
        trained_runs = sorted(
            path
            for path in trained_runs_dir.iterdir()
            if (path / "vocab.json").exists() and (path / "merges.json").exists()
        )
        if not trained_runs:
            parser.error(f"No trained tokenizer artifacts found in {trained_runs_dir}")
        tokenizer_dir = trained_runs[-1]
    else:
        tokenizer_dir = args.tokenizer_dir

    tokenizer = Tokenizer.from_files(
        tokenizer_dir / "vocab.json",
        tokenizer_dir / "merges.json",
        ["<|endoftext|>"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_dir / "artifacts" / "tokenized_runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "tokens.uint16"

    total_bytes = args.filename.stat().st_size if args.file_limit is None else None
    bytes_processed = 0
    token_count = 0
    started_at = perf_counter()

    with (
        args.filename.open(encoding="utf-8") as source,
        token_path.open("wb") as output,
        tqdm(
            total=total_bytes,
            desc="Encoding",
            unit="B",
            unit_scale=True,
            disable=args.no_progress,
        ) as progress,
    ):
        lines = islice(source, args.file_limit)

        def tracked_lines():
            nonlocal_bytes = 0
            for line in lines:
                line_bytes = len(line.encode("utf-8"))
                nonlocal_bytes += line_bytes
                progress.update(line_bytes)
                yield line
            return nonlocal_bytes

        token_ids = tokenizer.encode_iterable(tracked_lines())
        while True:
            batch = np.fromiter(islice(token_ids, args.batch_size), dtype=np.uint16)
            if batch.size == 0:
                break
            batch.tofile(output)
            token_count += batch.size

        bytes_processed = progress.n

    elapsed = perf_counter() - started_at
    metadata = {
        "timestamp": timestamp,
        "input_path": str(args.filename),
        "file_limit": args.file_limit,
        "tokenizer_dir": str(tokenizer_dir),
        "token_path": token_path.name,
        "dtype": "uint16",
        "bytes_processed": bytes_processed,
        "token_count": token_count,
        "bytes_per_token": bytes_processed / token_count if token_count else None,
        "elapsed_seconds": elapsed,
        "bytes_per_second": bytes_processed / elapsed if elapsed else None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"tokens={token_count:,}")
    print(f"bytes={bytes_processed:,}")
    print(f"bytes/token={metadata['bytes_per_token']:.4f}")
    print(f"time={elapsed:.3f}s")
    print(f"bytes/sec={metadata['bytes_per_second']:,.0f}")
    print(f"Saved token IDs: {token_path}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
