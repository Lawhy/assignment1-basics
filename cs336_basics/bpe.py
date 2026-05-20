from __future__ import annotations

import os
from typing import BinaryIO
import regex as re
from concurrent.futures import ProcessPoolExecutor, Future


class BPETokenizer:
    # GPT-2 regex for pre-tokenization (shared by encode and train).
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # Permanent tokenizer state: vocabulary and merge rules.
        self.vocab = vocab  # {id: token bytes}
        self.merges = merges  # [(token1, token2), ...] in training order
        self.special_tokens = special_tokens or []
        self.token_to_id = {v: k for k, v in self.vocab.items()}  # bytes -> id
        self.merge_priorities = {
            pair: i for i, pair in enumerate(self.merges)
        }  # pair -> training iteration index; lower = earlier learned = higher priority

        # Regex for splitting on special tokens but preserving them as separate segments.
        # Sort by length desc so longer overlapping tokens match first (regex alternation
        # is leftmost-wins, not longest-wins).
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted_tokens)) + ")")
        else:
            self.special_token_pattern = None

    #########################################################
    # Encoding and decoding
    #########################################################

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        # Split on special tokens but preserve them as separate segments.
        segments = re.split(self.special_token_pattern, text) if self.special_token_pattern else [text]

        for segment in segments:
            # Special tokens map straight to their pre-assigned id.
            if segment in self.special_tokens:
                token_ids.append(self.token_to_id[segment.encode("utf-8")])
                continue

            # Pre-tokenize the segment, then apply merges to each pre-token.
            for match in re.finditer(self.PAT, segment):
                token_bytes = [bytes([b]) for b in match.group(0).encode("utf-8")]

                # Priority-based merge: at each step, pick the applicable adjacent pair
                # with the smallest merge_priorities value (earliest learned) and apply
                # just that one. Loop until no applicable pair remains.
                while len(token_bytes) > 1:
                    priority_pair = None
                    priority_pair_index = -1
                    for i in range(len(token_bytes) - 1):
                        pair = (token_bytes[i], token_bytes[i + 1])
                        if pair in self.merge_priorities:
                            if (
                                priority_pair is None
                                or self.merge_priorities[pair] < self.merge_priorities[priority_pair]
                            ):
                                priority_pair = pair
                                priority_pair_index = i
                    if priority_pair_index == -1:
                        break
                    token_bytes = (
                        token_bytes[:priority_pair_index]
                        + [priority_pair[0] + priority_pair[1]]
                        + token_bytes[priority_pair_index + 2 :]
                    )

                token_ids += [self.token_to_id[b] for b in token_bytes]

        return token_ids

    def encode_iterable(self, iterable):
        """Stream-encode an iterable of strings, yielding token IDs.

        Memory-efficient: only one chunk is held in memory at a time."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in token_ids).decode("utf-8", errors="replace")

    #########################################################
    # Training
    #########################################################

    @classmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 4,
        **_kwargs,
    ) -> BPETokenizer:
        """Train a byte-level BPE tokenizer on the corpus at `input_path`.

        Returns a constructed BPETokenizer with the trained `vocab` and `merges`.
        Algorithm: parallel pre-tokenization → incremental pair-count merge loop."""

        # Initial vocab: 256 single-byte tokens + special tokens.
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for s in special_tokens:
            vocab[len(vocab)] = s.encode("utf-8")

        # Pre-tokenize the corpus in parallel; result is {pretoken_tuple: frequency}.
        pretoken_counts = cls._pretokenize_corpus(input_path, num_processes, special_tokens)

        # Seed the incremental state used by the merge loop:
        #   pair_counts[(a,b)]          total occurrences of pair (a,b) in current corpus.
        #   pair_to_pretokens[(a,b)]  pre-tokens currently containing the pair;
        #                               reverse index for O(affected) work per merge.
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        for pretoken, n in pretoken_counts.items():
            for a, b in zip(pretoken, pretoken[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + n
                pair_to_pretokens.setdefault((a, b), set()).add(pretoken)

        # Merge loop. Each iteration picks the highest-frequency pair (lexicographically
        # greater on ties), records it, and patches the incremental state by touching
        # only the pre-tokens that actually contained the merged pair.
        merges: list[tuple[bytes, bytes]] = []
        while len(vocab) < vocab_size and pair_counts:
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merged_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[len(vocab)] = merged_token

            affected_pretokens = pair_to_pretokens.pop(best_pair)
            pair_counts.pop(best_pair)

            for old_pretoken in affected_pretokens:
                n = pretoken_counts.pop(old_pretoken)

                # Withdraw old_pretoken's contributions to every pair except best_pair
                # (which we already popped wholesale above).
                for a, b in zip(old_pretoken, old_pretoken[1:]):
                    if (a, b) == best_pair:
                        continue
                    pair_counts[(a, b)] -= n
                    if pair_counts[(a, b)] <= 0:
                        pair_counts.pop((a, b))
                    containing_pretokens = pair_to_pretokens.get((a, b))
                    if containing_pretokens is not None:
                        containing_pretokens.discard(old_pretoken)
                        if not containing_pretokens:
                            pair_to_pretokens.pop((a, b))

                # Rewrite the pre-token with the merge applied, then register the
                # new shape's pair contributions.
                new_pretoken = cls._apply_merge(old_pretoken, best_pair, merged_token)
                for a, b in zip(new_pretoken, new_pretoken[1:]):
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + n
                    pair_to_pretokens.setdefault((a, b), set()).add(new_pretoken)

                # Two distinct old pre-tokens may collapse to the same new shape — sum counts.
                pretoken_counts[new_pretoken] = pretoken_counts.get(new_pretoken, 0) + n

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    #########################################################
    # Training helpers (kept as @staticmethod so they remain picklable for ProcessPoolExecutor)
    #########################################################

    @staticmethod
    def _pretokenize_chunk(
        file_path: str, start: int, end: int, special_tokens: list[str]
    ) -> dict[tuple[bytes, ...], int]:
        """Read bytes [start:end) from file_path, split on special tokens, pre-tokenize
        each document with the GPT-2 regex, and return a per-pretoken frequency dict."""
        counts: dict[tuple[bytes, ...], int] = {}
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            docs = re.split("|".join(map(re.escape, special_tokens)), chunk) if special_tokens else [chunk]
            for doc in docs:
                for match in re.finditer(BPETokenizer.PAT, doc):
                    token = tuple(bytes([t]) for t in match.group(0).encode("utf-8"))
                    counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _pretokenize_corpus(
        input_path: str | os.PathLike,
        num_processes: int,
        special_tokens: list[str],
    ) -> dict[tuple[bytes, ...], int]:
        """Chunk the corpus on special-token boundaries and pre-tokenize each chunk
        in its own worker process; merge the per-chunk counters into a single dict."""
        input_path = str(input_path)
        if special_tokens:
            with open(input_path, "rb") as f:
                boundaries = BPETokenizer._find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        else:
            boundaries = [0, os.path.getsize(input_path)]

        futures: list[Future] = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                futures.append(executor.submit(BPETokenizer._pretokenize_chunk, input_path, start, end, special_tokens))

        counts: dict[tuple[bytes, ...], int] = {}
        for future in futures:
            for token, n in future.result().items():
                counts[token] = counts.get(token, 0) + n
        return counts

    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """Chunk the file into byte ranges, each starting at an occurrence of
        `split_special_token`. Boundaries are pulled forward from a uniform initial
        guess by scanning ahead 4 KB at a time for the next delimiter occurrence."""
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses, uniformly spaced.
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Scan ahead in 4 KB increments.

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Deduplicate in case two initial guesses snapped to the same boundary.
        return sorted(set(chunk_boundaries))

    @staticmethod
    def _apply_merge(pretoken: tuple[bytes, ...], pair: tuple[bytes, bytes], merged_token: bytes) -> tuple[bytes, ...]:
        """Rewrite a single pre-token, collapsing every left-to-right occurrence of
        `pair` into `merged_token` in one pass."""
        out: list[bytes] = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                out.append(merged_token)
                i += 2
            else:
                out.append(pretoken[i])
                i += 1
        return tuple(out)
