"""Persistent project memory with TF-IDF keyword search."""

import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone


class MemoryStore:
    """JSONL-backed project memory with keyword search."""

    def __init__(self, project_dir: str = "."):
        self.dir = os.path.join(project_dir, ".coder")
        self.path = os.path.join(self.dir, "memories.jsonl")
        self.memories: list[dict] = []
        self._load()

    def _load(self):
        """Load memories from disk."""
        self.memories = []
        if not os.path.isfile(self.path):
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.memories.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _save(self):
        """Write all memories to disk."""
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path, "w") as f:
            for mem in self.memories:
                f.write(json.dumps(mem) + "\n")

    def add(
        self,
        content: str,
        tags: list[str] | None = None,
        mem_type: str = "task",
    ) -> dict:
        """Add a new memory. Long content is auto-chunked."""
        chunks = self._chunk(content)
        added = []
        for chunk in chunks:
            mem = {
                "id": uuid.uuid4().hex[:12],
                "content": chunk,
                "tags": tags or [],
                "type": mem_type,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.memories.append(mem)
            added.append(mem)
        self._save()
        return added[0] if len(added) == 1 else added

    def _chunk(self, text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
        """Split long text into overlapping chunks (~token = ~word)."""
        words = text.split()
        if len(words) <= max_tokens:
            return [text]
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + max_tokens]
            chunks.append(" ".join(chunk_words))
            i += max_tokens - overlap
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """TF-IDF keyword search over memories."""
        if not self.memories:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return self.memories[:top_k]

        # Build document frequency
        doc_count = len(self.memories)
        df = Counter()
        doc_tokens = []
        for mem in self.memories:
            tokens = set(self._tokenize(mem["content"] + " " + " ".join(mem.get("tags", []))))
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] += 1

        # Score each memory
        scored = []
        for i, mem in enumerate(self.memories):
            tokens = doc_tokens[i]
            tf = Counter(self._tokenize(mem["content"]))
            total = sum(tf.values()) or 1
            score = 0.0
            for term in query_terms:
                if term in tokens:
                    term_tf = tf[term] / total
                    term_idf = math.log((doc_count + 1) / (df.get(term, 0) + 1))
                    score += term_tf * term_idf
            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization + lowercasing."""
        return re.findall(r"\w+", text.lower())

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID (or prefix)."""
        for i, mem in enumerate(self.memories):
            if mem["id"].startswith(memory_id):
                self.memories.pop(i)
                self._save()
                return True
        return False

    def list_all(self) -> list[dict]:
        """Return all memories."""
        return list(self.memories)

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self.memories)

    def get_context(self, query: str = "", max_memories: int = 5) -> str:
        """Get memory context string for injection into system prompt."""
        if query:
            mems = self.search(query, top_k=max_memories)
        else:
            # Return most recent memories
            mems = self.memories[-max_memories:] if self.memories else []

        if not mems:
            return ""

        lines = ["## Relevant Project Memories"]
        for mem in mems:
            tags = ", ".join(mem.get("tags", []))
            prefix = f"[{mem['type']}]"
            if tags:
                prefix += f" ({tags})"
            lines.append(f"- {prefix} {mem['content']}")
        return "\n".join(lines)
