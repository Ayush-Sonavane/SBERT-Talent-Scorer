import os
import multiprocessing
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class SemanticMatcher:
    def __init__(self, model_path: str = "models/all-MiniLM-L6-v2"):
        quant_path = os.path.join(model_path, "model_quantized.onnx")
        onnx_path = os.path.join(model_path, "model.onnx")

        if os.path.exists(quant_path):
            chosen_onnx = quant_path
        elif os.path.exists(onnx_path):
            chosen_onnx = onnx_path
        else:
            raise FileNotFoundError(
                f"No ONNX model found at '{model_path}'. "
                f"Expected '{quant_path}' or '{onnx_path}'."
            )

        self.use_onnx = True

        opts = ort.SessionOptions()
        num_threads = max(1, min(4, multiprocessing.cpu_count()))
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(chosen_onnx, opts, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.jd_embedding = None
        self.model_file = chosen_onnx

    def _encode_onnx(self, texts: list, batch_size: int = 512) -> np.ndarray:
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = [t[:300] for t in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=64, return_tensors="np"
            )
            feeds = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                feeds["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])

            (hidden_state,) = self.session.run(None, feeds)
            mask = encoded["attention_mask"].astype(np.float32)
            mask_expanded = np.expand_dims(mask, axis=-1)
            summed = np.sum(hidden_state * mask_expanded, axis=1)
            counts = np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
            pooled = summed / counts

            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            normalized = pooled / norms
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)

    def fit_jd(self, jd_text: str):
        emb = self._encode_onnx([jd_text])
        self.jd_embedding = emb[0]

    def build_candidate_text(self, candidate: dict) -> str:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        parts = []
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        if headline:
            parts.append(headline)
        if summary:
            parts.append(summary[:80])

        for job in career_history[:2]:
            title = job.get("title", "")
            if title:
                parts.append(title)

        skill_names = [s.get("name", "") for s in skills[:15] if s.get("name")]
        if skill_names:
            parts.append(", ".join(skill_names))

        return " ".join(parts)

    def score_candidate(self, candidate: dict) -> float:
        if self.jd_embedding is None:
            raise ValueError("Job description not fit yet. Call fit_jd first.")

        cand_text = self.build_candidate_text(candidate)
        if not cand_text.strip():
            return 0.0

        cand_emb = self._encode_onnx([cand_text])
        sim = float(np.dot(self.jd_embedding, cand_emb[0]))

        return max(0.0, min(1.0, sim))

    def score_candidates_batch(self, candidates: list) -> list:
        if self.jd_embedding is None:
            raise ValueError("Job description not fit yet. Call fit_jd first.")

        texts = [self.build_candidate_text(cand) for cand in candidates]
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        scores = [0.0] * len(candidates)
        if not valid_texts:
            return scores

        embeddings = self._encode_onnx(valid_texts)

        similarities = np.dot(embeddings, self.jd_embedding)
        for i, idx in enumerate(valid_indices):
            scores[idx] = float(max(0.0, min(1.0, similarities[i])))

        return scores
