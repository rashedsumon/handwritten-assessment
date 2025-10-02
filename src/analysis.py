# src/analysis.py
import json
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Lightweight semantic model (for prototyping)
_sem_model = None
def _get_semantic_model():
    global _sem_model
    if _sem_model is None:
        _sem_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sem_model

def _keyword_score(answer_text, keywords, threshold=0.5):
    # simple fuzzy keyword presence scoring (fraction of keywords present)
    present = 0
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', answer_text, flags=re.IGNORECASE):
            present += 1
    return present / max(len(keywords),1)

def phase1_generate_reports(ocr_results: dict, teacher_df):
    """
    Align teacher-corrected data with OCR text and produce student-level insights.
    teacher_df expected columns: FILENAME, IDENTITY, CORRECTED_ANSWERS_JSON, SCORE
    """
    reports = {}
    for _, row in teacher_df.iterrows():
        fname = row['FILENAME']
        sid = row.get('IDENTITY', fname)
        corrected = {}
        try:
            corrected = json.loads(row.get('CORRECTED_ANSWERS_JSON', "{}"))
        except Exception:
            corrected = {}
        ocr_text = ocr_results.get(fname, "")
        # Basic matching: compute topic counts and simple strengths/weaknesses
        strengths = []
        weaknesses = []
        topic_breakdown = {}
        # naive: if corrected contains per-question flags
        for q, ans_meta in corrected.items():
            ok = ans_meta.get('correct', None) if isinstance(ans_meta, dict) else None
            topic = ans_meta.get('topic','General') if isinstance(ans_meta, dict) else 'General'
            topic_breakdown.setdefault(topic, {'total':0,'correct':0})
            topic_breakdown[topic]['total'] += 1
            if ok:
                topic_breakdown[topic]['correct'] += 1
            # heuristics for strengths/weaknesses
            if ok:
                strengths.append(f"{topic}: {q}")
            else:
                weaknesses.append(f"{topic}: {q}")
        # build suggestions from weaknesses
        suggestions = []
        for w in weaknesses[:5]:
            suggestions.append(f"Revise topics in {w.split(':')[0]}. Practice short answers and keywords.")
        reports[sid] = {
            "filename": fname,
            "score": row.get('SCORE', None),
            "strengths": strengths[:10],
            "weaknesses": weaknesses[:10],
            "topic_breakdown": topic_breakdown,
            "suggestions": suggestions,
            "ocr_excerpt": ocr_text[:200]
        }
    return reports

def phase2_auto_check(ocr_results: dict, answer_key_json_str: str):
    """
    A demo auto-grader:
    - MCQ: exact match
    - Short answer: keyword fraction
    - Essay: semantic similarity (sentence-transformers)
    answer_key_json_str: JSON with question types and expected answers
    """
    try:
        answer_key = json.loads(answer_key_json_str)
    except Exception:
        answer_key = {}
    model = _get_semantic_model()
    reports = {}
    for fname, text in ocr_results.items():
        # naive split by lines -> pretend each line is an answer "Q1: ..." (in practice need layout detection)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        student_answers = {}
        for i, ln in enumerate(lines[:50]):
            student_answers[f"line_{i+1}"] = ln
        total = 0
        got = 0
        per_q = {}
        # MCQ
        mcq = answer_key.get('MCQ', {})
        for q, correct in mcq.items():
            total += 1
            # find presence of q or option in text
            if re.search(re.escape(str(correct)), text, flags=re.IGNORECASE):
                got += 1
                per_q[q] = {"score":1,"max":1}
            else:
                per_q[q] = {"score":0,"max":1}
        # Short answers
        short = answer_key.get('SHORT', {})
        for q, keywords in short.items():
            total += 1
            if isinstance(keywords, str):
                keywords = [keywords]
            score = _keyword_score(text, keywords)
            per_q[q] = {"score": float(score), "max":1}
            got += score
        # Essay checks (semantic)
        essays = answer_key.get('ESSAY', {})
        for q, exemplar in essays.items():
            total += 1
            emb_a = model.encode([exemplar], convert_to_tensor=True)
            emb_b = model.encode([text], convert_to_tensor=True)
            sim = float(util.cos_sim(emb_a, emb_b).cpu().numpy().squeeze())
            # convert sim (-1..1) to 0..1
            sim_score = max(0, (sim+1)/2)
            per_q[q] = {"score": sim_score, "max":1, "sim": sim}
            got += sim_score
        # final
        score_percent = (got/total*100) if total>0 else None
        reports[fname] = {
            "total_questions": total,
            "raw_score": got,
            "score_percent": score_percent,
            "per_question": per_q,
            "ocr_excerpt": text[:200]
        }
    return reports
