"""
Advanced Memory Manager with similarity-based retrieval.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, util


class MemoryManager:
    """Manage intelligent memory of reasoning sessions."""

    def __init__(self, memory_file: str = "reasoning_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()

        # 🔥 Add embedding model (IMPORTANT)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    # =========================
    # LOAD / SAVE
    # =========================
    def load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_empty_memory()
        return self._create_empty_memory()

    def _create_empty_memory(self) -> Dict[str, Any]:
        return {
            'sessions': [],
            'current_session_id': None,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
        }

    def save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    # =========================
    # SESSION MANAGEMENT
    # =========================
    def create_session(self, session_name: str = None) -> str:

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not session_name:
            session_name = f"Session_{session_id}"

        session = {
            'session_id': session_id,
            'name': session_name,
            'created_at': datetime.now().isoformat(),
            'query': None,
            'context': None,
            'branches': [],
            'selected_branch': None,
            'scores': {},
            'confidences': {},
            'final_decision': None,
            'embedding': None  # 🔥 NEW
        }

        self.memory['sessions'].append(session)
        self.memory['current_session_id'] = session_id
        self.save_memory()

        return session_id

    # =========================
    # SAVE SESSION (INTELLIGENT)
    # =========================
    def save_reasoning_session(self, query: str, context: str,
                              branches: List[Dict], scores: Dict,
                              confidences: Dict, selected_branch: int,
                              final_decision: str) -> str:

        session_id = self.create_session()

        # 🔥 Create embedding of query
        query_embedding = self.model.encode(query).tolist()

        for session in self.memory['sessions']:
            if session['session_id'] == session_id:
                session.update({
                    'query': query,
                    'context': context[:500] + "...",
                    'branches': branches,
                    'scores': scores,
                    'confidences': confidences,
                    'selected_branch': selected_branch,
                    'final_decision': final_decision,
                    'embedding': query_embedding,
                    'completed_at': datetime.now().isoformat()
                })
                break

        self.save_memory()
        return session_id

    # =========================
    # SIMILAR QUERY RETRIEVAL 🔥
    # =========================
    def find_similar_sessions(self, query: str, top_k: int = 3) -> List[Dict]:

        if not self.memory['sessions']:
            return []

        query_emb = self.model.encode(query, convert_to_tensor=True)

        scored_sessions = []

        for session in self.memory['sessions']:
            if session.get('embedding') is None:
                continue

            session_emb = self.model.encode(session['query'], convert_to_tensor=True)

            similarity = util.cos_sim(query_emb, session_emb).item()

            scored_sessions.append((similarity, session))

        scored_sessions.sort(key=lambda x: x[0], reverse=True)

        return [s[1] for s in scored_sessions[:top_k]]

    # =========================
    # BASIC FUNCTIONS
    # =========================
    def get_session(self, session_id: str) -> Optional[Dict]:
        return next((s for s in self.memory['sessions']
                     if s['session_id'] == session_id), None)

    def list_sessions(self) -> List[Dict]:
        return [{
            'session_id': s['session_id'],
            'name': s['name'],
            'created_at': s['created_at'],
            'query': s.get('query', 'No query'),
            'final_decision': s.get('final_decision', 'Pending')
        } for s in self.memory['sessions']]

    def clear_memory(self):
        self.memory = self._create_empty_memory()
        self.save_memory()

    def delete_session(self, session_id: str) -> bool:
        before = len(self.memory['sessions'])
        self.memory['sessions'] = [
            s for s in self.memory['sessions']
            if s['session_id'] != session_id
        ]

        self.save_memory()
        return len(self.memory['sessions']) < before
