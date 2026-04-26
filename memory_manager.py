"""
Memory management for persisting reasoning history.
Handles saving, loading, and managing reasoning sessions in JSON format.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class MemoryManager:
    """Manage persistence of reasoning sessions."""
    
    def __init__(self, memory_file: str = "reasoning_memory.json"):
        """
        Initialize memory manager.
        
        Args:
            memory_file: Path to JSON memory file
        """
        self.memory_file = memory_file
        self.memory = self.load_memory()
        
    def load_memory(self) -> Dict[str, Any]:
        """
        Load memory from JSON file.
        
        Returns:
            Dictionary containing memory data
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_empty_memory()
        else:
            return self._create_empty_memory()
    
    def _create_empty_memory(self) -> Dict[str, Any]:
        """
        Create empty memory structure.
        
        Returns:
            Empty memory dictionary
        """
        return {
            'sessions': [],
            'current_session_id': None,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
    
    def save_memory(self) -> None:
        """Save memory to JSON file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Error saving memory: {str(e)}")
    
    def create_session(self, session_name: str = None) -> str:
        """
        Create a new reasoning session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
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
            'confidence': {},
            'final_decision': None
        }
        
        self.memory['sessions'].append(session)
        self.memory['current_session_id'] = session_id
        self.save_memory()
        
        return session_id
    
    def save_reasoning_session(self, query: str, context: str, 
                              branches: List[Dict], scores: Dict,
                              confidences: Dict, selected_branch: int,
                              final_decision: str) -> str:
        """
        Save complete reasoning session to memory.
        
        Args:
            query: User query
            context: Retrieved context
            branches: List of branch dictionaries
            scores: Branch scores
            confidences: Branch confidences
            selected_branch: Index of selected branch
            final_decision: Final decision made
            
        Returns:
            Session ID
        """
        session_id = self.create_session()
        
        # Find and update the session
        for session in self.memory['sessions']:
            if session['session_id'] == session_id:
                session.update({
                    'query': query,
                    'context': context[:500] + "...",  # Store truncated context
                    'branches': branches,
                    'scores': scores,
                    'confidences': confidences,
                    'selected_branch': selected_branch,
                    'final_decision': final_decision,
                    'completed_at': datetime.now().isoformat()
                })
                break
        
        self.save_memory()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dictionary or None
        """
        for session in self.memory['sessions']:
            if session['session_id'] == session_id:
                return session
        return None
    
    def list_sessions(self) -> List[Dict]:
        """
        List all sessions.
        
        Returns:
            List of session summaries
        """
        sessions = []
        for session in self.memory['sessions']:
            sessions.append({
                'session_id': session['session_id'],
                'name': session['name'],
                'created_at': session['created_at'],
                'query': session.get('query', 'No query'),
                'final_decision': session.get('final_decision', 'Pending')
            })
        return sessions
    
    def clear_memory(self) -> None:
        """Clear all memory data."""
        self.memory = self._create_empty_memory()
        self.save_memory()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False otherwise
        """
        initial_length = len(self.memory['sessions'])
        self.memory['sessions'] = [s for s in self.memory['sessions'] 
                                   if s['session_id'] != session_id]
        
        if len(self.memory['sessions']) < initial_length:
            if self.memory['current_session_id'] == session_id:
                self.memory['current_session_id'] = None
            self.save_memory()
            return True
        
        return False
    
    def update_selected_branch(self, session_id: str, branch_index: int) -> None:
        """
        Update the selected branch for a session.
        
        Args:
            session_id: Session identifier
            branch_index: Selected branch index
        """
        session = self.get_session(session_id)
        if session:
            session['selected_branch'] = branch_index
            self.save_memory()