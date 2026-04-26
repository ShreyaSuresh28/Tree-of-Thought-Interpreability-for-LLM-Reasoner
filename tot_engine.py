"""
Tree-of-Thought reasoning engine.
Generates and manages reasoning branches for decision support.
"""

import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class ThoughtNode:
    """Represents a node in the Tree-of-Thought."""
    
    def __init__(self, id: str, text: str, parent_id: Optional[str] = None, 
                 depth: int = 0, branch_type: str = "reasoning"):
        self.id = id
        self.text = text
        self.parent_id = parent_id
        self.depth = depth
        self.branch_type = branch_type  # reasoning, yes, no, final
        self.children = []
        self.score = None
        self.created_at = datetime.now().isoformat()
        
class ToTEngine:
    """Tree-of-Thought reasoning engine."""
    
    def __init__(self):
        """Initialize Tree-of-Thought engine."""
        self.tree = nx.DiGraph()
        self.nodes = {}
        self.root_id = None
        
    def create_tree(self, question: str, context: str) -> str:
        """
        Create a new Tree-of-Thought for a question.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Root node ID
        """
        self.tree = nx.DiGraph()
        self.nodes = {}
        
        # Create root node
        root_text = f"Question: {question}\n\nContext: {context[:500]}..."
        self.root_id = self.add_node(root_text, branch_type="root")
        
        return self.root_id
    
    def add_node(self, text: str, parent_id: Optional[str] = None, 
                 branch_type: str = "reasoning") -> str:
        """
        Add a new reasoning node to the tree.
        
        Args:
            text: Node text content
            parent_id: ID of parent node (None for root)
            branch_type: Type of branch (reasoning, yes, no, final)
            
        Returns:
            New node ID
        """
        node_id = str(uuid.uuid4())[:8]
        
        if parent_id:
            depth = self.nodes[parent_id].depth + 1
        else:
            depth = 0
            
        node = ThoughtNode(node_id, text, parent_id, depth, branch_type)
        self.nodes[node_id] = node
        self.tree.add_node(node_id, **node.__dict__)
        
        if parent_id:
            self.tree.add_edge(parent_id, node_id)
            self.nodes[parent_id].children.append(node_id)
            
        return node_id
    
    def generate_yes_no_branches(self, question: str, context: str) -> Dict[str, str]:
        """
        Generate YES and NO reasoning branches for binary questions.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with 'yes' and 'no' branch node IDs
        """
        # Create YES branch
        yes_reasoning = self._generate_reasoning_path(question, context, "YES")
        yes_id = self.add_node(yes_reasoning, self.root_id, branch_type="yes")
        
        # Create NO branch
        no_reasoning = self._generate_reasoning_path(question, context, "NO")
        no_id = self.add_node(no_reasoning, self.root_id, branch_type="no")
        
        # Add supporting evidence nodes
        self._add_evidence_nodes(yes_id, context, "YES")
        self._add_evidence_nodes(no_id, context, "NO")
        
        return {'yes': yes_id, 'no': no_id}
    
    def generate_open_ended_branches(self, question: str, context: str, 
                                    num_branches: int = 3) -> List[str]:
        """
        Generate multiple reasoning branches for open-ended questions.
        
        Args:
            question: User's question
            context: Retrieved context
            num_branches: Number of branches to generate
            
        Returns:
            List of branch node IDs
        """
        branch_ids = []
        
        for i in range(num_branches):
            perspective = self._get_perspective(i)
            reasoning = self._generate_reasoning_path(question, context, perspective)
            branch_id = self.add_node(reasoning, self.root_id, branch_type="reasoning")
            branch_ids.append(branch_id)
            
            # Add supporting evidence
            self._add_evidence_nodes(branch_id, context)
            
        return branch_ids
    
    def expand_branch(self, parent_id: str, question: str, context: str) -> str:
        """
        Expand a branch with deeper reasoning.
        
        Args:
            parent_id: ID of node to expand
            question: Original question
            context: Retrieved context
            
        Returns:
            New node ID
        """
        parent_node = self.nodes[parent_id]
        deeper_reasoning = self._generate_deeper_reasoning(
            parent_node.text, question, context
        )
        
        new_id = self.add_node(deeper_reasoning, parent_id, branch_type="reasoning")
        return new_id
    
    def _generate_reasoning_path(self, question: str, context: str, 
                                stance: str = "") -> str:
        """
        Generate a reasoning path (simulated LLM reasoning).
        
        Args:
            question: User's question
            context: Retrieved context
            stance: Reasoning stance/perspective
            
        Returns:
            Generated reasoning text
        """
        # This is a placeholder - in production, this would call an LLM
        reasoning = f"""
        Reasoning Path ({stance if stance else 'Neutral'}):
        
        Based on the provided context, let me analyze the question: "{question}"
        
        Key points from context:
        {context[:300]}...
        
        Analysis steps:
        1. First, identify the main requirements in the question
        2. Cross-reference with available information in context
        3. Evaluate supporting and contradicting evidence
        4. Consider implications and edge cases
        5. Formulate logical conclusion
        
        Conclusion: {stance if stance else 'Based on the evidence, a balanced approach is recommended.'}
        """
        
        return reasoning.strip()
    
    def _add_evidence_nodes(self, parent_id: str, context: str, 
                           stance: str = "") -> None:
        """
        Add evidence nodes supporting a reasoning branch.
        
        Args:
            parent_id: Parent node ID
            context: Retrieved context
            stance: Reasoning stance
        """
        # Extract key evidence from context
        evidence_points = context.split('.')[:3]  # Top 3 sentences
        
        for i, evidence in enumerate(evidence_points):
            if len(evidence.strip()) > 20:
                evidence_text = f"Evidence {i+1}: {evidence.strip()}"
                self.add_node(evidence_text, parent_id, branch_type="evidence")
    
    def _generate_deeper_reasoning(self, current_reasoning: str, 
                                  question: str, context: str) -> str:
        """
        Generate deeper reasoning by expanding on current reasoning.
        
        Args:
            current_reasoning: Current reasoning text
            question: Original question
            context: Retrieved context
            
        Returns:
            Deeper reasoning text
        """
        deeper = f"""
        Expanding on previous reasoning:
        
        {current_reasoning[:200]}...
        
        Additional considerations:
        - What assumptions were made?
        - Are there alternative interpretations?
        - How robust is the evidence?
        - What are the potential counterarguments?
        
        Refined conclusion: The evidence strongly supports the initial reasoning,
        with particular emphasis on the most relevant contextual information.
        """
        
        return deeper.strip()
    
    def _get_perspective(self, index: int) -> str:
        """
        Get different reasoning perspectives.
        
        Args:
            index: Perspective index
            
        Returns:
            Perspective name
        """
        perspectives = [
            "Conservative Approach",
            "Optimistic Outlook", 
            "Risk-Aware Analysis",
            "Evidence-Focused",
            "Balanced Perspective"
        ]
        
        return perspectives[index % len(perspectives)]
    
    def get_branch_nodes(self, branch_root_id: str) -> List[str]:
        """
        Get all nodes in a branch.
        
        Args:
            branch_root_id: Root node ID of the branch
            
        Returns:
            List of node IDs in the branch
        """
        nodes = [branch_root_id]
        
        def collect_children(node_id):
            for child_id in self.nodes[node_id].children:
                nodes.append(child_id)
                collect_children(child_id)
        
        collect_children(branch_root_id)
        return nodes
    
    def get_tree_structure(self) -> nx.DiGraph:
        """
        Get the entire tree structure.
        
        Returns:
            NetworkX DiGraph of the tree
        """
        return self.tree
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            ThoughtNode object or None
        """
        return self.nodes.get(node_id)