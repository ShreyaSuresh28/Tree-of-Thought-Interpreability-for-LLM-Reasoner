# Human-in-the-Loop Tree-of-Thought Framework for Document-Grounded Decision Support

An interactive AI system that combines Retrieval-Augmented Generation (RAG) with multi-branch Tree-of-Thought reasoning, featuring automatic scoring, confidence estimation, and human oversight.

## Features

- 📄 **Document Processing**: Upload PDF or TXT documents for grounding
- 🔍 **RAG Retrieval**: Automatic chunking, embedding, and semantic search
- 🌳 **Tree-of-Thought Reasoning**: Multi-branch reasoning with automatic generation
- 📊 **Automatic Scoring**: Multi-metric branch evaluation (relevance, depth, logic, evidence)
- 📈 **Confidence Estimation**: Dynamic confidence calculation for each branch
- 👤 **Human-in-the-Loop**: Interactive branch selection and override capability
- 💾 **Memory Persistence**: JSON-based reasoning history storage
- 🎨 **Interactive Visualization**: NetworkX/Plotly tree visualization

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone or create project directory**
```bash
mkdir tot-decision-support
cd tot-decision-support