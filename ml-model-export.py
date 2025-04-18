import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline
)

class LegalChatbotModel:
    def __init__(self, documents_path="./frontend/data"):
        # System prompt
        system_prompt = """
        You are an AI assistant specializing in legal and procedural guidance. 
        Provide clear, concise, and helpful answers based on the context provided.
        """
        
        # Load documents
        try:
            documents = SimpleDirectoryReader(documents_path).load_data()
            print(f"Loaded {len(documents)} documents")
        except Exception as e:
            print(f"Error loading documents: {e}")
            documents = []
        
        # Login to Hugging Face
        login(token="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")
        
        # Initialize embedding model
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )
        print("Embedding model initialized")
        
        # Initialize LLM with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B", 
            token="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"
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # Create pipeline
        self.llm = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.2
        )
        print("LLM initialized")
        
        # Configure settings
        Settings.chunk_size = 1024
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        
        # Create vector index
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine()
    
    def query(self, question):
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Flask Application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
legal_model = None

@app.route('/chat', methods=['POST'])
def chat():
    global legal_model
    
    # Initialize model if not already done
    if legal_model is None:
        try:
            legal_model = LegalChatbotModel()
        except Exception as e:
            print(f"Model initialization error: {e}")
            return jsonify({"error": f"Model initialization failed: {str(e)}"}), 500
    
    # Get user message
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Query the model
        response = legal_model.query(question)
        return jsonify({"reply": response})
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)