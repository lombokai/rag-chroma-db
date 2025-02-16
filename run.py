import sys
sys.path.append("src")

import os
import gradio as gr
from rag.pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()


# initialize the pipeline
pipe = RAGPipeline(
    data_path = "data/mami_bubu",
    embedder_path = "weights/mxbai-embeddings",
    collection_name = "rag-collection",
    api_key = os.getenv("OPENAI_TOKEN")
)

# function to run the pipeline
def chat_with_rag(user_input):
    relevant_chunks = pipe.retriever.hybrid_search(
        query=user_input,
        top_n=5,
        sparse_weight=0.5,
        dense_weight=0.5
    )
    answer = pipe.generator.generate_response(user_input, relevant_chunks)
    return answer.content


# create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG Chatbot Demo")
    
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Enter your question: ")
    submit_btn = gr.Button("send")

    def respond(message, history):
        response = chat_with_rag(message)
        history.append((message, response))
        return history, ""

    submit_btn.click(
        respond, 
        inputs=[user_input, chatbot], 
        outputs=[chatbot, user_input]
    )

if __name__ == "__main__":
    demo.launch()
