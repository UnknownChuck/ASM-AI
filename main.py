from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from google import genai
import io

app = FastAPI()

# Enable CORS so your HTML file can talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini 3 Client (Replace with your 2026 API Key)
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Local Knowledge Base (Memory)
asm_data_context = ""

@app.post("/train")
async def train_with_csv(file: UploadFile = File(...)):
    global asm_data_context
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # "Train" the model by converting the CSV to a summarized context string
    # In a more advanced version, you'd use a Vector Database (FAISS)
    asm_data_context = f"Project ASM Dataset Loaded:\n{df.head(20).to_string()}"
    return {"status": "success", "rows_indexed": len(df)}

@app.post("/ask")
async def ask_gemini(data: dict):
    user_query = data.get("query")
    
    # The "Gemini 3" System Prompt
    prompt = f"""
    You are Cortana, the ASM Project AI. 
    Current Project Context: {asm_data_context}
    
    User Question: {user_query}
    
    Instructions: Use the provided context to answer. If the data isn't there, 
    use your general astronomy knowledge. Be concise, tactical, and helpful.
    """
    
    response = client.models.generate_content(
        model="gemini-3-pro",
        contents=prompt
    )
    
    return {"response": response.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
