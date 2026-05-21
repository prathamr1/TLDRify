from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TLDRify",
              description="Text Summarization using T5",)

# Model and Tokenizer Loading
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device 
device = ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Templating
templates = Jinja2Templates(directory=".")

#Input Scehma --> for dialogues 
class DialogueInput(BaseModel):  # Client Request received as JSON
    dialogue: str

    def clean_data(text):
        text = re.sub(r"\r\n"," ",text)
        text = re.sub(r"\s+"," ",text)
        text = re.sub(r"<.*?>"," ",text)
        text = text.strip().lower()
        return text


    def summarize_dialogue(dialogue : str) -> str:
        dialogue = clean_data(dialogue) # Clean Dialogue
        # Tokenise
        input = tokenizer(dialogue, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)

        model.to(device)
        # Summary Generation in Tokens
        target = model.generate(
            input_ids = input["input_ids"],
            attention_mask = input["attention_mask"],
            max_length = 128,
            num_beams = 4, # beams --> transformer will generate 4 summaries and return the best one
            early_stopping = True
        )

        # Token ids ==> Text using DECODING
        summary = tokenizer.decode(target[0], skip_special_tokens=True)

        return summary
    

# API Endpoints
# GET == Client needs to GET data from Server
# POST == Client needs to Send(POST) data to server
# SYNChronous -> serial execution of tasks 
# ASYNChronos -> parallel execution of tasks with minor delay
@app.post("/summarize/")
async def summarize(dialogue_input : DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary" : summary}

@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})