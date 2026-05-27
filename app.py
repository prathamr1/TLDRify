from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TLDRify",
              description="Text Summarization using T5",)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model and Tokenizer Loading
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device 
device = ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Templating
templates = Jinja2Templates(directory="templates")

#Input Scehma --> for dialogues 
class DialogueInput(BaseModel):  # Client Request received as JSON
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n"," ",text)
    text = re.sub(r"\s+"," ",text)
    text = re.sub(r"<.*?>"," ",text)
    text = text.strip().lower()
    return text

def restore_capitalization(text: str) -> str:
    # Capitalize sentence beginnings
    text = ". ".join(
        sentence.strip().capitalize()
        for sentence in text.split(".")
        if sentence.strip()
    )
    return text

def fix_punctuation(text: str) -> str:

    text = text.strip()

    # Add full stop if missing
    if text and text[-1] not in ".!?":
        text += "."

    # Remove extra spaces before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    return text

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # Clean Dialogue
    dialogue = "summarize: " + dialogue
    # Tokenise
    input = tokenizer(dialogue, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)

    model.to(device)

    # Summary Generation in Tokens
    target = model.generate(
        input_ids = input["input_ids"],
        attention_mask = input["attention_mask"],
        min_length = 20,
        max_length = 150,
        num_beams = 4, # beams --> transformer will generate 4 summaries and return the best one
        early_stopping = True,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Token ids ==> Text using DECODING
    summary = tokenizer.decode(target[0], skip_special_tokens=True)
    summary = restore_capitalization(summary)
    summary = fix_punctuation(summary)
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