from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

class MedicalNoteInput(BaseModel):
    medical_note: str

@app.post("/summarize/")
async def summarize(medical_note_input: MedicalNoteInput):
    try:
        medical_note = medical_note_input.medical_note
        inputs = tokenizer.encode("summarize: " + medical_note, return_tensors="pt", max_length=512, truncation=True)
        
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
