import pymupdf4llm
from litellm import completion, batch_completion


PROMPT = """You are a materials science expert. Your task is to extract ONLY explicitly stated synthesis-related information from the provided research paper. Do NOT assume or infer any information not directly stated in the paper.

If the paper does not contain any synthesis-related information, simply respond with:  
**"NOT A MATERIAL SYNTHESIS PAPER"**

───────────────────────────────────────────────

## 0. Key Contributions

Explicitly list all of the following contributions, if mentioned:

● Deposited materials: <summary>
● Key Deposition/Process Method: <summary>
● Achieved Mobility and Stability (e.g., VTH shifts, SS, Ion/Ioff): <summary>
● Primary Application Domain: <summary>

## 1. Materials

Explicitly extract the following for all materials:

● Channel material name:  
● Composition:  
● Crystallinity (e.g., amorphous/crystalline):  
● Band gap (eV):  

## 2. Device Structure (per sample)

For each explicitly named sample (e.g., Sample A, B, etc.), list:

 
● Device structure type:  
● Gate electrode material:  
● Gate insulator material:  
● Gate insulator process (e.g., sputtering):  
● Gate insulator deposition temperature:  
● Gate insulator precursor/target composition:  
● Gate insulator thickness (nm):  
● Substrate material:  
● Semiconductor (channel) thickness (nm):  
● S/D electrode material:  
● Channel length (µm):  
● Channel width (µm):  
● Passivation layer material:  
● Passivation layer thickness (nm):  
● Passivation process:  

## 3. Deposition Process

For each synthesis process (e.g., ALD, PVD, etc.), extract explicitly stated deposition parameters.  
ALD uses a **specialized set of parameters** (see below); PVD and others use the common format.

---

### ⬛ ALD Process (if applicable)

● Precursor:  
● Process Temperature (°C):  
● Process Pressure (Torr):  
● Oxygen partial pressure (Torr): 
● Reactant Gas:  
● Subcycle Ratio (e.g., O/Ga):  
● Precursor Feeding Time (sec):  
● Plasma Power (W):  
● Plasma Exposure Time (sec):  

---

### ⬛ PVD / Sputtering Process (or all other non-ALD)

● Deposition method (e.g., RF sputtering):  
● Power (W):  
● Target:  
● Gas type / flow (e.g., Ar/O2, sccm):  
● Oxygen partial pressure (Torr): 
● Process pressure (Torr):  
● Substrate temperature (°C):  

───────────────────────────────────────────────

## 4. Post-Deposition Annealing (explicit info only)

● Annealing time (sec or min):  
● Annealing temperature (°C):  
● Annealing atmosphere (e.g., N₂, forming gas):  

───────────────────────────────────────────────

## 5. Product Characteristics (per sample)

For each labeled sample, extract the following parameters (if explicitly reported).  
If not reported, write “N/A”.

● Field-effect mobility (cm²/V·s):  
● Threshold voltage (V):  
● Subthreshold swing (V/dec):  
● On/off ratio (Ion/Ioff):  
● Threshold voltage shift (V):  
● Stability measurement condition(s):  

───────────────────────────────────────────────

EXTRACTION RULES (MANDATORY)

1. Use ONLY explicitly stated information.
2. Use exact values and units from the text.
3. Always list sample-specific data with sample labels.
4. If a parameter is not reported, explicitly write “N/A”.
5. Do not interpret or infer missing values.
6. Quote unusual methods verbatim for clarity.
7. Maintain formatting structure exactly as shown.

Your output must be accurate, structured, and faithful to the original paper.
"""


import fitz  # PyMuPDF

def pdf_bytes_to_markdown(pdf_bytes):
    # Create a PyMuPDF document from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Convert to markdown
    md_text = pymupdf4llm.to_markdown(pdf_document)
    
    # Close the document to free resources
    pdf_document.close()
    
    return md_text

def extract_recipe_from_text(texts, model="gpt-4o-2024-11-20"):
    def filter_text(text):
        if len(text) < 100:
            return None
        if len(text) > 50000:
            text = text[:50000]
        return text
    
    texts = [filter_text(text) for text in texts]
    messages = [[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "Scientific Paper:\n" + text},
    ] for text in texts if text is not None]

    messages = batch_completion(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.6,
    )
    return [message.choices[0].message.content for message in messages]

def read_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
        text = pdf_bytes_to_markdown(pdf_bytes)
        return text
    
def pdf_bytelist_to_recipes(pdf_bytelist, model="gpt-4o-2024-11-20"):
    texts = [pdf_bytes_to_markdown(pdf_bytes) for pdf_bytes in pdf_bytelist]
    return extract_recipe_from_text(texts, model=model)

if __name__ == "__main__":
    pdf_files = ["test.pdf", "test.pdf"]
    texts = [read_pdf(pdf_file) for pdf_file in pdf_files]
    texts = extract_recipe_from_text(texts, model="gpt-4o-mini")
    print("\n\nExtracted Recipes:\n")
    for i, text in enumerate(texts):
        print(f"Recipe {i + 1}:\n{text}\n\n")