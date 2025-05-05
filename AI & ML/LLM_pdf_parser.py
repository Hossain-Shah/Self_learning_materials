# importing dependencies
import pymupdf4llm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from typing import Optional
import os
from google.colab import userdata
import torch

"""# OCR processed with LLM, NER processed with BERT"""

# Initialize NER extraction pipeline
extract_ner_with_bert = pipeline(
    "ner",
    model="jplu/tf-xlm-r-ner-40-lang",
    tokenizer=(
        'jplu/tf-xlm-r-ner-40-lang',
        {"use_fast": True}),
    framework="tf"
)

# Function to process and extract NER for any text
def extract_ner(text):
    if text and isinstance(text, str) and text.strip():  # Ensure text is a string and not None or empty/whitespace
        ner_entities = extract_ner_with_bert(text)
        return ner_entities
    else:
        return []

# Get information from the PDF
info = pymupdf4llm.to_markdown(
    doc="/content/drive/MyDrive/inbound8315523191781495843.pdf",
    page_chunks=True,
    write_images=True,
    image_path="/content/drive/MyDrive/Colab_Notebooks/",
    image_format="jpg",
    dpi=200,
)

# Function to print structured information with real-time NER extraction
def print_info(info):
    # Print metadata details
    print("Metadata:")
    metadata = info[0]['metadata']
    for key, value in metadata.items():
        print(f"{key}: {value}")
        # Extract and print NER for metadata
        ner_entities = extract_ner(value)
        print(f"NER for metadata: {ner_entities}")

    # Print table of contents items
    print("\nTable of Contents:")
    toc_items = info[0]['toc_items']
    for item in toc_items:
        print(f"Level: {item[0]}, Title: {item[1]}, Page: {item[2]}")
        # Extract and print NER for Table of Contents title
        ner_entities = extract_ner(item[1])
        print(f"NER for Table of Contents title: {ner_entities}")

    # Print tables information
    print("\nTables:")
    tables = info[0]['tables']
    for table in tables:
        print(f"Bounding box: {table['bbox']}, Rows: {table['rows']}, Columns: {table['columns']}")
        # Extract and print NER for tables (if necessary, from bounding box or other attributes)
        ner_entities = extract_ner(str(table['bbox']))  # or another relevant part of the table
        print(f"NER for table bounding box: {ner_entities}")

    # Print images information
    print("\nImages:")
    images = info[0]['images']
    for image in images:
        print(f"Image number: {image['number']}")
        print(f"Bounding box: {image['bbox']}")
        print(f"Transform: {image['transform']}")
        print(f"Width: {image['width']}, Height: {image['height']}")
        print(f"Color space: {image['cs-name']}, Resolution: {image['xres']}x{image['yres']}")
        # Extract and print NER for image data (if relevant)
        ner_entities = extract_ner(f"Image number {image['number']} details")
        print(f"NER for image info: {ner_entities}")

    # Print extracted text
    print("\nExtracted Text:")
    text = info[0]['text']
    print(text)
    # Extract and print NER for extracted text
    ner_entities = extract_ner(text)
    print(f"NER for extracted text: {ner_entities}")

# Call function to print info
print_info(info)

"""# OCR processed with LLM, NER processed with GPT"""

# Function to generate the prompt for NER extraction
def generate_prompt(input_text: str, instruction: Optional[str] = None) -> str:
    text = f"### Question: {input_text}\n\n### Answer: "
    if instruction:
        text = f"### Instruction: {instruction}\n\n{text}"
    return text

# Extract information from the PDF (OCR process)
info = pymupdf4llm.to_markdown(
    doc="/content/drive/MyDrive/inbound8315523191781495843.pdf",
    page_chunks=True,
    write_images=True,
    image_path="/content/drive/MyDrive/Colab_Notebooks/",
    image_format="jpg",
    dpi=200,
)

# Function to process OCR-extracted text and use the Gemma model for NER-like extraction
def generate_ner(text: str):
    # Construct a prompt asking the model to extract named entities from the text
    prompt = f"### Extract all named entities (people, organizations, locations, etc.) from the following text:\n\n{text}\n\n### Named Entities:"

    # Inference using the "gemma-2b" model
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = merged_model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# Initialize model for further analysis
hugging_face_token = os.getenv('hugging_face_token')
if not hugging_face_token:
    hugging_face_token = userdata.get('hugging_face_token')
    os.environ["hugging_face_token"] = hugging_face_token

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token=hugging_face_token)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=hugging_face_token)

lora_model = PeftModel.from_pretrained(base_model, "vdpappu/lora_medicalqa")
merged_model = lora_model.merge_and_unload()

eos_token = '<eos>'
eos_token_id = tokenizer.encode(eos_token, add_special_tokens=False)[-1]

generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id,
    min_length=5,
    max_length=1000,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3,
    early_stopping=True
)

# Function to print structured information with real-time NER extraction using Gemma
def print_info(info):
    # Print metadata details
    print("Metadata:")
    # Extract and process Metadata
    metadata = info[0]['metadata']
    for key, value in metadata.items():
        print(f"{key}: {value}")
        # Extract and print NER for metadata
        ner_entities = generate_ner(value)
        print(f"NER for metadata: {ner_entities}")

    # Print table of contents items
    print("\nTable of Contents:")
    # Extract and process Table of Contents (TOC) items
    toc_items = info[0]['toc_items']
    for item in toc_items:
        print(f"Level: {item[0]}, Title: {item[1]}, Page: {item[2]}")
        # Extract and print NER for Table of Contents title
        ner_entities = generate_ner(item[1])
        print(f"NER for Table of Contents title: {ner_entities}")

    # Print tables information
    print("\nTables:")
    # Extract and process Tables
    tables = info[0]['tables']
    for table in tables:
        print(f"Bounding box: {table['bbox']}, Rows: {table['rows']}, Columns: {table['columns']}")
        # Extract and print NER for tables (if necessary, from bounding box or other attributes)
        ner_entities = generate_ner(str(table['bbox']))  # or another relevant part of the table
        print(f"NER for table bounding box: {ner_entities}")

    # Print images information
    print("\nImages:")
    # Extract and process Images
    images = info[0]['images']
    for image in images:
        print(f"Image number: {image['number']}")
        print(f"Bounding box: {image['bbox']}")
        print(f"Transform: {image['transform']}")
        print(f"Width: {image['width']}, Height: {image['height']}")
        print(f"Color space: {image['cs-name']}, Resolution: {image['xres']}x{image['yres']}")
        # Extract and print NER for image data (if relevant)
        ner_entities = generate_ner(f"Image number {image['number']} details")
        print(f"NER for image info: {ner_entities}")

    # Print extracted text
    print("\nExtracted Text:")
    # Extract and process Extracted Text
    text = info[0]['text']
    print(text)
    # Extract and print NER for extracted text
    ner_entities = generate_ner(text)
    print(f"NER for extracted text: {ner_entities}")

# Call the function to process and print the info with NER extraction
print_info(info)
