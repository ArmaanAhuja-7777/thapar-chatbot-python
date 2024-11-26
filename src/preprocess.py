import spacy
import random
import json
import re

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Function to clean and preprocess the text (remove unwanted characters and excessive spaces)
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces and newlines
    text = re.sub(r"[^\w\s.,:;!?'-]", "", text)  # Remove non-text characters
    return text.strip()

# Function to generate a variety of random question-answer pairs
def generate_random_questions(text, max_questions=50):
    qa_pairs = []

    # Process the text using spaCy NLP to extract named entities, sentences, etc.
    doc = nlp(text)

    # Extract named entities (persons, organizations, dates, locations, etc.)
    entities = {ent.text: ent.label_ for ent in doc.ents}

    # Create questions about entities
    for entity, label in entities.items():
        # Person-related questions
        if label == "PERSON":
            qa_pairs.append({"question": f"Who is {entity}?", "answer": entity})
            qa_pairs.append({"question": f"Tell me about {entity}.", "answer": entity})
        
        # Organization-related questions
        elif label == "ORG":
            qa_pairs.append({"question": f"What is {entity}?", "answer": entity})
            qa_pairs.append({"question": f"Where is {entity} located?", "answer": entity})
        
        # Geopolitical entities (locations)
        elif label == "GPE":
            qa_pairs.append({"question": f"Where is {entity} located?", "answer": entity})
            qa_pairs.append({"question": f"What is the capital of {entity}?", "answer": entity})
        
        # Date-related questions
        elif label == "DATE":
            qa_pairs.append({"question": f"When did {entity} happen?", "answer": entity})
            qa_pairs.append({"question": f"What happened in {entity}?", "answer": entity})
        
        # Time-based (period) questions
        elif label == "TIME":
            qa_pairs.append({"question": f"When did this occur?", "answer": entity})

        # Stop if we've reached the maximum number of questions
        if len(qa_pairs) >= max_questions:
            break

    # Generate questions based on custom keywords related to birth, contact, phone, etc.
    if "born" in text.lower():
        birth_date = re.search(r"born\s*on\s*([\w\s,]+)", text, re.IGNORECASE)
        if birth_date:
            qa_pairs.append({"question": f"When was the person born?", "answer": birth_date.group(1)})
        
        if len(qa_pairs) >= max_questions:
            return qa_pairs

    # Handle phone numbers (simple phone format recognition)
    phone_numbers = re.findall(r"\+?\d{1,4}[\s\-]?\(?\d{1,3}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}", text)
    for phone in phone_numbers:
        qa_pairs.append({"question": "What is the contact number?", "answer": phone})
        if len(qa_pairs) >= max_questions:
            return qa_pairs

    # Handle emails (simple email format recognition)
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    for email in emails:
        qa_pairs.append({"question": "What is the email address?", "answer": email})
        if len(qa_pairs) >= max_questions:
            return qa_pairs

    # Locations
    locations = re.findall(r"located in (\w+|\w+\s\w+|\w+\s\w+\s\w+)", text)
    for location in locations:
        qa_pairs.append({"question": f"Where is {location} located?", "answer": location})
        if len(qa_pairs) >= max_questions:
            return qa_pairs

    # Create questions based on sentence patterns
    sentences = [sent.text for sent in doc.sents]
    random.shuffle(sentences)  # Randomize sentence order for diversity

    for sentence in sentences:
        # General questions based on the presence of certain keywords
        if "collaboration" in sentence.lower() or "agreement" in sentence.lower():
            question = "What is the recent collaboration or agreement?"
            answer = sentence
            qa_pairs.append({"question": question, "answer": answer})

        # Questions about locations and campus
        if "campus" in sentence.lower():
            question = "What is the size of the campus?"
            answer = re.search(r"\d+-acre campus", sentence)
            if answer:
                qa_pairs.append({"question": question, "answer": answer.group(0)})

        # Time-based questions
        if "date" in sentence.lower() or "year" in sentence.lower():
            question = "When did this event occur?"
            answer = sentence
            qa_pairs.append({"question": question, "answer": answer})

        # What questions for definitions or descriptions
        if "school" in sentence.lower() or "institute" in sentence.lower():
            question = "What is this institution?"
            answer = sentence
            qa_pairs.append({"question": question, "answer": answer})

        # Stop if we've reached the maximum number of questions
        if len(qa_pairs) >= max_questions:
            break

    # Return the generated questions, ensuring we don't exceed the max limit
    return qa_pairs[:max_questions]

# Function to process the raw text from a JSON file and generate Q&A pairs
def process_json_file(input_file, output_file):
    """
    Processes raw text data from a JSON file and generates Q&A pairs from the entire dataset in the file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check if data is loaded correctly
    print(f"Loaded {len(data)} records from the JSON file.")
    
    qa_pairs = []

    # Iterate over each entry in the JSON and extract content
    for entry in data:
        content = entry.get("content", "")
        cleaned_text = clean_text(content)

        # Check if cleaned text is not empty
        if cleaned_text:
            print(f"Processing content: {cleaned_text[:100]}...")  # Print the first 100 characters for reference

            # Generate random question-answer pairs from the cleaned text
            qa_pairs.extend(generate_random_questions(cleaned_text, max_questions=50))

        # Stop if we've reached the maximum number of questions
        if len(qa_pairs) >= 50:
            break

    # Check if any QA pairs were generated
    if not qa_pairs:
        print("No questions were generated from the content.")
    else:
        print(f"Generated {len(qa_pairs)} Q&A pairs.")

    # Save the generated questions and answers to a file
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, qa in enumerate(qa_pairs, start=1):  # Use enumerate to get the index
            print(f"Generating Question {idx}:")
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}\n")
            f.write(f"Q: {qa['question']}\nA: {qa['answer']}\n\n")

    print(f"Generated Q&A pairs saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "./data/raw/thapar_scraped.json"  # Input JSON file path
    output_file = "./data/cleaned/random_qa_pairs.json"  # Output Q&A file path
    process_json_file(input_file, output_file)
