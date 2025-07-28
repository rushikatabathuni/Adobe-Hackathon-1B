import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils.pdf_utils import extract_outline_and_text
from utils.section_extraction import extract_sections_from_outline
from utils.relevance_ranking import rank_sections, extract_top_paragraphs # Assuming these are the updated ones
from utils.formatter import format_output

bi_encoder_model = SentenceTransformer("paraphrase_minilm_l12/")
cross_encoder_model = CrossEncoder("cross-encoder-ms-marco-MiniLM-L-12-v2/")

INPUT_DIR = "input"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

def load_input_json():
    """Load and validate input JSON configuration."""
    path = os.path.join(INPUT_DIR, "input.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input JSON at {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    required_fields = ["documents", "persona", "job_to_be_done"]
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field '{field}' in input JSON")
    
    if "role" not in input_data["persona"]:
        raise ValueError("Missing 'role' in persona configuration")
    
    if "task" not in input_data["job_to_be_done"]:
        raise ValueError("Missing 'task' in job_to_be_done configuration")
    
    return input_data

def process_pdf(doc):
    """
    Process a single PDF document to extract sections.
    Enhanced with better error handling and validation.
    """
    filename = doc["filename"]
    pdf_path = os.path.join(INPUT_DIR, "pdf", filename)
    
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file not found: {pdf_path}")
        return filename, []
    try:
        print(f"Processing {filename}...")
        result = extract_outline_and_text(pdf_path)
        if not result:
            print(f"⚠Warning: No content extracted from {filename}")
            return filename, []
        
        outline_json = {
            "title": result.get("title", filename),
            "outline": result.get("outline", [])
        }
        page_texts = result.get("page_text", {})
        
        if not page_texts:
            print(f"Warning: No page text extracted from {filename}")
            return filename, []
        
        sections = extract_sections_from_outline(outline_json, page_texts, filename)
        
        if not sections:
            print(f"Warning: No sections extracted from {filename}")
            return filename, []
        print(f"Extracted {len(sections)} sections from {filename}")
        return filename, sections
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return filename, []

def validate_sections(all_sections):
    """
    Validate and clean extracted sections.
    Remove sections with insufficient content.
    """
    valid_sections = []
    removed_count = 0
    
    for section in all_sections:
        if not all(key in section for key in ["document", "section_title", "page_number"]):
            removed_count += 1
            continue
        content = section.get("content", "")
        title = section.get("section_title", "")
        
        # Require either meaningful content or a descriptive title
        if len(content.strip()) < 10 and len(title.strip()) < 3:
            removed_count += 1
            continue
        
        # Ensure page number is valid
        try:
            page_num = int(section.get("page_number", 0))
            if page_num < 1:
                section["page_number"] = 1
        except (ValueError, TypeError):
            section["page_number"] = 1
        
        valid_sections.append(section)
    
    if removed_count > 0:
        print(f"🧹 Removed {removed_count} low-quality sections")
    
    return valid_sections

def extract_para_wrapper(sec, section_text_lookup, bi_encoder_model, cross_encoder_model, job_task): # Updated signature
    """
    Wrapper function for paragraph extraction with strict content refinement.
    Now passes both bi_encoder_model and cross_encoder_model.
    """
    try:
        doc = sec["document"]
        title = sec["section_title"]
        page = sec["page_number"]
        key = (doc, title)
        content = section_text_lookup.get(key, "")
        
        if not content or len(content.strip()) < 30:
            print(f"Warning: Insufficient content for section '{title}' in {doc}")
            return []
        
        snippets = extract_top_paragraphs(
            content, job_task, page, doc, 
            bi_encoder_model=bi_encoder_model, cross_encoder_model=cross_encoder_model, # Pass both models
            section_title=title, top_k=1
        )
        
        validated_snippets = []
        for snippet in snippets:
            refined_text = snippet.get("refined_text", "").strip()
            word_count = len(refined_text.split())
            
            if 5 <= word_count <= 60:  # Much stricter word limits
                validated_snippets.append(snippet)
            elif word_count > 60:
                words = refined_text.split()
                truncated = " ".join(words[:50]) + "..."
                snippet["refined_text"] = truncated
                validated_snippets.append(snippet)
                print(f"Truncated long snippet in '{title}'")
            else:
                print(f"Rejected snippet in '{title}' (too short: {word_count} words)")
        
        return validated_snippets
        
    except Exception as e:
        print(f"Error extracting snippets from '{sec.get('section_title', 'Unknown')}': {str(e)}")
        return []

def main():
    """
    Enhanced main processing function with better error handling,
    validation, and quality analysis.
    """
    try:
        input_data = load_input_json()
        print("✅ Loaded and validated input JSON")
        
        documents = input_data["documents"]
        persona_role = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]
        
        print(f"Persona: {persona_role}")
        print(f"Task: {job_task[:100]}{'...' if len(job_task) > 100 else ''}")
        print(f"Documents to process: {len(documents)}")
        
        print("\nStarting PDF processing...")
        all_sections = []
        section_text_lookup = {}
        successful_docs = 0
        
        with ThreadPoolExecutor(max_workers=min(8, len(documents))) as executor:
            futures = [executor.submit(process_pdf, doc) for doc in documents]
            
            for future in as_completed(futures):
                filename, sections = future.result()
                if sections:
                    all_sections.extend(sections)
                    successful_docs += 1
                    
                    for sec in sections:
                        key = (sec["document"], sec["section_title"])
                        section_text_lookup[key] = sec.get("content", "")
        
        print(f"Successfully processed {successful_docs}/{len(documents)} documents")
        
        if not all_sections:
            raise RuntimeError("No sections extracted from any document.")
        
        all_sections = validate_sections(all_sections)
        
        if not all_sections:
            raise RuntimeError("No valid sections remaining after quality filtering.")
        
        print("🔍 Ranking sections using Bi-encoder + Cross-encoder:")
        top_sections = rank_sections(
            sections=all_sections, 
            persona_role=persona_role, 
            job_task=job_task, 
            bi_encoder_model=bi_encoder_model, # Pass bi-encoder
            cross_encoder_model=cross_encoder_model, # Pass cross-encoder
            top_n=5
        )
        
        if not top_sections:
            raise RuntimeError("No sections ranked.")
        
        print(f"Selected top {len(top_sections)} sections")
        print("Extracting refined paragraphs:")
        top_paragraphs = []
        max_workers = min(5, len(top_sections))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(extract_para_wrapper, sec, section_text_lookup, bi_encoder_model, cross_encoder_model, job_task): i # Pass both models
                for i, sec in enumerate(top_sections)
            }
            results = [None] * len(top_sections)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
            for paragraphs in results:
                if paragraphs:
                    top_paragraphs.extend(paragraphs)
        
        print(f"Extracted {len(top_paragraphs)} refined paragraphs")
        
        print("Verifying section-paragraph continuity: ")
        if top_paragraphs:
            current_top_section_idx = 0
            for i, para in enumerate(top_paragraphs):
                if current_top_section_idx < len(top_sections):
                    expected_doc = top_sections[current_top_section_idx].get("document", "")
                    actual_doc = para.get("document", "")
                    
                    if expected_doc != actual_doc:
                        found_next_section = False
                        for j in range(current_top_section_idx + 1, len(top_sections)):
                            if top_sections[j].get("document", "") == actual_doc:
                                current_top_section_idx = j
                                found_next_section = True
                                break
                        if not found_next_section:
                             print(f"Warning: Document order mismatch at paragraph {i+1}. Expected doc from top section {current_top_section_idx+1} ('{expected_doc}'), got '{actual_doc}'.")
                             break
                else:
                    print(f"Warning: More paragraphs than top sections, or unexpected paragraph at position {i+1}.")
                    break
            else:
                print("Section-paragraph order verified successfully (heuristic check)")
        else:
            print("No paragraphs recieved to verify order.")
        print(".............................")
        timestamp = datetime.now().isoformat()
        output_json = format_output(input_data, top_sections, top_paragraphs, timestamp)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            print(f"Output written to {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error writing output file: {str(e)}")
            raise
        print(f"\nProcessing completed successfully.")
        
    except Exception as e:
        print(f"\nError occured in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
