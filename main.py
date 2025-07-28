import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, CrossEncoder # Import CrossEncoder

# Ensure utils are available
try:
    from utils.pdf_utils import extract_outline_and_text
    from utils.section_extraction import extract_sections_from_outline
    from utils.relevance_ranking import rank_sections, extract_top_paragraphs # Assuming these are the updated ones
    from utils.formatter import format_output
except ImportError:
    print("Warning: Could not import utility functions from utils/. Using dummy placeholders.")
    # Dummy placeholders for demonstration if utils are not available
    def extract_outline_and_text(pdf_path):
        print(f"Dummy: Extracting outline and text from {pdf_path}")
        return {"title": "Dummy Title", "outline": [], "page_text": {1: "Dummy page content."}}
    def extract_sections_from_outline(outline_json, page_texts, filename):
        print("Dummy: Extracting sections from outline")
        return [{"document": filename, "section_title": "Dummy Section", "content": "This is dummy content for a section.", "page_number": 1}]
    # These will be replaced by the actual updated functions you provided
    # The actual updated functions are provided in the previous turn's response.
    # For a complete runnable example, you'd copy those into utils/relevance_ranking.py
    # For now, I'll put a simplified dummy here to allow the main function to run.
    def rank_sections(sections, persona_role, job_task, bi_encoder_model, cross_encoder_model, top_n=5):
        print("Dummy: Ranking sections")
        if not sections: return []
        # Simulate some ranking
        for i, sec in enumerate(sections):
            sec['importance_rank'] = i + 1
        return sections[:top_n]
    def extract_top_paragraphs(section_text, query, page_number, document, bi_encoder_model, cross_encoder_model, section_title="", top_k=1):
        print("Dummy: Extracting top paragraphs")
        return [{"document": document, "refined_text": f"Dummy snippet for '{query}' from '{section_title}'.", "page_number": page_number}]
    def format_output(input_data, top_sections, top_paragraphs, timestamp):
        print("Dummy: Formatting output")
        return {"summary": "Dummy output summary", "timestamp": timestamp}


# Load models once globally
# Using all-MiniLM-L12-v2 as bi-encoder (your original choice)
bi_encoder_model = SentenceTransformer("paraphrase_minilm_l12/")

# Using a MiniLM-based cross-encoder for efficiency and good performance
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
    
    # Validate required fields
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
        print(f"⚠️ Warning: PDF file not found: {pdf_path}")
        return filename, []
    
    try:
        print(f"📄 Processing {filename}...")
        result = extract_outline_and_text(pdf_path)
        
        # Validate extraction result
        if not result:
            print(f"⚠️ Warning: No content extracted from {filename}")
            return filename, []
        
        outline_json = {
            "title": result.get("title", filename),
            "outline": result.get("outline", [])
        }
        page_texts = result.get("page_text", {})
        
        if not page_texts:
            print(f"⚠️ Warning: No page text extracted from {filename}")
            return filename, []
        
        sections = extract_sections_from_outline(outline_json, page_texts, filename)
        
        if not sections:
            print(f"⚠️ Warning: No sections extracted from {filename}")
            return filename, []
        
        print(f"✅ Extracted {len(sections)} sections from {filename}")
        return filename, sections
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return filename, []

def validate_sections(all_sections):
    """
    Validate and clean extracted sections.
    Remove sections with insufficient content.
    """
    valid_sections = []
    removed_count = 0
    
    for section in all_sections:
        # Check for required fields
        if not all(key in section for key in ["document", "section_title", "page_number"]):
            removed_count += 1
            continue
        
        # Check content quality
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
            print(f"⚠️ Warning: Insufficient content for section '{title}' in {doc}")
            return []
        
        # Extract highly focused snippets, passing both models
        snippets = extract_top_paragraphs(
            content, job_task, page, doc, 
            bi_encoder_model=bi_encoder_model, cross_encoder_model=cross_encoder_model, # Pass both models
            section_title=title, top_k=1
        )
        
        # Validate snippet quality
        validated_snippets = []
        for snippet in snippets:
            refined_text = snippet.get("refined_text", "").strip()
            word_count = len(refined_text.split())
            
            # Strict quality control
            if 5 <= word_count <= 60:  # Much stricter word limits
                validated_snippets.append(snippet)
            elif word_count > 60:
                # Emergency truncation
                words = refined_text.split()
                truncated = " ".join(words[:50]) + "..."
                snippet["refined_text"] = truncated
                validated_snippets.append(snippet)
                print(f"⚠️ Truncated overly long snippet in '{title}'")
            else:
                print(f"⚠️ Rejected snippet in '{title}' (too short: {word_count} words)")
        
        return validated_snippets
        
    except Exception as e:
        print(f"❌ Error extracting snippets from '{sec.get('section_title', 'Unknown')}': {str(e)}")
        return []

def main():
    """
    Enhanced main processing function with better error handling,
    validation, and quality analysis.
    """
    try:
        # Step 0: Load and validate input
        input_data = load_input_json()
        print("✅ Loaded and validated input JSON")
        
        documents = input_data["documents"]
        persona_role = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]
        
        print(f"👤 Persona: {persona_role}")
        print(f"🎯 Task: {job_task[:100]}{'...' if len(job_task) > 100 else ''}")
        print(f"📚 Documents to process: {len(documents)}")
        
        # Step 1: Parallel PDF Processing with improved error handling
        print("\n🔄 Starting PDF processing...")
        all_sections = []
        section_text_lookup = {}
        successful_docs = 0
        
        with ThreadPoolExecutor(max_workers=min(8, len(documents))) as executor:
            futures = [executor.submit(process_pdf, doc) for doc in documents]
            
            for future in as_completed(futures):
                filename, sections = future.result()
                if sections:  # Only count successful extractions
                    all_sections.extend(sections)
                    successful_docs += 1
                    
                    # Build lookup table for paragraph extraction
                    for sec in sections:
                        key = (sec["document"], sec["section_title"])
                        section_text_lookup[key] = sec.get("content", "")
        
        print(f"✅ Successfully processed {successful_docs}/{len(documents)} documents")
        
        if not all_sections:
            raise RuntimeError("❌ No sections extracted from any document. Check PDF files and extraction logic.")
        
        # Step 1.5: Validate and clean sections
        all_sections = validate_sections(all_sections)
        
        if not all_sections:
            raise RuntimeError("❌ No valid sections remaining after quality filtering.")
        
        # Step 2: Rank sections with enhanced algorithm, passing both models
        print("🔍 Ranking sections using enhanced algorithm (Bi-encoder + Cross-encoder)...")
        top_sections = rank_sections(
            sections=all_sections, 
            persona_role=persona_role, 
            job_task=job_task, 
            bi_encoder_model=bi_encoder_model, # Pass bi-encoder
            cross_encoder_model=cross_encoder_model, # Pass cross-encoder
            top_n=5
        )
        
        if not top_sections:
            raise RuntimeError("❌ No sections ranked. Check ranking algorithm.")
        
        print(f"✅ Selected top {len(top_sections)} sections")
        
        # Step 3: **ORDERED** Paragraph Extraction
        print("🧠 Extracting refined paragraphs...")
        top_paragraphs = []
        
        print("📝 Processing sections in parallel while preserving order...")
        max_workers = min(5, len(top_sections))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit futures with their index to preserve order, passing both models
            future_to_index = {
                executor.submit(extract_para_wrapper, sec, section_text_lookup, bi_encoder_model, cross_encoder_model, job_task): i # Pass both models
                for i, sec in enumerate(top_sections)
            }
            
            # Collect results in order
            results = [None] * len(top_sections)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
            
            # Add results to top_paragraphs in the correct order
            for paragraphs in results:
                if paragraphs:
                    top_paragraphs.extend(paragraphs)
        
        print(f"✅ Extracted {len(top_paragraphs)} refined paragraphs")
        
        # Step 3.5: Verify order continuity
        print("🔍 Verifying section-paragraph continuity...")
        if top_paragraphs:
            # This check is a bit simplified, it assumes a 1:1 or 1:many mapping where
            # paragraphs from a section appear consecutively. For true continuity,
            # you'd need to track which section each paragraph came from more explicitly.
            # For now, it checks if the document source matches the top sections' order.
            current_top_section_idx = 0
            for i, para in enumerate(top_paragraphs):
                if current_top_section_idx < len(top_sections):
                    expected_doc = top_sections[current_top_section_idx].get("document", "")
                    actual_doc = para.get("document", "")
                    
                    if expected_doc != actual_doc:
                        # If the document changes, advance the top_section_idx
                        # This is a heuristic and might need refinement based on exact output requirements
                        found_next_section = False
                        for j in range(current_top_section_idx + 1, len(top_sections)):
                            if top_sections[j].get("document", "") == actual_doc:
                                current_top_section_idx = j
                                found_next_section = True
                                break
                        if not found_next_section:
                             print(f"⚠️ Warning: Document order mismatch at paragraph {i+1}. Expected doc from top section {current_top_section_idx+1} ('{expected_doc}'), got '{actual_doc}'.")
                             # If we can't find the next section, it's a real mismatch
                             break
                else:
                    # More paragraphs than top sections, or something unexpected
                    print(f"⚠️ Warning: More paragraphs than top sections, or unexpected paragraph at position {i+1}.")
                    break
            else:
                print("✅ Section-paragraph order verified successfully (heuristic check)")
        else:
            print("No paragraphs to verify order for.")
        
        # Step 4: Format and Write Output
        print("📝 Formatting output...")
        timestamp = datetime.now().isoformat()
        output_json = format_output(input_data, top_sections, top_paragraphs, timestamp)
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Write output with error handling
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            print(f"✅ Output written to {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Error writing output file: {str(e)}")
            raise
        
        # Final summary
        print(f"\n🎉 Processing completed successfully!")
        print(f"   🔗 Order Continuity: Maintained (via parallel executor order preservation)")
        
    except Exception as e:
        print(f"\n💥 Fatal error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()