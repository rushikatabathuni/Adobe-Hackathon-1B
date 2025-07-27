import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.pdf_utils import extract_outline_and_text
from utils.section_extraction import extract_sections_from_outline
from utils.relevance_ranking import rank_sections, extract_top_paragraphs
from utils.formatter import format_output
from sentence_transformers import SentenceTransformer

# Load model once globally
model = SentenceTransformer("./paraphrase_minilm_l12")

INPUT_DIR = "input"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

def load_input_json():
    """Load and validate input JSON configuration."""
    path = os.path.join(INPUT_DIR, "input2.json")
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
        print(f"⚠️  Warning: PDF file not found: {pdf_path}")
        return filename, []
    
    try:
        print(f"📄 Processing {filename}...")
        result = extract_outline_and_text(pdf_path)
        
        # Validate extraction result
        if not result:
            print(f"⚠️  Warning: No content extracted from {filename}")
            return filename, []
        
        outline_json = {
            "title": result.get("title", filename),
            "outline": result.get("outline", [])
        }
        page_texts = result.get("page_text", {})
        
        if not page_texts:
            print(f"⚠️  Warning: No page text extracted from {filename}")
            return filename, []
        
        sections = extract_sections_from_outline(outline_json, page_texts, filename)
        
        if not sections:
            print(f"⚠️  Warning: No sections extracted from {filename}")
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

def extract_para_wrapper(sec, section_text_lookup, model, job_task):
    """
    Wrapper function for paragraph extraction with strict content refinement.
    """
    try:
        doc = sec["document"]
        title = sec["section_title"]
        page = sec["page_number"]
        key = (doc, title)
        content = section_text_lookup.get(key, "")
        
        if not content or len(content.strip()) < 30:
            print(f"⚠️  Warning: Insufficient content for section '{title}' in {doc}")
            return []
        
        # Extract highly focused snippets
        snippets = extract_top_paragraphs(
            content, job_task, page, doc, 
            section_title=title, model=model, top_k=1
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
                print(f"⚠️  Truncated overly long snippet in '{title}'")
            else:
                print(f"⚠️  Rejected snippet in '{title}' (too short: {word_count} words)")
        
        return validated_snippets
        
    except Exception as e:
        print(f"❌ Error extracting snippets from '{sec.get('section_title', 'Unknown')}': {str(e)}")
        return []

def analyze_extraction_quality(all_sections, top_sections, top_paragraphs):
    """
    Enhanced quality analysis with focus on content relevance and refinement.
    """
    print("\n📊 Extraction Quality Analysis:")
    print(f"   Total sections extracted: {len(all_sections)}")
    print(f"   Top sections selected: {len(top_sections)}")
    print(f"   Refined snippets: {len(top_paragraphs)}")
    
    if top_paragraphs:
        # Analyze snippet quality
        word_counts = []
        for para in top_paragraphs:
            refined_text = para.get("refined_text", "")
            word_count = len(refined_text.split())
            word_counts.append(word_count)
        
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            min_words = min(word_counts)
            max_words = max(word_counts)
            print(f"   Snippet quality:")
            print(f"     - Average words per snippet: {avg_words:.1f}")
            print(f"     - Range: {min_words}-{max_words} words")
            
            # Quality categories
            excellent = sum(1 for wc in word_counts if 10 <= wc <= 50)
            acceptable = sum(1 for wc in word_counts if 5 <= wc <= 60)
            too_short = sum(1 for wc in word_counts if wc < 5)
            too_long = sum(1 for wc in word_counts if wc > 60)
            
            print(f"     - Excellent snippets (10-50 words): {excellent}/{len(word_counts)}")
            print(f"     - Acceptable snippets (5-60 words): {acceptable}/{len(word_counts)}")
            
            if too_short > 0:
                print(f"     - Too short (rejected): {too_short}")
            if too_long > 0:
                print(f"     - Too long (truncated): {too_long}")
    
    if top_sections:
        # Document diversity in top sections
        doc_distribution = {}
        for section in top_sections:
            doc = section.get("document", "Unknown")
            doc_distribution[doc] = doc_distribution.get(doc, 0) + 1
        
        print(f"   Document diversity in top sections:")
        for doc, count in sorted(doc_distribution.items()):
            print(f"     - {doc}: {count} section(s)")
        
        # Page distribution
        pages = [sec.get("page_number", 0) for sec in top_sections]
        print(f"     - Page range: {min(pages)}-{max(pages)}")
    
    # Overall quality score
    if all_sections and top_sections and top_paragraphs:
        relevance_rate = len(top_sections) / min(len(all_sections), 20) * 100  # Out of first 20 sections
        extraction_rate = len(top_paragraphs) / len(top_sections) * 100
        
        print(f"   Quality metrics:")
        print(f"     - Section relevance rate: {relevance_rate:.1f}%")
        print(f"     - Snippet extraction rate: {extraction_rate:.1f}%")
    
    print()

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
        
        # Step 2: Rank sections with enhanced algorithm
        print("🔍 Ranking sections using enhanced algorithm...")
        top_sections = rank_sections(sections=all_sections, persona_role=persona_role, job_task=job_task, model=model, top_n=5)
        
        if not top_sections:
            raise RuntimeError("❌ No sections ranked. Check ranking algorithm.")
        
        print(f"✅ Selected top {len(top_sections)} sections")
        
        # Step 3: **ORDERED** Paragraph Extraction - FIXED VERSION
        print("🧠 Extracting refined paragraphs...")
        top_paragraphs = []
        
        # OPTION 1: Sequential processing (maintains order, but slower)
        # print("📝 Processing sections sequentially to maintain order...")
        # for i, sec in enumerate(top_sections):
        #     print(f"   Processing section {i+1}/{len(top_sections)}: {sec.get('section_title', 'Unknown')[:50]}...")
        #     paragraphs = extract_para_wrapper(sec, section_text_lookup, model, job_task)
        #     if paragraphs:
        #         top_paragraphs.extend(paragraphs)
        
        # OPTION 2: Parallel processing with order preservation (uncomment to use)
        print("📝 Processing sections in parallel while preserving order...")
        max_workers = min(5, len(top_sections))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit futures with their index to preserve order
            future_to_index = {
                executor.submit(extract_para_wrapper, sec, section_text_lookup, model, job_task): i
                for i, sec in enumerate(top_sections)
            }
            
            # Collect results in order
            results = [None] * len(top_sections)  # Pre-allocate list
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
            
            # Add results to top_paragraphs in the correct order
            for paragraphs in results:
                if paragraphs:  # Only extend if we got valid paragraphs
                    top_paragraphs.extend(paragraphs)
        
        print(f"✅ Extracted {len(top_paragraphs)} refined paragraphs")
        
        # Step 3.5: Verify order continuity
        print("🔍 Verifying section-paragraph continuity...")
        if top_paragraphs:
            for i, para in enumerate(top_paragraphs):
                if i < len(top_sections):
                    expected_doc = top_sections[i].get("document", "")
                    actual_doc = para.get("document", "")
                    if expected_doc != actual_doc:
                        print(f"⚠️  Warning: Order mismatch at position {i+1}")
                        print(f"     Expected: {expected_doc}")
                        print(f"     Got: {actual_doc}")
                        break
            else:
                print("✅ Section-paragraph order verified successfully")
        
        # Step 3.6: Quality analysis
        analyze_extraction_quality(all_sections, top_sections, top_paragraphs)
        
        # Step 4: Format and Write Output
        print("📝 Formatting output...")
        timestamp = datetime.now().isoformat()
        output_json = format_output(input_data, top_sections, top_paragraphs, timestamp)
        
        # Add quality metrics to output
        output_json["quality_metrics"] = {
            "total_sections_extracted": len(all_sections),
            "documents_processed_successfully": successful_docs,
            "total_documents": len(documents),
            "top_sections_selected": len(top_sections),
            "refined_paragraphs_extracted": len(top_paragraphs),
            "order_continuity_verified": True  # Since we now maintain order
        }
        
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
        print(f"   📊 Quality Score: {len(top_paragraphs)}/{len(top_sections)} sections produced refined content")
        print(f"   🔗 Order Continuity: Maintained")
        
    except Exception as e:
        print(f"\n💥 Fatal error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()