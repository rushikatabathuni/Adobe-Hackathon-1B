import fitz  # PyMuPDF
import re
from collections import defaultdict
import json
import sys

def extract_headings(doc):
    """
    Extracts text lines from a PDF document along with their properties
    like font size, font name, bounding box coordinates, and page number.
    Also includes the page width for later centering calculations.
    """
    lines = []
    for page_num, page in enumerate(doc):
        # Get page dimensions for centering calculation later
        page_width = page.rect.width
        page_height = page.rect.height # Get page height for footer checks
        
        blocks = page.get_text("dict")['blocks']
        for b in blocks:
            for l in b.get("lines", []):
                line_text = " ".join([span["text"] for span in l.get("spans", [])]).strip()
                if not line_text:
                    continue
                
                # Get font sizes and names for all non-empty spans in the line
                font_details = [(span["size"], span["font"], span.get("color", 0)) for span in l["spans"] if span["text"].strip()]
                if not font_details:
                    continue

                # For simplicity, take the average size and the font name of the most prominent span.
                # Prominence is determined by font size.
                avg_size = sum(fd[0] for fd in font_details) / len(font_details)
                
                # Determine the main font name and color for the line.
                # If there are multiple spans, pick the font of the largest span,
                # otherwise just use the first span's font.
                main_font_name = font_details[0][1] 
                main_color = font_details[0][2]
                if len(font_details) > 1:
                    largest_span = max(font_details, key=lambda x: x[0])
                    main_font_name = largest_span[1]
                    main_color = largest_span[2]

                # Ensure bbox exists for all spans to get min/max x0, x1, top, bottom
                valid_spans = [span for span in l['spans'] if 'bbox' in span]
                if not valid_spans:
                    continue

                # Calculate overall bounding box for the line
                x0 = min(span['bbox'][0] for span in valid_spans)
                x1 = max(span['bbox'][2] for span in valid_spans)
                top = min(span['bbox'][1] for span in valid_spans)
                bottom = max(span['bbox'][3] for span in valid_spans)

                # Check for bold/italic style indicators in font name
                font_name_lower = main_font_name.lower()
                is_bold = 'bold' in font_name_lower or 'black' in font_name_lower or 'demi' in font_name_lower or 'heavy' in font_name_lower
                is_italic = 'italic' in font_name_lower or 'oblique' in font_name_lower

                lines.append({
                    "text": line_text,
                    "font_size": avg_size,
                    "font_name": main_font_name,
                    "color": main_color,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "x0": x0,
                    "x1": x1,
                    "top": top,
                    "bottom": bottom,
                    "page": page_num + 1,
                    "page_width": page_width,
                    "page_height": page_height # Add page height here
                })
    return lines

def detect_title(lines):
    """
    Detects the main title of the document from the extracted lines.
    It uses a scoring system based on font size, position, centering, and word count.
    """
    # Filter lines to only those on the first page and within a reasonable vertical area
    first_page_lines = [l for l in lines if l["page"] == 1 and l["top"] < 400]

    if not first_page_lines:
        return ""

    # Calculate the largest font size on the first page to identify prominent text
    all_first_page_font_sizes = [l['font_size'] for l in first_page_lines]
    if not all_first_page_font_sizes:
        return ""
    
    # Get unique font sizes and sort them in descending order to find the largest
    unique_font_sizes = sorted(list(set(all_first_page_font_sizes)), reverse=True)
    
    # The absolute largest font size is a strong indicator for the title
    largest_font_size = unique_font_sizes[0] if unique_font_sizes else 0

    candidates = []
    for l in first_page_lines:
        # Basic disqualification checks for lines unlikely to be a title
        if not l['text'].strip():
            continue
        if l['text'].endswith('.'):
            continue
        if re.search(r'(.)\1{3,}', l['text']): # Avoid lines with repeating characters (e.g., ---)
            continue
        if len(set(l['text'].split())) < 1:
             continue
        if l['font_size'] < 10:
            continue
        if l['text'].lower().startswith(("table of contents", "contents", "abstract", "introduction", "acknowledgements", "preface", "index")):
            continue
        if l['text'].strip().isdigit() or re.match(r'^\d+(\.\d+)*$', l['text'].strip()): # Exclude page numbers or simple section numbers
            continue

        score = 0
        
        # Font size scoring
        if l['font_size'] == largest_font_size:
            score += 10
        elif l['font_size'] >= largest_font_size * 0.8:
            score += 5
        else:
            score += (l['font_size'] / largest_font_size) * 3

        # Vertical Position (higher up is better)
        score += (400 - l['top']) / 40

        # Horizontal Centering
        page_width = l.get('page_width', 600)
        line_center = (l['x0'] + l['x1']) / 2
        page_center = page_width / 2
        
        distance_from_center = abs(line_center - page_center)
        if distance_from_center < 0.1 * page_width: # Closer to center
            score += 4
        elif distance_from_center < 0.2 * page_width:
            score += 2

        # Word Count
        word_count = len(l['text'].split())
        if 2 <= word_count <= 15: # Optimal word count for titles
            score += 3
        elif word_count == 1 or (16 <= word_count <= 25):
            score += 1

        # Capitalization
        if l['text'] and l['text'][0].isupper():
            score += 1
        if l['text'].isupper() and word_count <= 10: # All caps for shorter titles
            score += 2
        
        # Boldness
        if l.get('is_bold', False):
            score += 1

        candidates.append({'line': l, 'score': score})

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_candidate = candidates[0]['line']
    title_text = best_candidate['text']

    # Multi-line title detection: Check if the next highest scoring candidate is directly below and similar in style
    if len(candidates) > 1:
        second_best_candidate_info = candidates[1]
        second_best_line = second_best_candidate_info['line']

        # Check for proximity, similar font size/boldness, and score similarity
        if (0 < (second_best_line['top'] - best_candidate['bottom']) < 20 and # Close vertical proximity
            abs(best_candidate['font_size'] - second_best_line['font_size']) < 2 and # Similar font size
            best_candidate['is_bold'] == second_best_line['is_bold'] and # Same bold status
            abs(candidates[0]['score'] - second_best_candidate_info['score']) < 5): # Scores are close
            
            # Combine lines based on their vertical order
            if best_candidate['top'] < second_best_line['top']:
                title_text += " " + second_best_line['text']
            else:
                title_text = second_best_line['text'] + " " + title_text

    return title_text.strip()

def score_heading(line, all_lines_on_page, previous_line=None):
    """
    Scores a line to determine if it's a potential heading.
    Considers length, position, font size relative to page, and common patterns.
    Enhanced protection against detecting bullet points and numbered list items/sentences.
    Adds scoring for indentation and vertical spacing.
    """
    line_text_stripped = line['text'].strip()

    # Immediate disqualification for very short lines or lines ending with common sentence punctuation
    if len(line_text_stripped) < 3 or line_text_stripped.endswith(('.', ':', ';', ',', '!', '?')):
        return 0

    # Strong disqualification for common bullet point characters or simple single letters/numbers
    bullet_point_only_pattern = re.compile(r'^(?:[\u2022\*\-\–\—>]|\(\s*[a-z]\s*\)|[a-zA-Z]\.)(?:\s+|\t+)') 
    if bullet_point_only_pattern.match(line_text_stripped) or line_text_stripped.strip().isdigit() or len(line_text_stripped) <= 2:
        return 0

    # General disqualification for very long lines that are likely body text
    if len(line_text_stripped.split()) > 15:
        return 0
    
    # Exclude common headers/footers or introductory text based on position
    # Use page_height for a more robust check if available
    page_height = line.get('page_height', 800) # Default to 800 if not found
    if line['top'] < 50 or line['bottom'] > (page_height - 50):
        return 0 # Likely a header/footer

    # Extract content after any numerical/alphabetical/roman prefix for capitalization check
    content_without_prefix_match = re.match(r'^\s*(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?|[A-Z]\.?|\([a-z]\))\s*(.*)', line_text_stripped)
    content_for_case_check = content_without_prefix_match.group(1) if content_without_prefix_match else line_text_stripped
    
    if not content_for_case_check.strip(): # After removing prefix, if it's empty, disqualify
        return 0
    
    # Capitalization analysis: Most significant words should be capitalized (Title Case)
    small_words = {'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'for', 'yet', 'so', 'at', 'by', 'in', 'of', 'on', 'to', 'up', 'as', 'is', 'it', 'with', 'from', 'for', 'vs', 'via'}
    words_in_content = content_for_case_check.split()
    
    if words_in_content:
        lowercase_significant_words = sum(
            1 for word in words_in_content 
            if word and word[0].islower() and word.lower() not in small_words
        )
        total_significant_words = sum(1 for word in words_in_content if word.lower() not in small_words)

        # If too many significant words start with lowercase, it's probably not a heading
        if total_significant_words > 0 and (lowercase_significant_words / total_significant_words) > 0.4: # Increased tolerance slightly
            return 0
        if line_text_stripped.isupper() and len(line_text_stripped.split()) > 20: # Long all-caps lines are often body text
            return 0


    score = 0

    # 1. Numerical/Roman Heading Pattern (Strong indicator)
    if re.match(r'^(?:[0-9]+\.)+(?:\s|$)', line_text_stripped) or re.match(r'^[IVXLCDM]+\.', line_text_stripped):
        score += 5 # Higher score for clear numbering

    # 2. Font size ranking within page (relative to other text on the page)
    # Get all font sizes on the current page to determine relative prominence
    page_font_sizes = sorted(list(set(l['font_size'] for l in all_lines_on_page)), reverse=True)
    if page_font_sizes:
        # Score based on how large this font is compared to the largest on the page
        if line['font_size'] >= page_font_sizes[0]: # Is it the largest?
            score += 4
        elif len(page_font_sizes) > 1 and line['font_size'] >= page_font_sizes[1]: # Is it the second largest?
            score += 2
        else: # Small bonus if it's generally larger than average
            avg_page_font_size = sum(l['font_size'] for l in all_lines_on_page) / max(len(all_lines_on_page), 1)
            if line['font_size'] > avg_page_font_size:
                score += 1
    
    # 3. Position checks (left alignment, reasonable width)
    if 'x0' in line and 'x1' in line:
        if line['x0'] < 100: # Close to left margin (typical for headings)
            score += 2
        
        # Check if the line occupies a reasonable portion of the page width (not too narrow like a list item or too wide like body text)
        line_width = line['x1'] - line['x0']
        page_width = line.get('page_width', 600) # Fallback if not present
        if 0.2 * page_width < line_width < 0.8 * page_width:
            score += 1

    # 4. Style bonuses
    if line.get('is_bold', False):
        score += 3 # Strong indicator
    if line.get('is_italic', False):
        score += 1 # Weaker indicator, but still counts

    # 5. Vertical Spacing (whitespace above the line)
    if previous_line and previous_line['page'] == line['page']:
        # Calculate vertical gap
        vertical_gap = line['top'] - previous_line['bottom']
        # If there's a significant gap (e.g., more than average line spacing)
        # A common line spacing might be around font_size * 1.2 to 1.5.
        # A heading often has a gap of font_size * 2 or more.
        avg_line_height_estimate = line['font_size'] * 1.2
        if vertical_gap > avg_line_height_estimate * 1.5: # More than 1.5 times normal line spacing
            score += 2
        elif vertical_gap > avg_line_height_estimate * 1: # More than normal line spacing
            score += 1
            
    # 6. Negative indicators: Avoid lines that appear multiple times (like running headers/footers)
    # This check is better done globally before calling score_heading, or within score_heading if `all_lines` were passed.
    # For now, keeping the global check outside or assuming it's done.

    return score

def create_style_signature(line):
    """
    Creates a signature for a line's visual style for similarity comparison.
    Returns the raw values rather than a tuple for flexible similarity checking.
    """
    return {
        'font_size': line['font_size'],
        'font_name': line['font_name'],
        'is_bold': line.get('is_bold', False),
        'is_italic': line.get('is_italic', False),
        'color': line.get('color', 0)
    }

def normalize_font_name(font_name):
    """Normalizes font names to a base family for better comparison."""
    base_name = font_name.lower()
    # Remove common suffixes and style indicators
    for suffix in ['-bold', '-italic', '-regular', '-light', '-medium', '-demi', '-heavy', 'bold', 'italic', 'regular', 'light', 'medium', 'demi', 'heavy']:
        base_name = base_name.replace(suffix, '')
    base_name = re.sub(r'[\s\-_]+', '', base_name) # Remove spaces and hyphens/underscores
    return base_name.strip()

def styles_are_similar(style1, style2, font_size_tolerance=1.0): # Reduced tolerance for stricter matching
    """
    Determines if two style signatures are similar enough to be considered the same cluster.
    
    Args:
        style1, style2: Style dictionaries from create_style_signature
        font_size_tolerance: Maximum difference in font size to consider similar
    
    Returns:
        bool: True if styles should be clustered together
    """
    # Font size must be within tolerance
    if abs(style1['font_size'] - style2['font_size']) > font_size_tolerance:
        return False
    
    # Normalized font family should be the same
    if normalize_font_name(style1['font_name']) != normalize_font_name(style2['font_name']):
        return False
    
    # Bold/italic status should match exactly for strong clustering
    if style1['is_bold'] != style2['is_bold']:
        return False
    
    if style1['is_italic'] != style2['is_italic']:
        return False
    
    # Color differences: allow some minor variations but significant differences matter
    # Color is an integer representation (e.g., 0xRRGGBB). A simple absolute diff might be too crude.
    # For simplicity, if they are not identical, ensure they are 'close enough' for common use cases.
    # For now, keeping it as a rough tolerance. Could be improved with color distance metrics.
    if abs(style1['color'] - style2['color']) > 1000000: # Still a rough tolerance, could be made more precise
        return False
    
    return True

def assign_heading_levels_dynamic(headings):
    """
    Dynamically assigns hierarchical levels (H1, H2, H3, etc.) to identified headings
    using dynamic clustering based on visual style similarity. Prioritizes larger fonts/boldness.
    """
    if not headings:
        return []

    # Sort headings primarily by font size (desc), then bold status (desc), then x0 (asc), then page/top.
    # This helps in establishing a natural hierarchy for initial clustering.
    headings_sorted_for_leveling = sorted(
        headings,
        key=lambda h: (h['font_size'], h['is_bold'], -h['x0'], h['page'], h['top']), # Negative x0 means smaller x0 (more left) comes first for same font/bold
        reverse=True # Larger font sizes, then bold come first
    )

    style_to_level_map = {} # Maps a representative style signature to its assigned level (e.g., 1 for H1, 2 for H2)
    next_available_level = 1
    
    outline_with_provisional_levels = []

    for h in headings_sorted_for_leveling:
        stripped_h_text = h['text'].strip()
        if not stripped_h_text:
            continue

        current_style = create_style_signature(h)
        assigned_level = None

        # Try to find an existing cluster for the current style
        # Iterate through items in style_to_level_map to check similarity
        for existing_style_str, level in style_to_level_map.items():
            # Convert the string key back to a dictionary for comparison
            existing_style = json.loads(existing_style_str.replace("'", "\"")) # Assuming single quotes from str(dict)
            if styles_are_similar(current_style, existing_style):
                assigned_level = level
                break
        
        if assigned_level is None:
            # No matching style found, assign a new level based on visual hierarchy.
            # Convert current_style dict to a string (or tuple) to use as a dict key
            # Using json.dumps ensures a consistent string representation for dicts as keys
            style_key = json.dumps(current_style, sort_keys=True) 
            style_to_level_map[style_key] = next_available_level
            assigned_level = next_available_level
            next_available_level += 1
        
        # Override or adjust level based on numerical pattern if more specific
        num_match = re.match(r'^(\d+(?:\.\d+)*)', stripped_h_text)
        if num_match:
            num_pattern_parts = num_match.group(1).split('.')
            level_from_pattern = len(num_pattern_parts)
            
            # If the numerical pattern suggests a deeper level than the current assigned visual level,
            # and the current level is not already extremely high (e.g., H6),
            # adjust it. This helps correct cases where visual similarity might group different numerical levels.
            if level_from_pattern > assigned_level and assigned_level < 5: # Limit depth based on pattern
                assigned_level = level_from_pattern
                # Update the style map if this numerical pattern indicates a more specific level for this style
                style_key = json.dumps(current_style, sort_keys=True)
                style_to_level_map[style_key] = assigned_level


        outline_with_provisional_levels.append({
            "text": stripped_h_text,
            "page": h['page'],
            "level_rank": assigned_level, # Store as integer for sorting
            "original_line_data": h # Keep original line data for sorting later
        })
    
    # Now sort by page and top, then adjust levels to be sequential (H1, H2, ...)
    # This step is crucial because the initial sorting for style clustering might not be page-sequential.
    outline_with_provisional_levels.sort(key=lambda x: (x['original_line_data']['page'], x['original_line_data']['top']))

    final_outline = []
    # Map the "level_rank" to H1, H2, etc., ensuring a contiguous hierarchy
    # The lowest level_rank found will be H1, the next H2, and so on.
    unique_levels_found = sorted(list(set(item['level_rank'] for item in outline_with_provisional_levels)))
    level_rank_to_h_map = {rank: i + 1 for i, rank in enumerate(unique_levels_found)}

    previous_h_level = 0
    for item in outline_with_provisional_levels:
        current_h_level = level_rank_to_h_map.get(item['level_rank'], 1)

        # Ensure that levels increase by at most 1 at a time (e.g., H1 -> H3 is not allowed, must be H1 -> H2 -> H3)
        if current_h_level > previous_h_level + 1 and previous_h_level != 0:
            current_h_level = previous_h_level + 1 # Clamp to ensure logical hierarchy progression

        final_outline.append({
            "text": item['text'],
            "page": item['page'],
            "level": f"H{current_h_level}"
        })
        previous_h_level = current_h_level
    
    return final_outline

def build_outline_from_toc(doc):
    """
    Builds an outline using the PDF's embedded Table of Contents (TOC).
    """
    toc = doc.get_toc()
    if not toc or len(toc) < 3: # Require at least 3 entries to consider it a valid TOC
        return None

    outline = []
    for entry in toc:
        level, title, page = entry[:3]
        stripped_title = title.strip()
        # Add a check here: only include if the stripped title is not empty
        if stripped_title:
            outline.append({
                "level": f"H{level}",
                "text": stripped_title,
                "page": page
            })

    lines = extract_headings(doc) # Still extract headings to detect title
    detected_title = detect_title(lines) or ""

    return {
        "title": detected_title,
        "outline": outline
    }

def build_outline_heuristic(doc):
    """
    Builds an outline using heuristic methods with dynamic clustering
    when an embedded TOC is not available or insufficient.
    """
    lines = extract_headings(doc)
    title = detect_title(lines) or ""

    # Group lines by page for context-aware scoring (e.g., for relative font sizes)
    lines_by_page = defaultdict(list)
    for line in lines:
        lines_by_page[line['page']].append(line)

    heading_candidates = []
    # To pass previous_line to score_heading, we need to iterate sequentially
    previous_line_on_page = {} # Dictionary to store the last line processed for each page

    # Iterate through lines in page and then vertical order
    # Ensure lines list is already sorted by page and 'top' from extract_headings for correct previous_line context
    # `extract_headings` produces lines in page-then-top order, so this loop is fine.
    for line in lines:
        # Skip the detected main title
        if line['text'].strip() == title:
            continue
        
        # Get the previous line on the current page for vertical spacing calculation
        prev_line = previous_line_on_page.get(line['page'])
        
        score = score_heading(line, lines_by_page[line['page']], prev_line)
        if score >= 3:  # Threshold for potential heading (can be tuned)
            heading_candidates.append(line) # Append the full line object

        previous_line_on_page[line['page']] = line # Update the previous line for the current page

    # Sort heading candidates by page and then by their vertical position (top)
    # This is crucial for correct hierarchical assignment later.
    # This sort is important even if lines are already sorted, as scoring might reorder effective candidates.
    heading_candidates.sort(key=lambda l: (l['page'], l['top']))
    
    # Use dynamic clustering for level assignment
    outline = []
    seen = set() # To avoid duplicate headings if identical lines are identified
    for h in assign_heading_levels_dynamic(heading_candidates):
        key = (h['text'], h['page'], h['level']) # Include level in key for uniqueness
        if key not in seen:
            seen.add(key)
            outline.append(h)

    return {
        "title": title,
        "outline": outline
    }
    
def extract_outline_and_text(pdf_path):
    """
    Extracts the outline and raw page text from a PDF document,
    writes it (optional), and returns the result dictionary.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: {
            "title": str,
            "outline": [...],
            "style_info": ..., # Note: style_info is not currently gathered in this output
            "page_text": {page_num: text}
        }
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error_data = {"error": f"Could not open PDF file at {pdf_path}: {e}"}
        # In a real environment, you might log this or re-raise, but for a script, exit is fine.
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        sys.exit(1) # Exit upon critical error

    # Try TOC first, fallback to heuristic
    result = build_outline_from_toc(doc)
    if not result or not result.get("outline"):
        result = build_outline_heuristic(doc)

    # Extract page-wise text (required for 1B)
    page_text = {i + 1: page.get_text() for i, page in enumerate(doc)}
    result["page_text"] = page_text

    # Optional: still write output.json for debugging/inspection
    # Ensure this file is handled carefully in a multi-threaded or production environment
    try:
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write output.json: {e}")
    
    doc.close() # Close the document after processing
    return result

if __name__ == '__main__':
    # Use a dummy path for testing if no argument is provided
    # In a real environment, you'd likely pass a valid PDF path
    path = sys.argv[1] if len(sys.argv) > 1 else "dummy_input.pdf" 
    
    # For actual testing, replace "dummy_input.pdf" with a path to a real PDF file
    # For example: path = "path/to/your/document.pdf"
    
    if path == "dummy_input.pdf":
        print(json.dumps({"error": "Please provide a PDF file path as an argument. Example: python pdf_utils.py your_document.pdf"}, indent=2))
        sys.exit(1)

    try:
        # The main logic is now encapsulated in extract_outline_and_text
        output_data = extract_outline_and_text(path)
        print(json.dumps(output_data, indent=2))
    except Exception as e:
        print(json.dumps({"error": f"An unexpected error occurred during processing: {e}"}, indent=2))
        sys.exit(1)