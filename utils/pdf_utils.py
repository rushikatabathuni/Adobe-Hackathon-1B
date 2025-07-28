import fitz 
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
        page_width = page.rect.width
        page_height = page.rect.height
        
        blocks = page.get_text("dict")['blocks']
        for b in blocks:
            for l in b.get("lines", []):
                line_text = " ".join([span["text"] for span in l.get("spans", [])]).strip()
                if not line_text:
                    continue
                font_details = [(span["size"], span["font"], span.get("color", 0)) for span in l["spans"] if span["text"].strip()]
                if not font_details:
                    continue
                    
                avg_size = sum(fd[0] for fd in font_details) / len(font_details)
                
                # Determine the main font name and color for the line
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

                x0 = min(span['bbox'][0] for span in valid_spans)
                x1 = max(span['bbox'][2] for span in valid_spans)
                top = min(span['bbox'][1] for span in valid_spans)
                bottom = max(span['bbox'][3] for span in valid_spans)

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
    first_page_lines = [l for l in lines if l["page"] == 1 and l["top"] < 400]

    if not first_page_lines:
        return ""

    all_first_page_font_sizes = [l['font_size'] for l in first_page_lines]
    if not all_first_page_font_sizes:
        return ""
    
    unique_font_sizes = sorted(list(set(all_first_page_font_sizes)), reverse=True)
    
    largest_font_size = unique_font_sizes[0] if unique_font_sizes else 0

    candidates = []
    for l in first_page_lines:
        # Basic disqualification checks for lines unlikely to be a title
        if not l['text'].strip():
            continue
        if l['text'].endswith('.'):
            continue
        if re.search(r'(.)\1{3,}', l['text']):
            continue
        if len(set(l['text'].split())) < 1:
             continue
        if l['font_size'] < 10:
            continue
        if l['text'].lower().startswith(("table of contents", "contents", "abstract", "introduction", "acknowledgements", "preface", "index")):
            continue
        if l['text'].strip().isdigit() or re.match(r'^\d+(\.\d+)*$', l['text'].strip()): 
            continue

        score = 0
        
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

    if len(candidates) > 1:
        second_best_candidate_info = candidates[1]
        second_best_line = second_best_candidate_info['line']

        if (0 < (second_best_line['top'] - best_candidate['bottom']) < 20 and # Close vertical proximity
            abs(best_candidate['font_size'] - second_best_line['font_size']) < 2 and # Similar font size
            best_candidate['is_bold'] == second_best_line['is_bold'] and # Same bold status
            abs(candidates[0]['score'] - second_best_candidate_info['score']) < 5): # Scores are close
            
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
    page_height = line.get('page_height', 800)
    if line['top'] < 50 or line['bottom'] > (page_height - 50):
        return 0 

    content_without_prefix_match = re.match(r'^\s*(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?|[A-Z]\.?|\([a-z]\))\s*(.*)', line_text_stripped)
    content_for_case_check = content_without_prefix_match.group(1) if content_without_prefix_match else line_text_stripped
    
    if not content_for_case_check.strip():
        return 0
    
    small_words = {'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'for', 'yet', 'so', 'at', 'by', 'in', 'of', 'on', 'to', 'up', 'as', 'is', 'it', 'with', 'from', 'for', 'vs', 'via'}
    words_in_content = content_for_case_check.split()
    
    if words_in_content:
        lowercase_significant_words = sum(
            1 for word in words_in_content 
            if word and word[0].islower() and word.lower() not in small_words
        )
        total_significant_words = sum(1 for word in words_in_content if word.lower() not in small_words)

        if total_significant_words > 0 and (lowercase_significant_words / total_significant_words) > 0.4:
            return 0
        if line_text_stripped.isupper() and len(line_text_stripped.split()) > 20:
            return 0


    score = 0

    if re.match(r'^(?:[0-9]+\.)+(?:\s|$)', line_text_stripped) or re.match(r'^[IVXLCDM]+\.', line_text_stripped):
        score += 5

    # 2. Font size ranking within page (relative to other text on the page)
    page_font_sizes = sorted(list(set(l['font_size'] for l in all_lines_on_page)), reverse=True)
    if page_font_sizes:
        if line['font_size'] >= page_font_sizes[0]:
            score += 4
        elif len(page_font_sizes) > 1 and line['font_size'] >= page_font_sizes[1]:
            score += 2
        else: 
            avg_page_font_size = sum(l['font_size'] for l in all_lines_on_page) / max(len(all_lines_on_page), 1)
            if line['font_size'] > avg_page_font_size:
                score += 1
    
    if 'x0' in line and 'x1' in line:
        if line['x0'] < 100: 
            score += 2
        
        line_width = line['x1'] - line['x0']
        page_width = line.get('page_width', 600) # Fallback if not present
        if 0.2 * page_width < line_width < 0.8 * page_width:
            score += 1
    if line.get('is_bold', False):
        score += 3 # Strong indicator
    if line.get('is_italic', False):
        score += 1

    # 5. Vertical Spacing (whitespace above the line)
    if previous_line and previous_line['page'] == line['page']:
        vertical_gap = line['top'] - previous_line['bottom']
        avg_line_height_estimate = line['font_size'] * 1.2
        if vertical_gap > avg_line_height_estimate * 1.5: # More than 1.5 times normal line spacing
            score += 2
        elif vertical_gap > avg_line_height_estimate * 1: 
            score += 1
            
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
    for suffix in ['-bold', '-italic', '-regular', '-light', '-medium', '-demi', '-heavy', 'bold', 'italic', 'regular', 'light', 'medium', 'demi', 'heavy']:
        base_name = base_name.replace(suffix, '')
    base_name = re.sub(r'[\s\-_]+', '', base_name)
    return base_name.strip()

def styles_are_similar(style1, style2, font_size_tolerance=1.0):
    """
    Determines if two style signatures are similar enough to be considered the same cluster.
    
    Args:
        style1, style2: Style dictionaries from create_style_signature
        font_size_tolerance: Maximum difference in font size to consider similar
    
    Returns:
        bool: True if styles should be clustered together
    """
    if abs(style1['font_size'] - style2['font_size']) > font_size_tolerance:
        return False
    
    if normalize_font_name(style1['font_name']) != normalize_font_name(style2['font_name']):
        return False
    
    if style1['is_bold'] != style2['is_bold']:
        return False
    
    if style1['is_italic'] != style2['is_italic']:
        return False
    
    # Color differences
    if abs(style1['color'] - style2['color']) > 1000000: 
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
        key=lambda h: (h['font_size'], h['is_bold'], -h['x0'], h['page'], h['top']),
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
            existing_style = json.loads(existing_style_str.replace("'", "\"")) 
            if styles_are_similar(current_style, existing_style):
                assigned_level = level
                break
        
        if assigned_level is None:
            # No matching style found, assign a new level based on visual hierarchy.
            style_key = json.dumps(current_style, sort_keys=True) 
            style_to_level_map[style_key] = next_available_level
            assigned_level = next_available_level
            next_available_level += 1
        
        num_match = re.match(r'^(\d+(?:\.\d+)*)', stripped_h_text)
        if num_match:
            num_pattern_parts = num_match.group(1).split('.')
            level_from_pattern = len(num_pattern_parts)
            
            if level_from_pattern > assigned_level and assigned_level < 5:
                assigned_level = level_from_pattern
                style_key = json.dumps(current_style, sort_keys=True)
                style_to_level_map[style_key] = assigned_level


        outline_with_provisional_levels.append({
            "text": stripped_h_text,
            "page": h['page'],
            "level_rank": assigned_level,
            "original_line_data": h 
        })
    
    outline_with_provisional_levels.sort(key=lambda x: (x['original_line_data']['page'], x['original_line_data']['top']))

    final_outline = []
    unique_levels_found = sorted(list(set(item['level_rank'] for item in outline_with_provisional_levels)))
    level_rank_to_h_map = {rank: i + 1 for i, rank in enumerate(unique_levels_found)}

    previous_h_level = 0
    for item in outline_with_provisional_levels:
        current_h_level = level_rank_to_h_map.get(item['level_rank'], 1)

        if current_h_level > previous_h_level + 1 and previous_h_level != 0:
            current_h_level = previous_h_level + 1 

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
    if not toc or len(toc) < 3: 
        return None

    outline = []
    for entry in toc:
        level, title, page = entry[:3]
        stripped_title = title.strip()
        if stripped_title:
            outline.append({
                "level": f"H{level}",
                "text": stripped_title,
                "page": page
            })

    lines = extract_headings(doc) 
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

    lines_by_page = defaultdict(list)
    for line in lines:
        lines_by_page[line['page']].append(line)

    heading_candidates = []
    previous_line_on_page = {} 

    for line in lines:
        if line['text'].strip() == title:
            continue
        
        prev_line = previous_line_on_page.get(line['page'])
        
        score = score_heading(line, lines_by_page[line['page']], prev_line)
        if score >= 3: 
            heading_candidates.append(line)

        previous_line_on_page[line['page']] = line 

    heading_candidates.sort(key=lambda l: (l['page'], l['top']))
    
    outline = []
    seen = set() 
    for h in assign_heading_levels_dynamic(heading_candidates):
        key = (h['text'], h['page'], h['level'])
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
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        sys.exit(1) 

    result = build_outline_from_toc(doc)
    if not result or not result.get("outline"):
        result = build_outline_heuristic(doc)

    page_text = {i + 1: page.get_text() for i, page in enumerate(doc)}
    result["page_text"] = page_text

    try:
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write output.json: {e}")
    
    doc.close()
    return result

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf" 
    
    
    if path == "sample.pdf":
        print(json.dumps({"error": "Please provide a PDF file path as an argument. Example: python pdf_utils.py your_document.pdf"}, indent=2))
        sys.exit(1)

    try:
        output_data = extract_outline_and_text(path)
        print(json.dumps(output_data, indent=2))
    except Exception as e:
        print(json.dumps({"error": f"An unexpected error occurred during processing: {e}"}, indent=2))
        sys.exit(1)
