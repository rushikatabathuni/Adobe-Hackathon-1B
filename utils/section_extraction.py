# /app/utils/section_extraction.py
import re

def extract_sections_from_outline(outline_json, page_texts, filename):
    """
    Convert a PDF's outline and text into structured sections.
    This version uses 'top' (y-coordinate) information from the outline
    to precisely segment content within pages.

    Args:
        outline_json (dict): Contains 'title' and 'outline' (list of heading dicts, now including 'top').
        page_texts (dict): Page-wise plain text {page_num: text}.
        filename (str): The name of the PDF document.

    Returns:
        list of dicts: [
            {
                "document": filename,
                "section_title": str,
                "page_number": int,
                "content": str
            },
            ...
        ]
    """

    outline = outline_json.get("outline", [])
    title = outline_json.get("title", filename)

    # Fallback: entire document as one section if no outline
    if not outline:
        full_content = "\n".join(page_texts.get(p, "") for p in sorted(page_texts.keys()))
        return [{
            "document": filename,
            "section_title": title,
            "page_number": 0,
            "content": full_content.strip()
        }]

    # Sort outline entries by page and then by their 'top' coordinate (y-position)
    # This is crucial for correct sequential processing of headings on the same page.
    outline_sorted = sorted(outline, key=lambda h: (h["page"], h.get("top", 0)))

    sections = []
    
    # Prepare a list of all pages' text with line-by-line information and y-coordinates
    # This is a simplification. A more robust solution might require parsing lines with their exact BBoxes.
    # For now, we'll search for the heading text within the page content and assume its position.

    for i, current_heading in enumerate(outline_sorted):
        section_title = current_heading["text"].strip()
        start_page = current_heading["page"]
        start_y = current_heading.get("top", -1) # Y-coordinate of the heading text on its page

        content_parts = []
        
        # Determine the end point of the current section
        end_page = start_page
        end_y = float('inf') # Default to end of page/document

        if i + 1 < len(outline_sorted):
            next_heading = outline_sorted[i + 1]
            next_heading_page = next_heading["page"]
            next_heading_y = next_heading.get("top", -1)

            if next_heading_page == start_page:
                # Next heading is on the same page: content ends just before the next heading's text
                end_page = start_page
                end_y = next_heading_y
            else:
                # Next heading is on a later page: content spans to the end of the page before next heading
                end_page = next_heading_page - 1
                end_y = float('inf') # Content goes to the end of this end_page
        else:
            # This is the last section, content spans to the end of the document
            end_page = max(page_texts.keys()) if page_texts else start_page
            end_y = float('inf') # Content goes to the end of this end_page

        # Ensure end_page is never less than start_page
        end_page = max(start_page, end_page)

        # Extract content for the determined page range and y-coordinates
        for p_num in range(start_page, end_page + 1):
            page_content = page_texts.get(p_num, "")
            
            if p_num == start_page:
                # For the starting page, content begins after the current heading's text (or its y-coordinate)
                
                # Try to find the heading's exact position to start content *after* it
                heading_start_idx = page_content.find(section_title)
                if heading_start_idx != -1:
                    # Content starts right after the title, potentially after a newline
                    content_start = heading_start_idx + len(section_title)
                    # Find the first real newline after the title for cleaner start
                    newline_after_title = page_content.find('\n', content_start)
                    if newline_after_title != -1:
                        content_start = newline_after_title + 1
                    
                    current_page_segment = page_content[content_start:]
                else:
                    # Fallback: if title not found or no y-coordinate, take from the start of the page
                    current_page_segment = page_content
                
                if p_num == end_page and end_y != float('inf'):
                    # If current and next heading on same page, clip at next heading's y-coord.
                    # This is a heuristic as we don't have line-level y-coords for all text.
                    # A more robust solution would re-parse page text with coordinates.
                    # For now, we'll try to find the next heading's text and clip there.
                    if i + 1 < len(outline_sorted) and outline_sorted[i+1]["page"] == p_num:
                        next_heading_text = outline_sorted[i+1]["text"].strip()
                        next_heading_idx = current_page_segment.find(next_heading_text)
                        if next_heading_idx != -1:
                            current_page_segment = current_page_segment[:next_heading_idx]
                
                content_parts.append(current_page_segment)

            elif p_num == end_page and end_y != float('inf'):
                # For the ending page (if it's not the start page AND next heading is on this page)
                # content should end before the next heading's text.
                if i + 1 < len(outline_sorted) and outline_sorted[i+1]["page"] == p_num:
                    next_heading_text = outline_sorted[i+1]["text"].strip()
                    next_heading_idx = page_content.find(next_heading_text)
                    if next_heading_idx != -1:
                        content_parts.append(page_content[:next_heading_idx])
                    else:
                        content_parts.append(page_content) # Fallback to full page if next heading not found
                else:
                    content_parts.append(page_content) # Default: include full page if no specific end Y
            else:
                # For intermediate pages, include full content
                content_parts.append(page_content)
        
        full_section_content = "\n".join(content_parts).strip()
        
        # If the content is empty or only whitespace after processing, skip it
        if not full_section_content:
            continue

        sections.append({
            "document": filename,
            "section_title": section_title,
            "page_number": start_page,
            "content": full_section_content
        })

    return sections