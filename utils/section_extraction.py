# /app/utils/section_extraction.py

def extract_sections_from_outline(outline_json, page_texts, filename):
    """
    Convert a PDF's outline and text into structured sections.
    
    Args:
        outline_json (dict): Contains 'title' and 'outline' (list of heading dicts).
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

    # Fallback: entire document as one section
    if not outline:
        return [{
            "document": filename,
            "section_title": title,
            "page_number": 1,
            "content": "\n".join(page_texts.get(p, "") for p in sorted(page_texts))
        }]

    # Sort outline entries by page (and optionally level)
    outline_sorted = sorted(outline, key=lambda h: h["page"])

    sections = []
    for i, current in enumerate(outline_sorted):
        start_page = current["page"]
        
        # --- START OF CHANGE ---
        # Calculate end_page for the current section's content span
        if i + 1 < len(outline_sorted):
            next_heading_page = outline_sorted[i + 1]["page"]
            
            # If the next heading is on the same page, this section's content
            # should ideally end just before the next heading's content starts.
            # However, without detailed line coordinates, we can only grab full pages.
            # So, if next heading is on same page, this section gets content only from its start_page.
            # If next heading is on a later page, this section gets content up to the page before the next heading.
            if next_heading_page == start_page:
                end_page = start_page # Content only from the current page
            else:
                end_page = next_heading_page - 1 # Content up to the page before the next heading
        else:
            # This is the last section, so its content spans to the end of the document
            end_page = max(page_texts.keys()) if page_texts else start_page
        
        # Ensure end_page is never less than start_page
        end_page = max(start_page, end_page)
        # --- END OF CHANGE ---

        content = ""
        for p in range(start_page, end_page + 1):
            if p in page_texts:
                content += f"\n{page_texts[p]}"

        sections.append({
            "document": filename,
            "section_title": current["text"].strip(),
            "page_number": start_page,
            "content": content.strip()
        })

    return sections