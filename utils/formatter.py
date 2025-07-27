# /app/utils/formatter.py

def format_output(input_data, extracted_sections, refined_paragraphs, timestamp):
    """
    Builds the final output JSON structure.

    Args:
        input_data (dict): Original input JSON (contains persona, job, docs)
        extracted_sections (list): Top N sections [{document, title, page, rank}]
        refined_paragraphs (list): Paragraphs [{document, refined_text, page_number}]
        timestamp (str): ISO datetime string

    Returns:
        dict: Final output JSON
    """

    return {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": timestamp
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": sec["section_title"],
                "importance_rank": sec["importance_rank"],
                "page_number": sec["page_number"]
            }
            for sec in extracted_sections
        ],
        "subsection_analysis": [
            {
                "document": para["document"],
                "refined_text": para["refined_text"],
                "page_number": para["page_number"]
            }
            for para in refined_paragraphs
        ]
    }
