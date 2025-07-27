from collections import Counter
import torch
from sentence_transformers import util, SentenceTransformer
import re
import unicodedata
import math
import numpy as np
import os # Import os for path joining


# --- Utility Functions ---

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text

def extract_requirements_and_constraints(job_task, model, batch_size=32):
    """
    Extract explicit requirements and constraints from the task using a hybrid approach
    of rule-based pattern matching and semantic analysis (embeddings).
    This helps the model focus on critical aspects of the task.
    """
    task_lower = clean_text(job_task).lower()
    
    # Define common positive and negative indicators for rule-based extraction
    positive_indicators = ["include", "for", "with", "contains", "featuring", "must have", "should have", "about", "related to", "such as"]
    negative_indicators = ["exclude", "without", "not including", "avoid", "no", "not", "except"]
    
    extracted_positive_terms = []
    extracted_negative_terms = []
    
    # Simple rule-based extraction for explicit constraints
    # This regex attempts to capture phrases after indicators until another indicator, punctuation, or end of string.
    indicator_pattern = "|".join(re.escape(i) for i in (positive_indicators + negative_indicators))
    
    for indicator in positive_indicators:
        # Pattern: indicator + whitespace + (capture group for terms) + (non-capturing group for delimiters/next indicator)
        matches = re.findall(rf'{re.escape(indicator)}\s+([a-zA-Z0-9\s,\-]+?)(?=\s*(?:{indicator_pattern}|[.,;?!]|\Z))', task_lower, re.IGNORECASE)
        for m in matches:
            terms = [t.strip() for t in m.split(',') if t.strip()]
            extracted_positive_terms.extend(terms)
    
    for indicator in negative_indicators:
        matches = re.findall(rf'{re.escape(indicator)}\s+([a-zA-Z0-9\s,\-]+?)(?=\s*(?:{indicator_pattern}|[.,;?!]|\Z))', task_lower, re.IGNORECASE)
        for m in matches:
            terms = [t.strip() for t in m.split(',') if t.strip()]
            extracted_negative_terms.extend(terms)

    # General keywords from the task (useful even if no explicit indicators)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', task_lower)
    key_words = [w for w in words if len(w) >= 4]
    
    # Clean and deduplicate extracted terms
    extracted_positive_terms = list(set([clean_text(t) for t in extracted_positive_terms if t]))
    extracted_negative_terms = list(set([clean_text(t) for t in extracted_negative_terms if t]))
    key_words = list(set([clean_text(w) for w in key_words if w]))

    # Combine all terms to get embeddings efficiently
    all_relevant_phrases = list(set(extracted_positive_terms + extracted_negative_terms + key_words))
    
    try:
        with torch.no_grad():
            task_embedding = model.encode(job_task, convert_to_tensor=True)
            
            term_embeddings = {}
            if all_relevant_phrases:
                # Batch encode all relevant phrases
                encoded_phrases = model.encode(all_relevant_phrases, convert_to_tensor=True, batch_size=batch_size)
                for i, phrase in enumerate(all_relevant_phrases):
                    term_embeddings[phrase] = encoded_phrases[i]

        return {
            'task_embedding': task_embedding,
            'positive_terms': extracted_positive_terms,
            'negative_terms': extracted_negative_terms,
            'key_words': key_words, # General keywords
            'term_embeddings': term_embeddings, # Embeddings for individual terms
            'full_task': job_task
        }
    except Exception as e:
        print(f"Warning: Error in extracting task embeddings/terms: {e}. Falling back to basic keywords.")
        return {
            'task_embedding': None,
            'positive_terms': [],
            'negative_terms': [],
            'key_words': key_words,
            'term_embeddings': {},
            'full_task': job_task
        }

def calculate_semantic_alignment(section, task_requirements, model, batch_size=32):
    """
    Calculate how well a section aligns with task requirements using semantic similarity
    and phrase-level matching.
    """
    title = clean_text(section.get("section_title", ""))
    content = clean_text(section.get("content", ""))
    
    if not title and not content:
        return 0.0
    
    # Combine title and content, with emphasis on title (repeat title)
    # Truncate content for embedding efficiency (MiniLM benefits from less noise)
    section_text = f"{title}. {title}. {content[:750]}" # Increased content slightly
    
    try:
        with torch.no_grad():
            section_embedding = model.encode(section_text, convert_to_tensor=True)
            task_embedding = task_requirements.get('task_embedding')
            
            if task_embedding is not None:
                # Primary semantic similarity
                semantic_score = util.cos_sim(task_embedding, section_embedding).item()
                
                # Phrase-level matching for more precise alignment using positive terms
                positive_terms = task_requirements.get('positive_terms', [])
                phrase_scores = []
                
                if positive_terms:
                    # Get embeddings for positive terms from pre-computed map
                    positive_term_embeddings = [task_requirements['term_embeddings'][p] for p in positive_terms if p in task_requirements['term_embeddings']]
                    
                    if positive_term_embeddings:
                        # Calculate similarity of each positive term to the section
                        # Using model.encode on a list of tensors should be efficient
                        term_sims = util.cos_sim(torch.stack(positive_term_embeddings), section_embedding).flatten().tolist()
                        phrase_scores.extend(term_sims)
                        
                        if phrase_scores:
                            # Average of top N phrase scores for a more robust signal
                            phrase_scores.sort(reverse=True)
                            # Consider top 2-3 most relevant phrases
                            avg_top_phrases_score = np.mean(phrase_scores[:min(len(phrase_scores), 3)])
                            
                            # Combine semantic and phrase scores (tunable weights)
                            combined_score = (semantic_score * 0.7) + (avg_top_phrases_score * 0.3)
                        else:
                            combined_score = semantic_score
                    else: # No embeddings for positive terms
                        combined_score = semantic_score
                else: # No positive terms extracted
                    combined_score = semantic_score
                
                return combined_score
            else:
                # Fallback to keyword matching if task embedding failed
                return calculate_keyword_overlap(section_text, task_requirements)
                
    except Exception as e:
        print(f"Error calculating semantic alignment: {e}. Falling back to keyword overlap.")
        # Fallback to keyword matching if embedding failed for section
        return calculate_keyword_overlap(section_text, task_requirements)

def calculate_keyword_overlap(section_text, task_requirements):
    """
    Fallback method using keyword overlap when embeddings are not available or fail.
    """
    section_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', section_text.lower()))
    task_words = set(task_requirements.get('key_words', []))
    
    if not task_words:
        return 0.0
    
    overlap = len(section_words.intersection(task_words))
    # Normalize by the number of task words to get a ratio
    overlap_ratio = overlap / len(task_words)
    
    # Cap the fallback score to prevent it from outcompeting semantic scores
    return min(0.8, overlap_ratio)

def detect_semantic_contradictions(section, task_requirements, model, batch_size=32):
    """
    Detect if a section semantically contradicts the task requirements.
    This uses a combination of explicit negative term matching and semantic dissimilarity.
    """
    title = clean_text(section.get("section_title", ""))
    content = clean_text(section.get("content", ""))
    
    section_text_lower = (title + " " + content).lower()
    
    # 1. Explicit String Matching for Negative Terms (Fast & Direct)
    negative_terms = task_requirements.get('negative_terms', [])
    if negative_terms:
        for term in negative_terms:
            if term and term in section_text_lower: # Check for direct presence of negative terms
                # print(f"DEBUG: Detected explicit negative term '{term}' in section. Skipping.")
                return True 

    # 2. Semantic Contradiction Check (More Nuanced)
    section_text_for_contradiction = f"{title}. {content[:500]}" # Use enough content for semantic signal
    if not section_text_for_contradiction.strip():
        return False # Nothing to contradict

    try:
        with torch.no_grad():
            task_embedding = task_requirements.get('task_embedding')
            if task_embedding is None:
                return False # Cannot perform semantic check without task embedding
            
            section_embedding = model.encode(section_text_for_contradiction, convert_to_tensor=True, batch_size=batch_size)
            
            # Overall similarity to task (very low similarity suggests irrelevance, not necessarily contradiction)
            task_similarity = util.cos_sim(task_embedding, section_embedding).item()
            
            # If the section is extremely irrelevant (very low similarity), we can filter it early.
            # This is a soft filter, not a direct contradiction.
            if task_similarity < 0.05: # Tunable threshold for extreme irrelevance
                return True # Treat as effectively contradictory/unusable

            # Check if the section is semantically similar to *negative* concepts
            for neg_term in negative_terms:
                neg_term_emb = task_requirements['term_embeddings'].get(neg_term)
                if neg_term_emb is not None:
                    # How similar is the section to this specific negative term?
                    neg_sim_to_section = util.cos_sim(neg_term_emb, section_embedding).item()
                    
                    # If section is highly similar to a negative term (e.g., > 0.65)
                    # AND its overall similarity to the *positive* task is relatively low (<0.4),
                    # then it's likely a contradiction. Tunable thresholds are crucial here.
                    if neg_sim_to_section > 0.65 and task_similarity < 0.4:
                        # print(f"DEBUG: Detected semantic opposition for '{neg_term}' (sim: {neg_sim_to_section:.2f}, task_sim: {task_similarity:.2f}). Skipping.")
                        return True
            
            return False
            
    except Exception as e:
        print(f"Error in semantic contradiction detection: {e}. Defaulting to no contradiction.")
        return False

def filter_and_deduplicate_sections(sections):
    """
    Remove duplicate and near-duplicate sections based on content similarity and normalized titles.
    """
    if not sections:
        return []
    
    filtered_sections = []
    seen_content_hashes = set()
    seen_titles_normalized = set()
    
    for section in sections:
        title = clean_text(section.get("section_title", ""))
        content = clean_text(section.get("content", ""))
        
        # Create content hash for exact duplicate detection (using a snippet for speed)
        content_hash = hash(f"{title}|{content[:200]}") # Only hash first 200 chars of content
        
        if content_hash in seen_content_hashes:
            continue
        
        # Normalize title for near-duplicate detection (word-based Jaccard)
        title_normalized_str = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower()).strip()
        
        # Simple check for very similar titles after normalization (e.g., "Introduction" vs "introduction")
        if title_normalized_str in seen_titles_normalized:
            continue

        # Jaccard similarity for more nuanced title similarity (e.g., "Executive Summary" vs "Summary for Executives")
        is_near_duplicate_title = False
        current_title_words = set(title_normalized_str.split())
        
        if current_title_words: # Only check if current title has words
            for seen_title_norm_str in list(seen_titles_normalized): # Iterate over a copy
                seen_title_words = set(seen_title_norm_str.split())
                if seen_title_words:
                    intersection = len(current_title_words.intersection(seen_title_words))
                    union = len(current_title_words.union(seen_title_words))
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    if jaccard_sim > 0.8:  # 80% similarity threshold for titles
                        is_near_duplicate_title = True
                        break
        
        if not is_near_duplicate_title:
            seen_content_hashes.add(content_hash)
            seen_titles_normalized.add(title_normalized_str)
            filtered_sections.append(section)
    
    return filtered_sections

def rank_sections(sections, persona_role, job_task, model,top_n=5, batch_size=32):
    """
    Redesigned ranking system focusing on semantic alignment and contradiction detection,
    optimized for smaller models and general purpose use.
    """
    if not sections:
        return []
    
    print(f"🔍 Analyzing task: {job_task[:100]}...")
    
    # Step 1: Extract task requirements and constraints (semantic signals for MiniLM)
    task_requirements = extract_requirements_and_constraints(job_task, model, batch_size=batch_size)
    
    # Step 2: Filter duplicates early to reduce processing load
    unique_sections = filter_and_deduplicate_sections(sections)
    print(f"📝 Filtered to {len(unique_sections)} unique sections from {len(sections)} total")
    
    if not unique_sections:
        return []
    
    # Step 3: Score sections and filter contradictions
    valid_sections = []
    contradiction_count = 0
    
    # Batch process sections for embedding and scoring
    sections_to_embed = []
    original_section_map = {} # Map index in sections_to_embed back to original section object

    for i, section in enumerate(unique_sections):
        title = clean_text(section.get("section_title", ""))
        content = clean_text(section.get("content", ""))
        section_text_for_embed = f"{title}. {title}. {content[:750]}" # Same text used for alignment
        
        if not section_text_for_embed.strip():
            continue

        # Perform contradiction check. If contradictory, skip early.
        is_contradictory = detect_semantic_contradictions(section, task_requirements, model, batch_size=batch_size)
        if is_contradictory:
            contradiction_count += 1
            continue

        # If not contradictory, add to batch for alignment scoring
        sections_to_embed.append(section_text_for_embed)
        original_section_map[len(sections_to_embed) - 1] = section
    
    if not sections_to_embed:
        print("⚠️  Warning: No sections passed initial filtering/contradiction check. Returning empty.")
        return []

    # Batch encode sections for alignment score calculation
    try:
        section_embeddings = model.encode(sections_to_embed, convert_to_tensor=True, batch_size=batch_size)
    except Exception as e:
        print(f"ERROR: Batch encoding sections failed: {e}. Falling back to individual encoding for remaining sections.")
        section_embeddings = None # Indicate batch encoding failed
    
    for i, section_text in enumerate(sections_to_embed):
        section = original_section_map[i]
        
        # Recalculate alignment score. If batch encoding failed, perform individual encoding.
        if section_embeddings is not None:
            # Use pre-computed embedding
            section_embedding_for_alignment = section_embeddings[i]
            # Call semantic alignment with pre-computed embedding
            alignment_score = calculate_semantic_alignment_with_embedding(
                section, task_requirements, model, section_embedding_for_alignment
            )
        else:
            # Fallback if batch encoding failed - will re-encode individually in calculate_semantic_alignment
            alignment_score = calculate_semantic_alignment(section, task_requirements, model, batch_size=batch_size)
        
        if alignment_score >= 0.05: # Keep sections with reasonable alignment
            section["alignment_score"] = alignment_score
            valid_sections.append(section)
    
    print(f"🚫 Filtered out {contradiction_count} contradictory sections")
    print(f"✅ {len(valid_sections)} sections remain after contradiction filtering and alignment check.")
    
    if not valid_sections:
        print("⚠️  Warning: No sections passed alignment filtering. Attempting to return few sections with lowest possible score.")
        # Fallback: if all sections are filtered, return a few low-scoring ones to prevent empty output
        for section in unique_sections[:min(top_n, len(unique_sections))]:
            section["alignment_score"] = 0.01 # Assign a minimal score
            valid_sections.append(section)
        # Ensure only unique sections are added in fallback, and avoid re-adding if already added above
        valid_sections = list({id(sec): sec for sec in valid_sections}.values()) # Deduplicate by object ID

    # Step 4: Enhance scores with quality indicators
    for section in valid_sections:
        base_score = section.get("alignment_score", 0.0)
        
        content = section.get("content", "")
        title = section.get("section_title", "")
        
        # Content depth (more substantial content gets slight bonus, log-scaled)
        content_words = len(content.split()) if content else 0
        depth_bonus = min(0.05, math.log1p(content_words) / 100) # Max bonus 0.05, scaled

        # Title quality (descriptive titles get bonus based on length)
        title_word_count = len(title.split())
        title_bonus = 0.02 if title_word_count >= 3 else 0.0 # Small bonus for 3+ word titles
        
        # Page position (slight preference for earlier pages, decaying. Max 50 pages influence)
        page_num = section.get("page_number", 1)
        page_bonus = max(0.0, (50 - page_num) / 1000) # Max bonus for page 1 = 0.049

        final_score = base_score + depth_bonus + title_bonus + page_bonus
        section["importance_score"] = final_score
    
    # Step 5: Sort and select top sections
    valid_sections.sort(key=lambda s: s["importance_score"], reverse=True)
    
    # Step 6: Ensure document diversity in final selection
    selected_sections = []
    document_counts = Counter() # Use Counter for convenience
    
    for section in valid_sections:
        if len(selected_sections) >= top_n:
            break
        
        doc = section.get("document", "")
        
        # Limit sections per document to ensure diversity, e.g., max 2 sections per document
        if document_counts[doc] < 2: # Tunable: max sections from one document
            selected_sections.append(section)
            document_counts[doc] += 1
    
    # If not enough sections after diversity filter, fill remaining slots from top ranked
    # without strict document limit
    if len(selected_sections) < top_n:
        for section in valid_sections:
            if len(selected_sections) >= top_n:
                break
            if section not in selected_sections: # Add if not already selected
                selected_sections.append(section)
    
    # Step 7: Assign final ranks and clean up temporary scores
    for i, section in enumerate(selected_sections[:top_n]):
        section["importance_rank"] = i + 1
        section.pop("alignment_score", None)
        section.pop("importance_score", None)
    
    print(f"🎯 Selected {len(selected_sections[:top_n])} top sections")
    
    return selected_sections[:top_n]


def calculate_semantic_alignment_with_embedding(section, task_requirements, model, section_embedding):
    """
    Helper for calculate_semantic_alignment when section_embedding is already provided.
    Avoids re-encoding.
    """
    try:
        with torch.no_grad():
            task_embedding = task_requirements.get('task_embedding')
            
            if task_embedding is None:
                # Fallback to keyword matching if task embedding failed (should ideally not happen here)
                return calculate_keyword_overlap(f"{section.get('section_title', '')} {section.get('content', '')}", task_requirements)
            
            semantic_score = util.cos_sim(task_embedding, section_embedding).item()
            
            positive_terms = task_requirements.get('positive_terms', [])
            phrase_scores = []
            
            if positive_terms:
                positive_term_embeddings = [task_requirements['term_embeddings'][p] for p in positive_terms if p in task_requirements['term_embeddings']]
                
                if positive_term_embeddings:
                    term_sims = util.cos_sim(torch.stack(positive_term_embeddings), section_embedding).flatten().tolist()
                    phrase_scores.extend(term_sims)
                    
                    if phrase_scores:
                        phrase_scores.sort(reverse=True)
                        avg_top_phrases_score = np.mean(phrase_scores[:min(len(phrase_scores), 3)])
                        combined_score = (semantic_score * 0.7) + (avg_top_phrases_score * 0.3)
                    else:
                        combined_score = semantic_score
                else:
                    combined_score = semantic_score
            else:
                combined_score = semantic_score
            
            return combined_score
    except Exception as e:
        print(f"Error calculating semantic alignment with pre-computed embedding: {e}. Falling back to keyword overlap.")
        return calculate_keyword_overlap(f"{section.get('section_title', '')} {section.get('content', '')}", task_requirements)


def extract_focused_snippet(text, query, model, max_words=50, context_sentences=1, batch_size=32):
    """
    Extract a highly focused snippet that directly addresses the query,
    including a tunable number of surrounding sentences for context.
    """
    if not text or len(text.strip()) < 15:
        return ""
    
    text = clean_text(text)
    
    # Split into sentences more robustly
    # Use a regex that keeps the delimiter but splits, then refine
    raw_sentences = re.split(r'([.!?;\n])', text) # Split by common sentence endings and newlines
    sentences = []
    buffer = ""
    for part in raw_sentences:
        buffer += part
        if part.strip() in ['.', '!', '?', ';', '\n']:
            sentence = buffer.strip()
            if sentence and len(sentence.split()) >= 2: # Minimum 2 words
                sentences.append(sentence)
            buffer = ""
    if buffer: # Add any remaining text as a sentence
        sentence = buffer.strip()
        if sentence and len(sentence.split()) >= 2:
            sentences.append(sentence)

    sentences = [s for s in sentences if s] # Remove empty strings

    if not sentences:
        words = text.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
    
    if len(sentences) == 1:
        words = sentences[0].split()
        if len(words) <= max_words:
            return sentences[0]
        else:
            return " ".join(words[:max_words]) + "..."
    
    try:
        with torch.no_grad():
            query_embedding = model.encode(query, convert_to_tensor=True)
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=batch_size)
            similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
        
        best_idx = similarities.argmax().item()
        
        # Combine best sentence with surrounding context sentences
        start_idx = max(0, best_idx - context_sentences)
        end_idx = min(len(sentences), best_idx + context_sentences + 1)
        
        contextual_snippet_sentences = sentences[start_idx:end_idx]
        combined_snippet_text = " ".join(contextual_snippet_sentences)
        
        # Truncate combined snippet to max_words
        words = combined_snippet_text.split()
        if len(words) <= max_words:
            return combined_snippet_text
        else:
            return " ".join(words[:max_words]) + "..."
            
    except Exception as e:
        print(f"Error in focused snippet extraction (semantic): {e}. Fallback to simple truncation.")
        # Fallback: return first sentence or truncated text
        if sentences:
            words = sentences[0].split()
            if len(words) <= max_words:
                return sentences[0]
            else:
                return " ".join(words[:max_words]) + "..."
        else:
            words = text.split()
            return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def extract_top_paragraphs(section_text, query, page_number, document, model, section_title="", top_k=1, batch_size=32):
    """
    Extract focused, relevant snippets from section content, leveraging the improved
    extract_focused_snippet function.
    """
    if not section_text or len(section_text.strip()) < 20:
        return []
    
    # Create focused extraction query
    extraction_query = f"Information about: {query}"
    if section_title:
        extraction_query = f"From {section_title}: {extraction_query}"
    
    # Split into paragraphs more carefully (removes empty ones)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', clean_text(section_text)) if p.strip()]
    
    if not paragraphs:
        # If no clear paragraphs, treat entire content as one block for snippet extraction
        snippet = extract_focused_snippet(section_text, extraction_query, model, max_words=50, batch_size=batch_size)
        if snippet and len(snippet.split()) >= 3:
            return [{
                "document": document,
                "refined_text": snippet,
                "page_number": page_number
            }]
        return []
    
    best_snippet = ""
    
    try:
        with torch.no_grad():
            query_embedding = model.encode(extraction_query, convert_to_tensor=True)
            para_embeddings = model.encode(paragraphs, convert_to_tensor=True, batch_size=batch_size)
            similarities = util.cos_sim(query_embedding, para_embeddings)[0]
        
        # Get best paragraph index
        best_idx = similarities.argmax().item()
        best_para = paragraphs[best_idx]
        
        # Extract focused snippet from the best paragraph, including context sentences
        best_snippet = extract_focused_snippet(best_para, extraction_query, model, max_words=50, context_sentences=1, batch_size=batch_size)
        
    except Exception as e:
        print(f"Error extracting top paragraph (semantic): {e}. Fallback to first paragraph.")
        # Fallback: use first paragraph if embedding fails
        if paragraphs:
            best_snippet = extract_focused_snippet(paragraphs[0], extraction_query, model, max_words=50, context_sentences=1, batch_size=batch_size)
    
    # Return result if we have a good snippet
    if best_snippet and len(best_snippet.split()) >= 3: # Ensure snippet has at least 3 words
        return [{
            "document": document,
            "refined_text": best_snippet,
            "page_number": page_number
        }]
    
    return []