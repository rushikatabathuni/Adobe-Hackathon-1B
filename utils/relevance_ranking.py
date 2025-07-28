from collections import Counter
import torch
from sentence_transformers import util
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any


def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def extract_key_concepts_and_constraints(job_task: str, model, batch_size: int = 32) -> Dict[str, Any]:
    """
    Extract key concepts and constraints from task using improved NLP techniques.
    
    This approach focuses on:
    1. Extracting noun phrases as key concepts
    2. Identifying constraint patterns (inclusion/exclusion)
    3. Using semantic similarity for better understanding
    """
    task_clean = clean_text(job_task).lower()
    
    # Extract meaningful noun phrases (2-4 words) as potential key concepts
    # This captures compound concepts like "luxury hotels", "outdoor activities", etc.
    noun_phrase_pattern = r'\b(?:[a-z]+(?:\s+[a-z]+){1,3})\b'
    potential_concepts = re.findall(noun_phrase_pattern, task_clean)
    
    # Filter out very common words and short phrases
    common_words = {
        'and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'must', 'shall', 'to', 'of', 'in', 'on', 'at', 'by',
        'for', 'with', 'as', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Filter and score concepts
    scored_concepts = []
    for concept in potential_concepts:
        words = concept.split()
        # Skip if contains only common words or is too short
        if len(words) < 2 or all(word in common_words for word in words):
            continue
        
        # Score based on length and position in task
        score = len(words) * 0.5  # Longer phrases are more specific
        if task_clean.index(concept) < len(task_clean) * 0.3:  # Early in task = more important
            score += 1.0
        
        scored_concepts.append((concept, score))
    
    # Sort by score and take top concepts
    scored_concepts.sort(key=lambda x: x[1], reverse=True)
    key_concepts = [concept for concept, _ in scored_concepts[:10]]  # Top 10 concepts
    
    # Extract inclusion/exclusion constraints using improved patterns
    inclusion_terms = []
    exclusion_terms = []
    
    # Inclusion patterns
    inclusion_patterns = [
        r'(?:including?|with|featuring|contains?|such as|especially|specifically)\s+([^.,;!?]+)',
        r'(?:focus on|looking for|interested in|need|want|require)\s+([^.,;!?]+)',
        r'(?:must have|should have|has to have)\s+([^.,;!?]+)'
    ]
    
    # Exclusion patterns  
    exclusion_patterns = [
        r'(?:excluding?|without|not including|avoid|no|not)\s+([^.,;!?]+)',
        r'(?:except|but not|other than)\s+([^.,;!?]+)',
        r'(?:don\'t want|do not want|cannot have)\s+([^.,;!?]+)'
    ]
    
    for pattern in inclusion_patterns:
        matches = re.findall(pattern, task_clean)
        for match in matches:
            # Clean and split the match
            terms = [term.strip() for term in re.split(r'[,;]', match) if term.strip()]
            inclusion_terms.extend(terms)
    
    for pattern in exclusion_patterns:
        matches = re.findall(pattern, task_clean)
        for match in matches:
            # Clean and split the match
            terms = [term.strip() for term in re.split(r'[,;]', match) if term.strip()]
            exclusion_terms.extend(terms)
    
    # Remove duplicates and clean
    key_concepts = list(set([clean_text(c) for c in key_concepts if c]))
    inclusion_terms = list(set([clean_text(t) for t in inclusion_terms if t]))
    exclusion_terms = list(set([clean_text(t) for t in exclusion_terms if t]))
    
    # Generate embeddings for all terms
    all_terms = key_concepts + inclusion_terms + exclusion_terms
    term_embeddings = {}
    task_embedding = None
    
    try:
        with torch.no_grad():
            if all_terms:
                embeddings = model.encode(all_terms, convert_to_tensor=True, batch_size=batch_size)
                for i, term in enumerate(all_terms):
                    term_embeddings[term] = embeddings[i]
            
            task_embedding = model.encode(job_task, convert_to_tensor=True)
    
    except Exception as e:
        print(f"Warning: Error generating embeddings: {e}")
        task_embedding = None
        term_embeddings = {}
    
    return {
        'task_embedding': task_embedding,
        'key_concepts': key_concepts,
        'inclusion_terms': inclusion_terms,
        'exclusion_terms': exclusion_terms,
        'term_embeddings': term_embeddings,
        'full_task': job_task
    }


def calculate_semantic_alignment(section_text: str, task_requirements: Dict, bi_encoder_model, batch_size: int = 32) -> float:
    """
    Calculate how well a section aligns with the task requirements using semantic similarity.
    """
    if not section_text or not task_requirements.get('task_embedding') is not None:
        return 0.0
    
    try:
        with torch.no_grad():
            section_embedding = bi_encoder_model.encode(section_text, convert_to_tensor=True, batch_size=batch_size)
            
            # Base similarity to the full task
            base_similarity = util.cos_sim(task_requirements['task_embedding'], section_embedding).item()
            
            # Boost for key concepts
            concept_boost = 0.0
            key_concepts = task_requirements.get('key_concepts', [])
            if key_concepts:
                concept_similarities = []
                for concept in key_concepts:
                    if concept in task_requirements['term_embeddings']:
                        concept_emb = task_requirements['term_embeddings'][concept]
                        sim = util.cos_sim(concept_emb, section_embedding).item()
                        concept_similarities.append(sim)
                
                if concept_similarities:
                    # Use max similarity to any key concept
                    concept_boost = max(concept_similarities) * 0.3
            
            # Boost for inclusion terms
            inclusion_boost = 0.0
            inclusion_terms = task_requirements.get('inclusion_terms', [])
            if inclusion_terms:
                inclusion_similarities = []
                for term in inclusion_terms:
                    if term in task_requirements['term_embeddings']:
                        term_emb = task_requirements['term_embeddings'][term]
                        sim = util.cos_sim(term_emb, section_embedding).item()
                        inclusion_similarities.append(sim)
                
                if inclusion_similarities:
                    inclusion_boost = max(inclusion_similarities) * 0.2
            
            return min(1.0, base_similarity + concept_boost + inclusion_boost)
    
    except Exception as e:
        print(f"Warning: Error calculating semantic alignment: {e}")
        return 0.0


def detect_exclusion_violations(section_text: str, task_requirements: Dict, bi_encoder_model, batch_size: int = 32) -> bool:
    """
    Detect if a section violates exclusion constraints.
    """
    exclusion_terms = task_requirements.get('exclusion_terms', [])
    if not exclusion_terms:
        return False
    
    section_lower = section_text.lower()
    
    # First check for direct string matches (fast)
    for term in exclusion_terms:
        if term.lower() in section_lower:
            return True
    
    # Then check semantic violations
    try:
        with torch.no_grad():
            section_embedding = bi_encoder_model.encode(section_text, convert_to_tensor=True, batch_size=batch_size)
            
            for term in exclusion_terms:
                if term in task_requirements['term_embeddings']:
                    term_emb = task_requirements['term_embeddings'][term]
                    similarity = util.cos_sim(term_emb, section_embedding).item()
                    
                    # If section is highly similar to an exclusion term, it violates constraints
                    if similarity > 0.7:
                        return True
    
    except Exception as e:
        print(f"Warning: Error in exclusion detection: {e}")
    
    return False


def extract_quality_features(section: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract quality features from a section for scoring.
    """
    title = clean_text(section.get("section_title", ""))
    content = clean_text(section.get("content", ""))
    
    features = {}
    
    # Content length features
    content_words = len(content.split()) if content else 0
    features['content_length'] = min(1.0, content_words / 200)  # Normalize to 200 words
    
    # Title quality features
    title_words = len(title.split()) if title else 0
    features['title_quality'] = 1.0 if 2 <= title_words <= 10 else 0.5
    
    # Structural features
    features['has_structured_content'] = 1.0 if content and ('\n' in content or '.' in content) else 0.0
    
    # Position features (earlier pages might be more important)
    page_num = section.get("page_number", 1)
    features['position_score'] = max(0.0, (100 - page_num) / 100)
    
    return features


def rank_sections(sections: List[Dict], persona_role: str, job_task: str, 
                 bi_encoder_model, cross_encoder_model, top_n: int = 5, batch_size: int = 32) -> List[Dict]:
    """
    Improved ranking system with better task understanding and quality assessment.
    """
    if not sections:
        return []
    
    print(f"🔍 Analyzing task for relevance ranking...")
    
    # Extract task requirements using improved method
    task_requirements = extract_key_concepts_and_constraints(job_task, bi_encoder_model, batch_size)
    
    print(f"📋 Extracted {len(task_requirements['key_concepts'])} key concepts")
    print(f"✅ Found {len(task_requirements['inclusion_terms'])} inclusion terms") 
    print(f"❌ Found {len(task_requirements['exclusion_terms'])} exclusion terms")
    
    # Filter out sections that violate exclusion constraints
    valid_sections = []
    excluded_count = 0
    
    for section in sections:
        title = clean_text(section.get("section_title", ""))
        content = clean_text(section.get("content", ""))
        section_text = f"{title}. {content[:800]}"  # Limit content for processing
        
        if detect_exclusion_violations(section_text, task_requirements, bi_encoder_model, batch_size):
            excluded_count += 1
            continue
        
        valid_sections.append(section)
    
    if excluded_count > 0:
        print(f"🚫 Excluded {excluded_count} sections due to constraint violations")
    
    if not valid_sections:
        print("⚠️ No valid sections after constraint filtering")
        return []
    
    # Calculate comprehensive scores for valid sections
    scored_sections = []
    
    for section in valid_sections:
        title = clean_text(section.get("section_title", ""))
        content = clean_text(section.get("content", ""))
        section_text = f"{title}. {content[:1000]}"
        
        # Calculate semantic alignment score
        alignment_score = calculate_semantic_alignment(section_text, task_requirements, bi_encoder_model, batch_size)
        
        # Extract quality features
        quality_features = extract_quality_features(section)
        
        # Calculate quality score
        quality_score = (
            quality_features['content_length'] * 0.3 +
            quality_features['title_quality'] * 0.2 +
            quality_features['has_structured_content'] * 0.3 +
            quality_features['position_score'] * 0.2
        )
        
        # Combined initial score
        initial_score = alignment_score * 0.7 + quality_score * 0.3
        
        scored_sections.append((initial_score, section, section_text))
    
    # Sort by initial score and select top candidates for cross-encoder re-ranking
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    # Select more candidates than needed for cross-encoder re-ranking
    rerank_pool_size = min(len(scored_sections), max(top_n * 3, 15))
    candidates_for_rerank = scored_sections[:rerank_pool_size]
    
    print(f"🔄 Re-ranking top {len(candidates_for_rerank)} candidates with cross-encoder")
    
    # Cross-encoder re-ranking
    final_sections = []
    
    try:
        # Prepare cross-encoder inputs
        ce_pairs = []
        section_map = []
        
        for score, section, section_text in candidates_for_rerank:
            ce_pairs.append([job_task, section_text])
            section_map.append((score, section))
        
        # Get cross-encoder scores
        with torch.no_grad():
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
        
        # Combine scores (weighted combination of bi-encoder and cross-encoder)
        for i, (initial_score, section) in enumerate(section_map):
            ce_score = ce_scores[i]
            
            # Weighted combination: cross-encoder gets more weight for final ranking
            final_score = initial_score * 0.3 + ce_score * 0.7
            
            section_copy = section.copy()
            section_copy['relevance_score'] = final_score
            final_sections.append(section_copy)
    
    except Exception as e:
        print(f"⚠️ Cross-encoder failed: {e}. Using bi-encoder scores only.")
        # Fallback to bi-encoder scores
        for score, section, _ in candidates_for_rerank:
            section_copy = section.copy()
            section_copy['relevance_score'] = score
            final_sections.append(section_copy)
    
    # Final sorting and selection
    final_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Ensure document diversity in final selection
    selected_sections = []
    doc_counts = Counter()
    max_per_doc = max(1, top_n // 3)  # Allow at most 1/3 of selections from same doc
    
    for section in final_sections:
        if len(selected_sections) >= top_n:
            break
        
        doc_name = section.get('document', '')
        if doc_counts[doc_name] < max_per_doc:
            selected_sections.append(section)
            doc_counts[doc_name] += 1
    
    # Fill remaining slots if needed
    if len(selected_sections) < top_n:
        for section in final_sections:
            if len(selected_sections) >= top_n:
                break
            if section not in selected_sections:
                selected_sections.append(section)
    
    # Assign final ranks and clean up
    for i, section in enumerate(selected_sections[:top_n]):
        section['importance_rank'] = i + 1
        section.pop('relevance_score', None)
    
    print(f"✅ Selected {len(selected_sections[:top_n])} top sections")
    
    return selected_sections[:top_n]


def extract_focused_snippet(text: str, query: str, bi_encoder_model, cross_encoder_model, 
                          max_words: int = 60, context_sentences: int = 1, batch_size: int = 32) -> str:
    """
    Improved snippet extraction with better sentence boundary detection and context preservation.
    Fixed to avoid truncation artifacts and ensure clean text output.
    """
    if not text or len(text.strip()) < 15:
        return ""
    
    text = clean_text(text)
    
    # Improved sentence splitting with better handling of sentence boundaries
    # Split on sentence boundaries but be more careful about reconstruction
    sentences = []
    
    # Use a more robust sentence splitting approach
    # First normalize whitespace and handle common abbreviations
    text_normalized = re.sub(r'\s+', ' ', text)
    
    # Split on sentence endings, keeping track of positions
    sentence_boundaries = list(re.finditer(r'[.!?;]\s+', text_normalized))
    
    if not sentence_boundaries:
        # No clear sentence boundaries found, return truncated text
        words = text_normalized.split()
        if len(words) <= max_words:
            return text_normalized
        else:
            # Find a good breaking point near the word limit
            truncated_words = words[:max_words]
            truncated_text = " ".join(truncated_words)
            
            # Try to end at a natural break (comma, semicolon, etc.)
            natural_breaks = [',', ';', ')', ']', '}']
            for i in range(len(truncated_text) - 1, max(0, len(truncated_text) - 50), -1):
                if truncated_text[i] in natural_breaks:
                    return truncated_text[:i + 1]
            
            return truncated_text
    
    # Extract sentences based on boundaries
    start = 0
    for boundary in sentence_boundaries:
        end = boundary.end()
        sentence = text_normalized[start:end].strip()
        if sentence and len(sentence.split()) >= 3:  # At least 3 words
            sentences.append(sentence)
        start = end
    
    # Add any remaining text as the last sentence
    if start < len(text_normalized):
        remaining = text_normalized[start:].strip()
        if remaining and len(remaining.split()) >= 3:
            sentences.append(remaining)
    
    if not sentences:
        # Fallback to word-based truncation
        words = text_normalized.split()
        if len(words) <= max_words:
            return text_normalized
        else:
            return " ".join(words[:max_words])
    
    if len(sentences) == 1:
        # Only one sentence, handle length constraint
        sentence = sentences[0]
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        else:
            # Truncate at word boundary, try to find natural break
            truncated_words = words[:max_words]
            truncated_text = " ".join(truncated_words)
            
            # Look for natural breaks near the end
            natural_breaks = [',', ';', ')', ']', '}', 'and', 'or', 'but', 'that', 'which']
            for i in range(len(truncated_words) - 1, max(0, len(truncated_words) - 10), -1):
                if truncated_words[i].rstrip('.,;!?') in natural_breaks:
                    return " ".join(truncated_words[:i + 1])
            
            return truncated_text
    
    # Use cross-encoder to find the most relevant sentence
    try:
        with torch.no_grad():
            # Create query-sentence pairs for cross-encoder
            ce_pairs = [[query, sentence] for sentence in sentences]
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
            
            # Find the best sentence
            best_idx = np.argmax(ce_scores)
            best_sentence = sentences[best_idx]
            
            # Add context sentences around the best one
            start_idx = max(0, best_idx - context_sentences)
            end_idx = min(len(sentences), best_idx + context_sentences + 1)
            
            context_sentences_list = sentences[start_idx:end_idx]
            combined_text = " ".join(context_sentences_list)
            
            # Handle length constraint for combined text
            words = combined_text.split()
            if len(words) <= max_words:
                return combined_text
            else:
                # Try to keep the most relevant sentence intact if possible
                best_sentence_words = best_sentence.split()
                if len(best_sentence_words) <= max_words:
                    # Keep the best sentence and add context up to word limit
                    remaining_words = max_words - len(best_sentence_words)
                    
                    # Add context before
                    before_sentences = sentences[start_idx:best_idx]
                    after_sentences = sentences[best_idx + 1:end_idx]
                    
                    context_parts = []
                    context_words_used = 0
                    
                    # Add sentences after the best one first (usually more relevant)
                    for sent in after_sentences:
                        sent_words = sent.split()
                        if context_words_used + len(sent_words) <= remaining_words:
                            context_parts.append(sent)
                            context_words_used += len(sent_words)
                        else:
                            break
                    
                    # Add sentences before if there's still room
                    for sent in reversed(before_sentences):
                        sent_words = sent.split()
                        if context_words_used + len(sent_words) <= remaining_words:
                            context_parts.insert(0, sent)
                            context_words_used += len(sent_words)
                        else:
                            break
                    
                    # Combine: before_context + best_sentence + after_context
                    before_context = " ".join([s for s in context_parts if sentences.index(s) < best_idx])
                    after_context = " ".join([s for s in context_parts if sentences.index(s) > best_idx])
                    
                    result_parts = []
                    if before_context:
                        result_parts.append(before_context)
                    result_parts.append(best_sentence)
                    if after_context:
                        result_parts.append(after_context)
                    
                    return " ".join(result_parts)
                else:
                    # Best sentence is too long, truncate it
                    truncated_words = best_sentence_words[:max_words]
                    return " ".join(truncated_words)
    
    except Exception as e:
        print(f"Warning: Cross-encoder failed in snippet extraction: {e}")
        # Fallback to first sentence with length handling
        first_sentence = sentences[0]
        words = first_sentence.split()
        if len(words) <= max_words:
            return first_sentence
        else:
            return " ".join(words[:max_words])


def extract_top_paragraphs(section_text: str, query: str, page_number: int, document: str,
                          bi_encoder_model, cross_encoder_model, section_title: str = "", 
                          top_k: int = 1, batch_size: int = 32) -> List[Dict]:
    """
    Improved paragraph extraction with better content segmentation and cleaner output.
    """
    if not section_text or len(section_text.strip()) < 30:
        return []
    
    # Create a more specific query for extraction
    extraction_query = f"Relevant information for: {query}"
    if section_title:
        extraction_query = f"From section '{section_title}': {extraction_query}"
    
    # Better paragraph segmentation
    # Split on multiple newlines, bullet points, and numbered lists
    paragraph_splits = re.split(r'\n\s*\n+|\n\s*[-•]\s*|\n\s*\d+\.\s*|\n\s*[a-zA-Z]\.\s*', section_text)
    paragraphs = [p.strip() for p in paragraph_splits if p.strip() and len(p.split()) >= 5]
    
    if not paragraphs:
        # Fallback: treat entire content as one paragraph
        snippet = extract_focused_snippet(section_text, extraction_query, bi_encoder_model, 
                                        cross_encoder_model, max_words=70, batch_size=batch_size)
        if snippet and len(snippet.split()) >= 5:
            return [{
                "document": document,
                "refined_text": snippet,
                "page_number": page_number
            }]
        return []
    
    # Use cross-encoder to find the best paragraph
    try:
        with torch.no_grad():
            # Limit paragraph length for processing
            processed_paragraphs = [p[:800] for p in paragraphs]
            
            ce_pairs = [[extraction_query, p] for p in processed_paragraphs]
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
            
            best_idx = np.argmax(ce_scores)
            best_paragraph = paragraphs[best_idx]
            
            # Extract focused snippet from the best paragraph
            snippet = extract_focused_snippet(best_paragraph, extraction_query, bi_encoder_model,
                                           cross_encoder_model, max_words=75, batch_size=batch_size)
            
            if snippet and len(snippet.split()) >= 5:
                return [{
                    "document": document,
                    "refined_text": snippet,
                    "page_number": page_number
                }]
    
    except Exception as e:
        print(f"Warning: Error in paragraph extraction: {e}")
        # Fallback to first paragraph
        if paragraphs:
            snippet = extract_focused_snippet(paragraphs[0], extraction_query, bi_encoder_model,
                                           cross_encoder_model, max_words=75, batch_size=batch_size)
            if snippet and len(snippet.split()) >= 5:
                return [{
                    "document": document,
                    "refined_text": snippet,
                    "page_number": page_number
                }]
    
    return []