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
    noun_phrase_pattern = r'\b(?:[a-z]+(?:\s+[a-z]+){1,3})\b'
    potential_concepts = re.findall(noun_phrase_pattern, task_clean)
    
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
        if len(words) < 2 or all(word in common_words for word in words):
            continue
        score = len(words) * 0.5
        if task_clean.index(concept) < len(task_clean) * 0.3: 
            score += 1.0
        
        scored_concepts.append((concept, score))
    
    scored_concepts.sort(key=lambda x: x[1], reverse=True)
    key_concepts = [concept for concept, _ in scored_concepts[:10]] 
    
    inclusion_terms = []
    exclusion_terms = []
    
    inclusion_patterns = [
        r'(?:including?|with|featuring|contains?|such as|especially|specifically)\s+([^.,;!?]+)',
        r'(?:focus on|looking for|interested in|need|want|require)\s+([^.,;!?]+)',
        r'(?:must have|should have|has to have)\s+([^.,;!?]+)'
    ]
     
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
            terms = [term.strip() for term in re.split(r'[,;]', match) if term.strip()]
            exclusion_terms.extend(terms)
    
    key_concepts = list(set([clean_text(c) for c in key_concepts if c]))
    inclusion_terms = list(set([clean_text(t) for t in inclusion_terms if t]))
    exclusion_terms = list(set([clean_text(t) for t in exclusion_terms if t]))
    
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
    if not section_text or task_requirements.get('task_embedding') is None:
        return 0.0
    
    try:
        with torch.no_grad():
            section_embedding = bi_encoder_model.encode(section_text, convert_to_tensor=True, batch_size=batch_size)
            
            base_similarity = util.cos_sim(task_requirements['task_embedding'], section_embedding).item()

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
                    concept_boost = max(concept_similarities) * 0.3
            
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
    
    for term in exclusion_terms:
        if term.lower() in section_lower:
            return True
    try:
        with torch.no_grad():
            section_embedding = bi_encoder_model.encode(section_text, convert_to_tensor=True, batch_size=batch_size)
            
            for term in exclusion_terms:
                if term in task_requirements['term_embeddings']:
                    term_emb = task_requirements['term_embeddings'][term]
                    similarity = util.cos_sim(term_emb, section_embedding).item()
                    
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
    
    content_words = len(content.split()) if content else 0
    features['content_length'] = min(1.0, content_words / 200)
    
    title_words = len(title.split()) if title else 0
    features['title_quality'] = 1.0 if 2 <= title_words <= 10 else 0.5
    
    features['has_structured_content'] = 1.0 if content and ('\n' in content or '.' in content) else 0.0
    
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
    
    print(f"Analyzing task for relevance ranking...")
    
    # Extract task requirements using improved method
    task_requirements = extract_key_concepts_and_constraints(job_task, bi_encoder_model, batch_size)
    
    print(f"Extracted {len(task_requirements['key_concepts'])} key concepts")
    print(f"Found {len(task_requirements['inclusion_terms'])} inclusion terms") 
    print(f"Found {len(task_requirements['exclusion_terms'])} exclusion terms")
    
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
        print(f"Excluded {excluded_count} sections due to constraint violations")
    
    if not valid_sections:
        print("No valid sections after constraint filtering")
        return []
    
    scored_sections = []
    
    job_task_embedding = task_requirements['task_embedding']
    
    document_names = list(set([clean_text(s.get('document', '')) for s in valid_sections]))
    doc_name_embeddings = {}
    if document_names:
        try:
            with torch.no_grad():
                doc_embeddings = bi_encoder_model.encode(document_names, convert_to_tensor=True, batch_size=batch_size)
                for i, doc_name in enumerate(document_names):
                    doc_name_embeddings[doc_name] = doc_embeddings[i]
        except Exception as e:
            print(f"Warning: Error encoding document names: {e}")


    for section in valid_sections:
        title = clean_text(section.get("section_title", ""))
        content = clean_text(section.get("content", ""))
        section_text = f"{title}. {content[:1000]}"
        document_name = clean_text(section.get("document", ""))
        
        alignment_score = calculate_semantic_alignment(section_text, task_requirements, bi_encoder_model, batch_size)
        quality_features = extract_quality_features(section)
        
        quality_score = (
            quality_features['content_length'] * 0.3 +
            quality_features['title_quality'] * 0.2 +
            quality_features['has_structured_content'] * 0.3 +
            quality_features['position_score'] * 0.2
        )

        # Calculate document name relevance score
        doc_name_relevance = 0.0
        if job_task_embedding is not None and document_name in doc_name_embeddings:
            try:
                doc_name_embedding = doc_name_embeddings[document_name]
                doc_name_relevance = util.cos_sim(job_task_embedding, doc_name_embedding).item()
            except Exception as e:
                print(f"Warning: Error calculating document name similarity: {e}")

        initial_score = (
            alignment_score * 0.7 + quality_score * 0.25 +  doc_name_relevance * 0.05
        )
        
        scored_sections.append((initial_score, section, section_text))
    
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    
    rerank_pool_size = min(len(scored_sections), max(top_n * 3, 15))
    candidates_for_rerank = scored_sections[:rerank_pool_size]
    
    print(f"Re-ranking top {len(candidates_for_rerank)} candidates with cross-encoder")
    final_sections = []
    try:
        ce_pairs = []
        section_map = []
        
        for score, section, section_text in candidates_for_rerank:
            ce_pairs.append([job_task, section_text])
            section_map.append((score, section))
        with torch.no_grad():
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
        for i, (initial_score, section) in enumerate(section_map):
            ce_score = ce_scores[i]
            final_score = initial_score * 0.25 + ce_score * 0.75
            
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
    
    final_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    selected_sections = []
    doc_counts = Counter()
    max_per_doc = max(1, top_n // 3)
    
    for section in final_sections:
        if len(selected_sections) >= top_n:
            break
        
        doc_name = section.get('document', '')
        if doc_counts[doc_name] < max_per_doc:
            selected_sections.append(section)
            doc_counts[doc_name] += 1
    
    if len(selected_sections) < top_n:
        for section in final_sections:
            if len(selected_sections) >= top_n:
                break
            if section not in selected_sections:
                selected_sections.append(section)
    
    for i, section in enumerate(selected_sections[:top_n]):
        section['importance_rank'] = i + 1
        section.pop('relevance_score', None)
    
    print(f"Selected {len(selected_sections[:top_n])} top sections")
    
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
    sentences = []
    text_normalized = re.sub(r'\s+', ' ', text)
    
    sentence_boundaries = list(re.finditer(r'[.!?;]\s+', text_normalized))
    
    if not sentence_boundaries:
        words = text_normalized.split()
        if len(words) <= max_words:
            return text_normalized
        else:
            truncated_words = words[:max_words]
            truncated_text = " ".join(truncated_words)
            natural_breaks = [',', ';', ')', ']', '}']
            for i in range(len(truncated_text) - 1, max(0, len(truncated_text) - 50), -1):
                if truncated_text[i] in natural_breaks:
                    return truncated_text[:i + 1]
            return truncated_text
    
    start = 0
    for boundary in sentence_boundaries:
        end = boundary.end()
        sentence = text_normalized[start:end].strip()
        if sentence and len(sentence.split()) >= 3:  
            sentences.append(sentence)
        start = end
    
    if start < len(text_normalized):
        remaining = text_normalized[start:].strip()
        if remaining and len(remaining.split()) >= 3:
            sentences.append(remaining)
    
    if not sentences:
        words = text_normalized.split()
        if len(words) <= max_words:
            return text_normalized
        else:
            return " ".join(words[:max_words])
    
    if len(sentences) == 1:
        sentence = sentences[0]
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        else:
            truncated_words = words[:max_words]
            truncated_text = " ".join(truncated_words)
            
            natural_breaks = [',', ';', ')', ']', '}', 'and', 'or', 'but', 'that', 'which']
            for i in range(len(truncated_words) - 1, max(0, len(truncated_words) - 10), -1):
                if truncated_words[i].rstrip('.,;!?') in natural_breaks:
                    return " ".join(truncated_words[:i + 1])
            
            return truncated_text
    
    try:
        with torch.no_grad():
            ce_pairs = [[query, sentence] for sentence in sentences]
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
            
            best_idx = np.argmax(ce_scores)
            best_sentence = sentences[best_idx]
            start_idx = max(0, best_idx - context_sentences)
            end_idx = min(len(sentences), best_idx + context_sentences + 1)
            
            context_sentences_list = sentences[start_idx:end_idx]
            combined_text = " ".join(context_sentences_list)
            
            words = combined_text.split()
            if len(words) <= max_words:
                return combined_text
            else:
                best_sentence_words = best_sentence.split()
                if len(best_sentence_words) <= max_words:
                    remaining_words = max_words - len(best_sentence_words)
                    
                    before_sentences = sentences[start_idx:best_idx]
                    after_sentences = sentences[best_idx + 1:end_idx]
                    
                    context_parts = []
                    context_words_used = 0
                    
                    for sent in after_sentences:
                        sent_words = sent.split()
                        if context_words_used + len(sent_words) <= remaining_words:
                            context_parts.append(sent)
                            context_words_used += len(sent_words)
                        else:
                            break
                    
                    for sent in reversed(before_sentences):
                        sent_words = sent.split()
                        if context_words_used + len(sent_words) <= remaining_words:
                            context_parts.insert(0, sent)
                            context_words_used += len(sent_words)
                        else:
                            break
                    
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
                    truncated_words = best_sentence_words[:max_words]
                    return " ".join(truncated_words)
    
    except Exception as e:
        print(f"Warning: Cross-encoder failed in snippet extraction: {e}")
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
    
    paragraph_splits = re.split(r'\n\s*\n+|\n\s*[-•]\s*|\n\s*\d+\.\s*|\n\s*[a-zA-Z]\.\s*', section_text)
    paragraphs = [p.strip() for p in paragraph_splits if p.strip() and len(p.split()) >= 5]
    
    if not paragraphs:
        snippet = extract_focused_snippet(section_text, extraction_query, bi_encoder_model, 
                                          cross_encoder_model, max_words=70, batch_size=batch_size)
        if snippet and len(snippet.split()) >= 5:
            return [{
                "document": document,
                "refined_text": snippet,
                "page_number": page_number
            }]
        return []
    
    try:
        with torch.no_grad():
            processed_paragraphs = [p[:800] for p in paragraphs]
            
            ce_pairs = [[extraction_query, p] for p in processed_paragraphs]
            ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=batch_size)
            
            best_idx = np.argmax(ce_scores)
            best_paragraph = paragraphs[best_idx]
            
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
