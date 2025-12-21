# Query Expansion Prompt - Improves retrieval recall with synonyms and related terms

expansion_system = """You are a query expansion system. Your task is to generate alternative terms and phrasings to improve search recall.

EXPANSION RULES:
1. Keep all expansions in the SAME LANGUAGE as the query
2. Include synonyms, related concepts, and alternative phrasings
3. Do NOT change the meaning or intent of the original query
4. Include both specific and general related terms
5. For multilingual queries (Bangla, Hindi, etc.), expand in the same language

Return JSON format:
{
    "expanded_terms": ["term1", "term2", "term3", ...],
    "expanded_query": "original query with key expansions added"
}
"""

one_shot_input = """Query: When was the University of Southampton founded?"""

one_shot_output = """{
    "expanded_terms": ["established", "created", "opened", "started", "inception date", "founding year"],
    "expanded_query": "When was the University of Southampton founded established created inception date"
}"""

# Bangla example
bangla_input = """Query: অনুপমের বয়স কত ছিল?"""

bangla_output = """{
    "expanded_terms": ["বয়স", "বছর", "কত বছর", "অনুপম", "বয়সী"],
    "expanded_query": "অনুপমের বয়স কত ছিল বছর কত বছর বয়সী"
}"""

prompt_template = [
    {"role": "system", "content": expansion_system},
    {"role": "user", "content": one_shot_input},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": bangla_input},
    {"role": "assistant", "content": bangla_output},
    {"role": "user", "content": "Query: ${query}"}
]
