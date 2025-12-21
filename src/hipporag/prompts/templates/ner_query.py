ner_system = """You're a very effective entity extraction system.

CRITICAL: Extract entities in the SAME LANGUAGE as the input question.
- If the question is in Bangla/Bengali, extract Bangla entities (e.g., "অনুপম" not "Anupam")
- If the question is in Hindi, extract Hindi entities
- If the question is in English, extract English entities
- Do NOT translate entities to another language
- Preserve the original script and spelling exactly as written
"""

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""
# query_prompt_template = """
# Question: {}

# """
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "Question: ${query}"}
]