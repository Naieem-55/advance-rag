ner_system = """You're a very effective entity extraction system for answering questions.

CRITICAL: Extract entities in the SAME LANGUAGE as the input question.
- If the question is in Bangla/Bengali, extract Bangla entities (e.g., "অনুপম" not "Anupam")
- If the question is in Hindi, extract Hindi entities
- If the question is in English, extract English entities
- Do NOT translate entities to another language
- Preserve the original script and spelling exactly as written

ENTITY TYPES TO EXTRACT FROM QUESTIONS:
1. Person names (ব্যক্তির নাম)
2. Organization names - institutions, companies, coaching centers, universities
   Examples: উদ্ভাস, উন্মেষ, উদ্ভাস-উন্মেষ, ঢাকা বিশ্ববিদ্যালয়, BUET, KUET
3. Location names - cities, districts, countries
   Examples: ঢাকা, মিরপুর, চট্টগ্রাম, বাংলাদেশ
4. Specific terms being asked about - শাখা, বিভাগ, অনুষদ
5. Numbers if specifically mentioned

IMPORTANT:
- Extract compound organization names as single entity: "উদ্ভাস-উন্মেষ" (not separate)
- Extract the KEY entities needed to answer the question
- For questions about "X এর Y", extract "X" as an entity
"""

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

# Bangla query example
bangla_query_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: উদ্ভাস-উন্মেষ এর কয়টি শাখা আছে? সেগুলার তথ্য দেউ?

"""
bangla_query_output = """
{"named_entities": ["উদ্ভাস-উন্মেষ", "শাখা"]}
"""

# Another Bangla example
bangla_query_input_2 = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: অপরিচিতা গল্পে অনুপমের বয়স কত ছিল?

"""
bangla_query_output_2 = """
{"named_entities": ["অপরিচিতা", "অনুপম"]}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": bangla_query_input},
    {"role": "assistant", "content": bangla_query_output},
    {"role": "user", "content": bangla_query_input_2},
    {"role": "assistant", "content": bangla_query_output_2},
    {"role": "user", "content": "Question: ${query}"}
]