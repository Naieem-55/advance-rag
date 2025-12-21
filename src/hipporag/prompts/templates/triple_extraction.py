from .ner import one_shot_ner_paragraph, one_shot_ner_output, bangla_ner_paragraph, bangla_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
- For tabular data (like branch lists, contact info), extract relationships like "has branch at", "phone number is", "address is".

CRITICAL - LANGUAGE PRESERVATION:
- Keep ALL entities in the SAME LANGUAGE as the source text
- If the passage is in Bangla, use Bangla entities (e.g., "অনুপম" not "Anupam")
- If the passage is in Hindi, use Hindi entities
- Do NOT translate entities or predicates to English
- Preserve the original script and spelling exactly
- Predicates can be in English for consistency, but entities MUST match the source language

"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

# Bangla example for triple extraction from tabular/structured data
bangla_re_input = ner_conditioned_re_frame.format(passage=bangla_ner_paragraph, named_entity_json=bangla_ner_output)

bangla_re_output = """{"triples": [
            ["উদ্ভাস-উন্মেষ", "has total branches", "১০৮ টি শাখা"],
            ["উদ্ভাস-উন্মেষ", "operates in", "৬৪ জেলা"],
            ["মিরপুর শাখা", "belongs to", "ঢাকা বিভাগ"],
            ["মিরপুর শাখা", "phone number is", "০১৭১৩২৩৬৭০৫"],
            ["মিরপুর শাখা", "located at", "ন্যাশনাল ব্যাংক"],
            ["মিরপুর শাখা", "address is", "মিরপুর-২"],
            ["পল্লবী শাখা", "belongs to", "ঢাকা বিভাগ"],
            ["পল্লবী শাখা", "phone number is", "০১৭১৩২৩৬৮১৮"]
    ]
}
"""

# Quote/dialogue attribution example
quote_ner_paragraph = """কাবুলি বললেন, "দুনিয়ার সব পরীক্ষা পাস করার চেয়ে বড় পরীক্ষা খাইবারপাস পাস করা।"
আমি বললুম, "আপনার মুখে ফুলচন্দন পড়ুক।"
"""

quote_ner_output = """{"named_entities": ["কাবুলি", "খাইবারপাস"]}"""

quote_re_input = ner_conditioned_re_frame.format(passage=quote_ner_paragraph, named_entity_json=quote_ner_output)

quote_re_output = """{"triples": [
            ["কাবুলি", "said", "দুনিয়ার সব পরীক্ষা পাস করার চেয়ে বড় পরীক্ষা খাইবারপাস পাস করা"],
            ["খাইবারপাস পাস করা", "described as", "বড় পরীক্ষা"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": bangla_re_input},
    {"role": "assistant", "content": bangla_re_output},
    {"role": "user", "content": quote_re_input},
    {"role": "assistant", "content": quote_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]