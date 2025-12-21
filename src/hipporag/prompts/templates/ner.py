ner_system = """Your task is to extract named entities from the given paragraph.

CRITICAL: Extract entities in the SAME LANGUAGE as the input text.
- If the text is in Bangla/Bengali, extract Bangla entities (e.g., "অনুপম" not "Anupam")
- If the text is in Hindi, extract Hindi entities
- If the text is in English, extract English entities
- Do NOT translate entities to another language
- Preserve the original script and spelling exactly as written

Extract: Person names, Organizations, Locations, Dates, Events, Products, etc.
Respond with a JSON list of entities.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]