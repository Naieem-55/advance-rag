ner_system = """Your task is to extract named entities from the given paragraph.

CRITICAL: Extract entities in the SAME LANGUAGE as the input text.
- If the text is in Bangla/Bengali, extract Bangla entities (e.g., "অনুপম" not "Anupam")
- If the text is in Hindi, extract Hindi entities
- If the text is in English, extract English entities
- Do NOT translate entities to another language
- Preserve the original script and spelling exactly as written

ENTITY TYPES TO EXTRACT:
1. Person names (ব্যক্তির নাম)
2. Organizations - companies, institutions, coaching centers, universities, banks, hospitals
   Examples: উদ্ভাস, উন্মেষ, ঢাকা বিশ্ববিদ্যালয়, সোনালী ব্যাংক, BUET, KUET
3. Locations - cities, districts, countries, addresses
   Examples: ঢাকা, মিরপুর, চট্টগ্রাম, বাংলাদেশ
4. Dates and Times (তারিখ)
5. Numbers and Quantities (সংখ্যা) - e.g., ১০৮ টি শাখা, ৬৪ জেলা
6. Products, Services, Events
7. Educational terms - বিভাগ, অনুষদ, শাখা, ক্যাম্পাস

IMPORTANT for Bangla:
- Extract compound names: "উদ্ভাস-উন্মেষ" as single entity
- Extract branch/location names: "মিরপুর শাখা", "পল্লবী শাখা"
- Extract institutional divisions: "ঢাকা বিভাগ", "চট্টগ্রাম বিভাগ"

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

# Bangla example for organization extraction
bangla_ner_paragraph = """দেশব্যাপী ৬৪ জেলায় উদ্ভাস-উন্মেষ এর ১০৮ টি শাখা রয়েছে।

ঢাকা বিভাগ
মিরপুর শাখার ফোন নং ০১৭১৩২৩৬৭০৫। ঠিকানা: ন্যাশনাল ব্যাংকের উপরে, মিরপুর-২।
পল্লবী শাখার ফোন নং ০১৭১৩২৩৬৮১৮।"""

bangla_ner_output = """{"named_entities":
    ["৬৪ জেলা", "উদ্ভাস-উন্মেষ", "১০৮ টি শাখা", "ঢাকা বিভাগ", "মিরপুর শাখা", "০১৭১৩২৩৬৭০৫", "ন্যাশনাল ব্যাংক", "মিরপুর-২", "পল্লবী শাখা", "০১৭১৩২৩৬৮১৮"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": bangla_ner_paragraph},
    {"role": "assistant", "content": bangla_ner_output},
    {"role": "user", "content": "${passage}"}
]