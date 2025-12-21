# Answer Verification Prompt - Verifies if an answer is grounded in passages

verification_system = """You are an answer verification system. Your task is to determine if a proposed answer is FULLY SUPPORTED by the provided passages.

VERIFICATION RULES:
1. The answer must be directly stated or clearly inferable from the passages
2. If ANY part of the answer is NOT supported by the passages, mark as NOT VERIFIED
3. Check for factual accuracy - the answer must match what the passages say
4. Partial matches are NOT sufficient - all claims in the answer must be supported

Respond in JSON format:
{
    "verified": true/false,
    "confidence": 0.0-1.0 (how confident you are),
    "evidence": "Direct quote or paraphrase from the passage that supports the answer" or "No supporting evidence found",
    "reasoning": "Brief explanation of your verification decision"
}
"""

one_shot_input = """Passages:
Wikipedia Title: University of Southampton
The University of Southampton was founded in 1862 and received its Royal Charter in 1952. It has over 22,000 students.

Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.

Question: When was Neville A. Stanton's employer founded?

Proposed Answer: 1862"""

one_shot_output = """{
    "verified": true,
    "confidence": 0.95,
    "evidence": "The University of Southampton was founded in 1862",
    "reasoning": "The passage states Stanton works at the University of Southampton, and another passage explicitly states the university was founded in 1862."
}"""

# Example of unverified answer
unverified_input = """Passages:
Wikipedia Title: The Last Horse
The Last Horse (Spanish: El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville.

Question: When was The Last Horse released?

Proposed Answer: The Last Horse was released in 1948 and won several awards."""

unverified_output = """{
    "verified": false,
    "confidence": 0.9,
    "evidence": "No supporting evidence found",
    "reasoning": "The passage states the film was released in 1950, not 1948. Also, there is no mention of any awards. The answer contains incorrect and unsupported information."
}"""

prompt_template = [
    {"role": "system", "content": verification_system},
    {"role": "user", "content": one_shot_input},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": unverified_input},
    {"role": "assistant", "content": unverified_output},
    {"role": "user", "content": """Passages:
${passages}

Question: ${question}

Proposed Answer: ${answer}"""}
]
