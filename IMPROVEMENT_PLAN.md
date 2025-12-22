# HippoRAG Retrieval Improvement Plan

## Problem Statement
Passages with correct answers are being ranked lower than expected because:
1. Triple extraction misses important relationships (dates, events, actions)
2. PPR over-relies on matched facts, ignoring strong DPR signals

---

## 1. Improve Triple Extraction - Date/Event Relationships

### Current Issue
```python
# Passage contains:
"মিন্টু চলে গেল জুন মাসে, জুনের ২৩ তারিখে"

# Extracted (wrong):
['বৌ', 'said', '...']
['রেডিও টেলিভিশনে', 'reports', '...']

# Missing (should extract):
['মিন্টু', 'left on', 'জুনের ২৩ তারিখে']
['মিন্টু', 'shifted house on', 'জুলাইয়ের পয়লা তারিখে']
```

### Solution: Add Few-Shot Examples

#### File: `src/hipporag/prompts/templates/triple_extraction.py`

Add new examples for:

**A. Date/Time Relationships**
```python
date_event_paragraph = """মিন্টু চলে গেল জুন মাসে, জুনের ২৩ তারিখে। জুলাইয়ের পয়লা তারিখে সে বাড়ি শিফট করল।
রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালের ৭ই মে জন্মগ্রহণ করেন এবং ১৯৪১ সালের ৭ই আগস্ট মৃত্যুবরণ করেন।"""

date_event_output = """{"triples": [
    ["মিন্টু", "left on", "জুনের ২৩ তারিখে"],
    ["মিন্টু", "shifted house on", "জুলাইয়ের পয়লা তারিখে"],
    ["রবীন্দ্রনাথ ঠাকুর", "born on", "১৮৬১ সালের ৭ই মে"],
    ["রবীন্দ্রনাথ ঠাকুর", "died on", "১৯৪১ সালের ৭ই আগস্ট"]
]}"""
```

**B. Action/Event Relationships**
```python
action_event_paragraph = """নুরুল হুদা কলেজে গেলেন সকাল ১০টায়। মিলিটারি তাকে ধরে নিয়ে গেল।
বঙ্গবন্ধু ১৯৭১ সালের ৭ই মার্চ ঐতিহাসিক ভাষণ দিলেন।"""

action_event_output = """{"triples": [
    ["নুরুল হুদা", "went to", "কলেজ"],
    ["নুরুল হুদা", "went at", "সকাল ১০টা"],
    ["মিলিটারি", "captured", "নুরুল হুদা"],
    ["বঙ্গবন্ধু", "gave historic speech on", "৭ই মার্চ ১৯৭১"]
]}"""
```

**C. Departure/Arrival Relationships**
```python
movement_paragraph = """সে ঢাকা থেকে চট্টগ্রাম গেল। ট্রেন ছাড়ল সন্ধ্যা ৬টায়।"""

movement_output = """{"triples": [
    ["সে", "departed from", "ঢাকা"],
    ["সে", "went to", "চট্টগ্রাম"],
    ["ট্রেন", "departed at", "সন্ধ্যা ৬টা"]
]}"""
```

### Implementation Steps
1. Add examples to `triple_extraction.py`
2. Update `prompt_template` list to include new examples
3. Clear LLM cache and re-index documents
4. Test with date-related queries

---

## 2. Increase DPR Weight - Adaptive Hybrid Scoring

### Current Issue
```
DPR ranks correct passage at #2 (score 0.9886)  ✓
PPR pushes it to #7 (score 0.0053)              ✗
```

PPR hurts retrieval when:
- Matched facts don't cover the query intent
- Query asks about relationships not in the knowledge graph

### Solution: Adaptive Hybrid Scoring

#### File: `src/hipporag/HippoRAG.py`

**A. Add Confidence-Based Weighting**
```python
def compute_hybrid_score(self, ppr_scores, dpr_scores, fact_confidence):
    """
    Combine PPR and DPR scores based on fact matching confidence.

    When fact_confidence is HIGH: Trust PPR more (knowledge graph is relevant)
    When fact_confidence is LOW: Trust DPR more (fall back to semantic search)
    """
    # Normalize scores to 0-1
    ppr_norm = self._normalize_scores(ppr_scores)
    dpr_norm = self._normalize_scores(dpr_scores)

    # Adaptive weight based on fact confidence
    # fact_confidence: max score of matched facts (0-1)
    if fact_confidence > 0.8:
        # High confidence: trust PPR (70% PPR, 30% DPR)
        alpha = 0.7
    elif fact_confidence > 0.5:
        # Medium confidence: balanced (50% each)
        alpha = 0.5
    else:
        # Low confidence: trust DPR (30% PPR, 70% DPR)
        alpha = 0.3

    hybrid_scores = alpha * ppr_norm + (1 - alpha) * dpr_norm
    return hybrid_scores
```

**B. Modify Retrieval Pipeline**
```python
def retrieve(self, queries, num_to_retrieve=None):
    # ... existing code ...

    # Get fact matching confidence
    max_fact_score = np.max(query_fact_scores) if len(query_fact_scores) > 0 else 0

    # Get DPR scores
    dpr_sorted_ids, dpr_scores = self.dense_passage_retrieval(query)

    # Get PPR scores
    ppr_scores = self._run_ppr(...)

    # Compute hybrid scores
    final_scores = self.compute_hybrid_score(
        ppr_scores=ppr_scores,
        dpr_scores=dpr_scores,
        fact_confidence=max_fact_score
    )

    # Sort by hybrid scores
    sorted_indices = np.argsort(final_scores)[::-1]
```

**C. Configuration Options**
```python
# In config_utils.py
hybrid_mode: str = field(
    default="adaptive",  # Options: "adaptive", "fixed", "ppr_only", "dpr_only"
    metadata={"help": "Hybrid scoring mode"}
)

hybrid_alpha: float = field(
    default=0.5,
    metadata={"help": "Fixed alpha for hybrid mode (0=DPR only, 1=PPR only)"}
)

fact_confidence_threshold_high: float = field(
    default=0.8,
    metadata={"help": "Above this: trust PPR more"}
)

fact_confidence_threshold_low: float = field(
    default=0.5,
    metadata={"help": "Below this: trust DPR more"}
)
```

### Implementation Steps
1. Add `compute_hybrid_score()` method to HippoRAG class
2. Modify `retrieve()` to use hybrid scoring
3. Add configuration options
4. Test with various query types

---

## Testing Plan

### Test Cases for Triple Extraction
| Query | Expected Triple | Expected Answer |
|-------|-----------------|-----------------|
| মিন্টু কত তারিখে চলে যায়? | ['মিন্টু', 'left on', 'জুনের ২৩'] | জুনের ২৩ তারিখে |
| রবীন্দ্রনাথ কবে জন্মগ্রহণ করেন? | ['রবীন্দ্রনাথ', 'born on', '৭ই মে'] | ১৮৬১ সালের ৭ই মে |

### Test Cases for Hybrid Scoring
| Scenario | Fact Confidence | Expected Behavior |
|----------|-----------------|-------------------|
| Strong fact match | > 0.8 | Trust PPR (70%) |
| Weak fact match | < 0.5 | Trust DPR (70%) |
| No facts matched | 0 | Use DPR only |

---

## Priority

1. **High Priority**: Improve Triple Extraction (immediate impact)
2. **Medium Priority**: Adaptive Hybrid Scoring (architectural improvement)

---

## Estimated Impact

| Improvement | Current Issue | Expected Improvement |
|-------------|---------------|----------------------|
| Triple Extraction | 30% date queries fail | 90%+ date queries succeed |
| Hybrid Scoring | PPR pushes good passages down | Balanced ranking |
