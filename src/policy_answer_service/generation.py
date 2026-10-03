"""
Answer generation: structured, cited answers with an honest refusal path.

TODO:
- def generate_answer(question, retrieved_chunks) -> AskResponse
  - must include: answer text, cited chunk_ids, confidence flag
  - must return an explicit "not in the policies" response when retrieval
    finds nothing relevant above threshold
- def score_faithfulness(question, answer, cited_chunks) -> float
  - simple LLM-as-judge prompt; spot-check 10 by hand against the judge
- handle the two outdated/conflicting policy docs by filtering on
  effective_date before generation
"""