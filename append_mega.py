import os

gemini_code = '''
    # ---------------- MEGA-PROMPTS (Subject-Level) ----------------

    def generate_subject_summary(self, subject_corpus: str, syllabus: str) -> str:
        prompt = f"""
### ROLE
You are a university lecturer creating a massive master summary for an entire semester's subject.

### TASK
Combine the massive lecture transcript corpus below into ONE extremely comprehensive master study guide.
If a topic from the SYLLABUS is missing from the CORPUS, use Knowledge Expansion / Inference to generate the supplemental material for it.

### OUTPUT FORMAT (MUST FOLLOW EXACTLY)
Overview:
<Broad 3-5 paragraph introduction to the entire subject>

Module-by-Module Breakdown:
- <Module/Topic Name>:
  <Detailed Explanation + Concept + Examples + Formulas (if any)>

Master Glossary:
- <term>: <definition>

SYLLABUS:
{syllabus}

CORPUS:
{subject_corpus[:500000]}
""".strip()
        retry_prompt = prompt + "\\n\\nRETRY INSTRUCTION: Output was invalid. You must output the full structure."
        out = self._generate(prompt, retry_prompt=retry_prompt)
        return "Insufficient content" if looks_bad(out) else out

    def generate_subject_flashcards(self, subject_corpus: str, syllabus: str, n: int = 50) -> List[Dict[str, str]]:
        prompt = f"""
### TASK
Generate exactly {n} flashcards spanning the ENTIRE subject corpus.
If a syllabus topic is minimally covered, use Knowledge Inference to add standard academic questions to ensure all syllabus topics are tested.

### OUTPUT (MUST BE VALID JSON ARRAY)
[ {{"topic":"...","question":"...","answer":"...","memory_anchor":"...","difficulty":2}}, ... ]

SYLLABUS:
{syllabus}

CORPUS:
{subject_corpus[:500000]}
""".strip()
        retry_prompt = prompt + f"\\n\\nRETRY: Output exactly {n} JSON objects."
        out = self._generate(prompt, retry_prompt=retry_prompt)
        if out.startswith("```json"): out = out[7:]
        if out.endswith("```"): out = out[:-3]
        
        parsed = _safe_json_loads(out.strip())
        if not isinstance(parsed, list): return []
        
        cleaned = []
        for item in parsed:
            if not isinstance(item, dict): continue
            q = str(item.get("question", item.get("q", ""))).strip()
            a = str(item.get("answer", item.get("a", ""))).strip()
            topic = str(item.get("topic", "")).strip()
            ma = str(item.get("memory_anchor", "")).strip()
            if not (q and a and topic and ma): continue
            cleaned.append({{"topic": topic, "question": q, "answer": a, "memory_anchor": ma, "difficulty": item.get("difficulty", 2)}})
        return cleaned

    def generate_subject_questions_with_answers(self, subject_corpus: str, syllabus: str) -> Dict[str, List[Dict[str, Any]]]:
        prompt = f"""
### TASK
Create a massive Master Exam Bank WITH answers for the entire semester.
MATCH THE EXACT FORMAT AND DIFFICULTY OF A "VIT" ENGINEERING EXAM.
Use Knowledge Inference heavily if syllabus topics are under-represented in the corpus.

### OUTPUT (MUST BE VALID JSON OBJECT)
Return ONLY a valid JSON object with EXACTLY keys "1-2","5","10","15".
Each value MUST be a JSON array of objects: {{"q":"...","a":"...","marks":<int>,"marking_points":["...","..."]}}.

Counts REQUIRED per section:
- "1-2" (Part A): 30 questions
- "5" (Part B): 15 questions 
- "10" (Part B): 10 questions
- "15" (Part C): 5 questions

SYLLABUS:
{syllabus}

CORPUS:
{subject_corpus[:500000]}
""".strip()
        retry_prompt = prompt + "\\n\\nRETRY: You MUST return ONLY the JSON object with the expected structure."
        out = self._generate(prompt, retry_prompt=retry_prompt)
        if out.startswith("```json"): out = out[7:]
        if out.endswith("```"): out = out[:-3]

        parsed = _safe_json_loads(out.strip())
        if not isinstance(parsed, dict): return {{"1-2": [], "5": [], "10": [], "15": []}}

        def clean_section(key: str, default_marks: int) -> List[Dict[str, Any]]:
            arr = parsed.get(key, [])
            if not isinstance(arr, list): return []
            res = []
            for item in arr:
                 if not isinstance(item, dict): continue
                 q = str(item.get("q", "")).strip()
                 a = str(item.get("a", item.get("answer", ""))).strip()
                 marks = item.get("marks", default_marks)
                 mp = item.get("marking_points", [])
                 if not mp: mp = []
                 if isinstance(mp, str): mp = [mp]
                 if not q or not a: continue
                 res.append({{"q": q, "a": a, "marks": marks, "marking_points": [str(x) for x in mp]}})
            return res

        return {{
            "1-2": clean_section("1-2", 2),
            "5": clean_section("5", 5),
            "10": clean_section("10", 10),
            "15": clean_section("15", 15)
        }}
'''

t5_code = '''
    # ---------------- MEGA-PROMPTS (Subject-Level) ----------------
    
    def generate_subject_summary(self, subject_corpus: str, syllabus: str) -> str:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")

    def generate_subject_flashcards(self, subject_corpus: str, syllabus: str, n: int = 50) -> List[Dict[str, str]]:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")

    def generate_subject_questions_with_answers(self, subject_corpus: str, syllabus: str) -> Dict[str, List[Dict[str, Any]]]:
        raise NotImplementedError("T5 cannot handle 100k+ tokens for Mega-Prompts.")
'''

with open('app/providers/gemini_provider.py', 'a', encoding='utf-8') as f:
    f.write(gemini_code)

with open('app/providers/t5_provider.py', 'a', encoding='utf-8') as f:
    f.write(t5_code)
