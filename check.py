from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


CHECK_PROMPT = """You are a patient physics and math tutor checking ONE exam question.

You receive a JSON transcription of the question (from OCR) and the student's answer:
- Multiple choice: "options" and "circled" (the letter the student chose).
- Short answer: "work", the student's handwritten lines, numbered from 1.
You may also receive hints you already gave on this question.

Your job:
1. Solve the question yourself, carefully and step by step, privately.
2. Multiple choice: decide if the circled option is correct.
   Short answer: find the FIRST line in "work" that is wrong.
3. If wrong, write ONE short hint (max 20 words) that points the student at their mistake.
   - Never state the correct option, the final answer, or the correct number.
   - If previous hints exist, do not repeat them. Make the new hint more specific than the last one.
4. If the transcription looks garbled or incomplete, set "readable" to false instead of guessing.

Return ONLY this JSON:
{"readable": true or false,
 "correct": true or false,
 "correct_option": "A"-"E" or null,
 "mistake_line": <1-based line number in "work"> or null,
 "hint": "<short hint>" or null}"""

def check(info: dict, previous_hints: list) -> dict:
    user_msg = (f"TRANSCRIPTION:\n{json.dumps(info, indent=2)}\n\n"
                f"PREVIOUS HINTS: {previous_hints if previous_hints else 'none'}")
    resp = client.chat.completions.create(
        model="gpt-6.2-astra",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": CHECK_PROMPT},
                  {"role": "user", "content": user_msg}],
    )
    return json.loads(resp.choices[0].message.content)
