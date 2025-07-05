import ollama

class LLM:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def generate_answer(self, query, retrieved, web_results):
        system_prompt = (
            "You are a helpful medical safety assistant. Always include disclaimers. "
            "Keep answers under 250 words. "
            "Always list first-aid steps clearly on separate lines as bullet points or numbers, "
            "and do not embed them inside paragraphs."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""⚠️ DISCLAIMER: This information is for educational purposes only and not a substitute for professional medical advice.

The user asked:
'{query}'

Relevant local knowledge:
{chr(10).join(retrieved)}

Relevant web evidence:
{chr(10).join(web_results)}

Please provide a clear, concise, medically appropriate first-aid answer including:
- the likely condition
- first-aid steps
- key medicine(s)
- references to the above sources
in fewer than 250 words.
"""}
            }
        ]
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content']
