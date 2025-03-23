import openai  
from my_secrets import OPEN_AI_KEY

class PromptClient:
    def __init__(self, api_key=OPEN_AI_KEY):
        self.api_key = api_key
        self.system_prompt = "You are a exercise advisor, particulaly at providing correction for people's posture and form for health. " \
        "Based on the given excercise make a suggestion of other exercises to do."
        openai.api_key = self.api_key

    def ask(self, user_input):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
