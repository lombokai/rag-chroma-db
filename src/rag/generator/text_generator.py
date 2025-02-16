from openai import OpenAI


class TextGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, question: str, relevant_chunks: list):
        context = "\n".join(relevant_chunks)

        prompt = (
            "Kamu adalah customer service bot. gunakan context dibawah ini "
            "ambil context dari database. jawab menggunakan bahasa indonesia"
            "\n\nContext:\n" + context + "\n\npertanyaan:\n" + question
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message
        return answer
