from openai import OpenAI
import base64

# Point the client at your LOCAL Lemonade server instead of OpenAI's servers
client = OpenAI(
    base_url="http://localhost:13305/api/v1",
    api_key="lemonade"  # required by the library, but Lemonade doesn't check it
)

def encode_image(path):
    """Read an image file from disk and convert it into a base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_path = "leaf.png"
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="Gemma-4-E2B-it-GGUF",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this frame a black screen, a frozen/stagnant shot, or otherwise 'dead space' with no meaningful visual action? Answer with exactly one word: YES or NO."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)