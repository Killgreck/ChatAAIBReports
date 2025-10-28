import numpy as np
from openai import AzureOpenAI

from recuperacion_consulta_faiss import search

client = AzureOpenAI(
    api_key="5FhrNYbHwABvRRYQ9kHzuJaAHtZRWJ0Ke1vpXGFfT0vFf7Wu6pqwJQQJ99BHACHYHv6XJ3w3AAABACOGvHwe",
    api_version='2024-12-01-preview',
    azure_endpoint='https://pnl-maestria.openai.azure.com/'
    )

query = 'how to lock a  G-TAWB'
res=search(query)
results = list(res['text'].values)

context = "\n\n".join(results)
prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"

# Nueva forma de hacer la petición
response = client.chat.completions.create(
    model="gpt-4.1-nano",# "o4-mini"
    messages=[
        {"role": "system", "content": "You are a expert in aircraft accidents, respond only in english"},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)



