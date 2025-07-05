import requests

class WebSearch:
    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, query, top_k=3):
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        data = {"q": query}
        response = requests.post(url, headers=headers, json=data)
        results = response.json()
        organic = results.get("organic", [])[:top_k]
        evidence = []
        for res in organic:
            evidence.append(f"{res['title']}: {res['snippet']} (source: {res['link']})")
        return evidence
