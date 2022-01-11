import requests
import json
import pandas as pd

query = """query { 
  viewer { 
    login
  }
}"""


access_token = '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'
headers = {'Authorization': 'bearer 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'}
url = 'https://api.github.com/graphql'
r = requests.post(url, json={'query': query}, headers=headers)

print()
