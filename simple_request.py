from requests import get
from json import loads
  
URL = "http://2cc6f0e1.ngrok.io"

r= get(URL, stream=True)

for chunk in r.iter_lines(chunk_size=1024):
        data = loads(chunk) # Dictionarie
        print(data) 