import requests
import json
import numpy as np

queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

input = {'queries':queries}
 
response = requests.post('http://localhost:3000/predictions/sentence_xformer', data={'data':json.dumps(input)})
data = response.content

print(np.array(json.loads(data.decode("utf-8"))).shape)