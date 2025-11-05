import json
import requests

# Testing out the GET endpoint
r = requests.get("http://127.0.0.1:8000")

print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()["message"]}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Testing out the POST endpoint
r = requests.post(url="http://127.0.0.1:8000/data/", data=json.dumps(data))

print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()["result"]}")
