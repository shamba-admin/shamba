## Shamba Server

### Enable Python Version

```sh
cd shamba
poetry install
poetry env use 3.10
eval $(poetry env activate)
```

### Run Server

Run the server in _development_ mode: `poetry run uvicorn main:app --reload`
