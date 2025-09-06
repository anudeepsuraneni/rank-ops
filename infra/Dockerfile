# use slim Python 3.11 base image
FROM python:3.11-slim 

# set working directory
WORKDIR /app 

# copy dependency files
COPY pyproject.toml poetry.lock ./ 

# install poetry and project dependencies (no dev deps)
RUN pip install poetry && poetry install --no-dev 

# copy application code
COPY . . 

# run app with uvicorn on port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
