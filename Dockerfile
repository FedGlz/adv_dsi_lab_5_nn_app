FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./models /code/models

COPY ./src /code/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]