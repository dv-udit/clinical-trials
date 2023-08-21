FROM python

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV NAME OPENAI_API_KEY

CMD [ "python" , "app.py"]