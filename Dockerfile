FROM python:3.7
WORKDIR /app
COPY . /app

ENV PIP_INDEX_URL https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

CMD [ "python","web_demo.py" ]