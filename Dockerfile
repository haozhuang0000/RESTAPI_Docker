FROM python:3.11

WORKDIR /app
COPY . /app
ENV PYTHONPATH="/app"

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY SampleUsage.py ./SampleUsage.py
COPY api.py ./api.py

VOLUME ["/data/zhuanghao/model_cache"]

EXPOSE 9999
EXPOSE 9998
CMD ["sh","/app/serve.sh"]
