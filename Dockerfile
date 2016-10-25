FROM python:latest

ADD src /code

WORKDIR /code

ENV PYTHONPATH $PYTHONPATH;/code/src

RUN apt-get update && apt-get install -y python3-dev libmysqlclient-dev python-matplotlib libmysqlclient-dev mysql-client

RUN pip install -r requirements/requirements.txt

ENTRYPOINT ["python"]
CMD ["src/webapp.py"]
