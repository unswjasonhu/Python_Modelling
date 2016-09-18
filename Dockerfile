FROM python:latest

ADD app /code

WORKDIR /code

RUN apt-get update

RUN apt-get install -y python3-dev libmysqlclient-dev python-matplotlib libmysqlclient-dev

RUN pip install -r requirements/requirements.txt

ENTRYPOINT ["python"]
CMD ["app/webapp.py"]
