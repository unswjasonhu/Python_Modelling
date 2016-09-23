FROM python:latest

ADD src /code

WORKDIR /code

RUN export PYTHONPATH=/code/src && echo "python path is" $PYTHONPATH

RUN apt-get update

RUN apt-get install -y python3-dev libmysqlclient-dev python-matplotlib libmysqlclient-dev

RUN pip install -r requirements/requirements.txt

ENTRYPOINT ["python"]
CMD ["src/webapp.py"]
