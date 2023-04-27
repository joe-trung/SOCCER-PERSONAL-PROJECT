FROM python

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install awscli


COPY . /

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]

