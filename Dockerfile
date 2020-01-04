FROM python:3.6-onbuild

CMD ["uwsgi", "--yaml", "app.yaml"]
