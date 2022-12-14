# How to run code:
# sudo docker build -t gp . && sudo docker run -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/python_workspace --name gp gp
# docker build -t gp . && docker run -it --rm --mount type=bind,source="C:\Users\gcis\git_workspace\gp_test",target=/usr/src/python_workspace --name gp gp

FROM python:3.9

WORKDIR /usr/src/python_workspace

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./parameteroptimisation.py"]
