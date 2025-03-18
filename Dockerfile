FROM python:3.10-slim

#following sample docker file here: https://www.quora.com/Is-it-better-to-pip-install-packages-in-the-Dockerfile-directly-or-using-a-requirements-txt-when-not-wanting-to-rebuild-a-container-unnecessarily

#default directory for commands
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

# copy executables to path
#COPY . .
COPY training.py .
COPY test.py . 

#RUN chmod +x  *.sh

#CMD ["python", "training.py"]
CMD ["python", "test.py"]


