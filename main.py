from summarizer import summarizer
import os
import sys

from flask import Flask
from flask import request

app = Flask(__name__)
@app.route('/summarize')
def summarize():
    if request.method == 'POST':
        f = request.files['document']
        s = summarizer()
        s.load()
        return s.summarize()

    return 'ONLY POST REQUESTS ARE SUPPORTED!'


def exec_cmd(cmdstr, echo=True):
    print(os.popen(cmdstr).read() if echo else '')


def main():
    print('Running flask')
    #exec_cmd('export FLASK_APP=run_flask.py flask run &')
    app.run(host='0.0.0.0')


if __name__ == "__main__":
    main()
