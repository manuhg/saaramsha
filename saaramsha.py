from __future__ import print_function
from summarizer import summarizer
from flask_cors import CORS
import os
import sys

from flask import Flask
from flask import request

s = summarizer.summarizer()
s.load()

application = Flask(__name__)
CORS(application)

@application.route('/summarize',methods=['GET','POST'])
def summarize():
    print("hello1", request)
    if request.method == 'POST':
        print("hello2")
        f = request.form['document']
        return 'Helo'
    return 'ONLY POST REQUESTS ARE SUPPORTED!'


def exec_cmd(cmdstr, echo=True):
    print(os.popen(cmdstr).read() if echo else '')


def main():
    print('Running flask')
    #exec_cmd('export FLASK_APP=run_flask.py flask run &')
    application.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    main()
