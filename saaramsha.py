from __future__ import print_function
from summarizer import summarizer
import os
import sys

from flask import Flask
from flask import request

s = summarizer()
s.load()

application = Flask(__name__)
@application.route('/summarize')
def summarize():
    if request.method == 'POST':
        f = request.files['document']
        enc = request.files.get('encoder_name')
        encoder_name = 'USE' if enc is None else enc
        return s.summarize(f,encoder_name)

    return 'ONLY POST REQUESTS ARE SUPPORTED!'


def exec_cmd(cmdstr, echo=True):
    print(os.popen(cmdstr).read() if echo else '')


def main():
    print('Running flask')
    #exec_cmd('export FLASK_APP=run_flask.py flask run &')
    application.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    main()
