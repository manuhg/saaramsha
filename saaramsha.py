from summarizer import summarizer
from flask_cors import CORS
import os
import sys
from flask import jsonify
from flask import Flask
from flask import request

s = summarizer.summarizer()
s.load()

application = Flask(__name__)
CORS(application)

@application.route('/summarize',methods=['GET','POST'])
def summarize():
    if request.method == 'POST':
        f = request.form['document']
        #enc = None request.files.get('encoder_name')
        encoder_name = 'USE' #if enc is None else enc
        result = s.summarize(f,encoder_name)
        dict = {'data': result}
        return jsonify(dict);
    return 'ONLY POST REQUESTS ARE SUPPORTED!'


def exec_cmd(cmdstr, echo=True):
    print(os.popen(cmdstr).read() if echo else '')


def main():
    print('Running flask')
    #exec_cmd('export FLASK_APP=run_flask.py flask run &')
    application.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    main()