"""
app

flask app serving external client calls.
"""
from random import randint
from uuid import uuid4

from flask import Flask
from flask import request, abort, send_file

from helper import _validate_integer
from serving import feed_serving


app = Flask("mnist_gan")


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        digit = randint(0, 9)
    elif request.method == 'POST':
        raw = request.get_data()

        if not request.data:
            abort(400)

        digit = _validate_integer(raw)

        if digit is None:
            abort(400)

    rv = feed_serving(digit)

    return send_file(rv,
                     as_attachment=False,
                     attachment_filename=f'{uuid4()}.png',
                     mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
