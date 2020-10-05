import os
from server import app
from config import HOST, PORT, MODELS

if __name__ == '__main__':
    if not os.path.exists(MODELS):
        os.makedirs(MODELS)

    app.run(
        host=HOST,
        port=PORT
    )