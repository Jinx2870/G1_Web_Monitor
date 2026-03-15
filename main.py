import atexit

from flask import Flask, Response, jsonify, render_template

from core.camera_manager import G1CameraManager
from core.utils import load_config


config = load_config("config.yaml")
app = Flask(__name__)
camera_manager = G1CameraManager(config["camera"]).start()


@app.route("/")
def index():
    return render_template("index.html")


def gen_stream(stream_name):
    while True:
        frame = camera_manager.get_encoded_frame(stream_name)
        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/stream/<stream_name>")
def stream_feed(stream_name):
    return Response(
        gen_stream(stream_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed")
def video_feed():
    return stream_feed("depth")


@app.route("/api/status")
def camera_status():
    return jsonify(camera_manager.get_status())


atexit.register(camera_manager.stop)


if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"],
        threaded=True,
    )