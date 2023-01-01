# If the script is run directly, assume remote engine
if __name__ == "__main__":

    from engine.glados import GladosEngine
    from flask import Flask, request, send_file

    # Remote Engine Veritables
    PORT = 8124

    app = Flask(__name__)

    app.debug = True

    engine = GladosEngine(play_file=False)


    @app.route('/synthesize')
    def synthesize():
        text = request.args.get("text", default="", type=str)
        if text == '': return 'No input'

        file_path = engine.glados_tts(text)
        return send_file(file_path)


    app.run(host="0.0.0.0", port=PORT)
