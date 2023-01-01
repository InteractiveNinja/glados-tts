from engine.glados import GladosEngine

engine = GladosEngine(True, False)

while True:
    text = input("Input: ")
    engine.glados_tts(text)