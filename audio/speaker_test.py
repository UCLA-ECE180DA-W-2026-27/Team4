import pyttsx3

speaker = pyttsx3.init()

speaker.setProperty("volume", 1.0)  # Maximum volume
speaker.setProperty("rate", 150)    # Speaking speed

print("Testing speaker...")

speaker.say("Hello! Starting game.")
speaker.runAndWait()

print("Speech finished.")