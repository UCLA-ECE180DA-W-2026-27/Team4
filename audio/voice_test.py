import speech_recognition as sr
import pyttsx3

# -------------------------
# Setup
# -------------------------

recognizer = sr.Recognizer()
speaker = pyttsx3.init()

# Use Windows default microphone
mic = sr.Microphone(device_index=0)


def speak(text):
    print("GAME:", text)
    speaker.say(text)
    speaker.runAndWait()


def process_command(command):
    command = command.lower()

    if "restart" in command:
        print("ACTION: RESTART GAME")
        speak("Restarting game")

    elif "pause" in command:
        print("ACTION: PAUSE GAME")
        speak("Game paused")

    elif "start" in command:
        print("ACTION: START GAME")
        speak("Starting game")

    elif "menu" in command:
        print("ACTION: GO TO MENU")
        speak("Returning to menu")

    else:
        print("ACTION: UNKNOWN COMMAND")
        speak("Command not recognized")


# -------------------------
# Listen
# -------------------------

try:
    with mic as source:

        print("Calibrating...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Listening...")
        print("Say: START, PAUSE, RESTART, or MENU")

        audio = recognizer.listen(
            source,
            timeout=10,
            phrase_time_limit=3
        )

    # Speech -> text
    command = recognizer.recognize_google(audio)

    print("I heard:", command)

    # Text -> action
    process_command(command)


except sr.WaitTimeoutError:
    print("No speech detected.")

except sr.UnknownValueError:
    print("Could not understand speech.")

except sr.RequestError as e:
    print("Speech recognition error:", e)