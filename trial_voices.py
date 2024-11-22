import pyttsx3
def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)
    for i in [18,29,19,30,140,87]:
        engine.setProperty('voice', voices[i].id)
        print(voices[i].name,'-',i)
        engine.say(audio)
        engine.runAndWait()
speak("This is my voice")
#these are the once i likes, 
# 18,29,19,30,140,87,140
# finalize either 87 or 140
# thats all 87 i.e Ralph is final