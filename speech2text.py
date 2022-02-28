import speech_recognition as sr

def recordSpeech():
    # Initialize the recognizer 
    r = sr.Recognizer() 

    while True:
        try:
            # use the microphone as source for input.
            with sr.Microphone() as source:
                    
                # wait for a second to let the recognizer adjust the energy threshold based on the surrounding noise level 
                r.adjust_for_ambient_noise(source, duration=0.2)
                print("Recognizing...")

                #listens for the user's input 
                audio = r.listen(source)

                # Using ggogle to recognize audio
                MyText = r.recognize_google(audio)
                MyText = MyText.lower()
                # print("Your command: " + MyText)
                return MyText

        except sr.UnknownValueError:
            print('error') 
            


if __name__ == '__main__':
    speech = recordSpeech()
    print("Your command: " + speech)