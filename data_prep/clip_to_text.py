import speech_recognition as sr
from scipy.io import wavfile
import glob

def get_file_list():
    for file in glob.glob("*.wav"):

        print('Found file: ', file)



        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            audio = r.record(source)

        # recognize speech using Sphinx
        try:
            print('converting ', file)
            text = r.recognize_sphinx(audio)
            text_fname = file[:-3]+'txt'
            print('output file name: ', text_fname)
            with open(text_fname, 'w') as output:
                output.write(text)
                print('success')
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))


if __name__ == "__main__":
    get_file_list()